import streamlit as st
import pandas as pd
import json
import requests
import openai
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Release Notes Assistant 🚀",
    page_icon="📝",
    layout="wide"
)

# --- Embedded Knowledge Base ---
RELEASE_KNOWLEDGE_BASE = {
  "release_structure": {
    "section_order": [ "New Features", "Enhancements", "Bug Fixes" ]
  },
  "writing_style_guide": {
    "feature_enhancement_writing": {
      "instruction": "First, create a short, descriptive title for the update from the 'Summary'. Do not include the Jira key in the title itself. The title should be bolded. On the next line, write a clear, benefit-oriented paragraph describing the update. At the very end of the paragraph, add the Jira key in parentheses.",
      "example_format": "**This is a Clean Feature Title**\\n\\nA descriptive paragraph about the feature, explaining its value to the user. (JIRA-KEY-123)"
    },
    "bug_fix_writing": {
      "instruction": "Generate a single, concise sentence for a bullet point. Start with 'Fixed an issue where...'. At the very end of the sentence, add the Jira key in parentheses.",
      "format": "* Fixed an issue where [description of the problem]. ([JIRA-KEY-123])"
    }
  },
  "key_product_areas": [
    "Open Connector Framework (OCF)", "Alation Analytics", "Data Governance", "Compose",
    "Catalog Experience", "Server Administration", "Data Products & Marketplace", "API & Integrations"
  ]
}

# --- Helper functions ---
def find_column_by_substring(df, substring):
    substring = substring.lower()
    for col in df.columns:
        if substring in col.lower():
            return col
    return None

def build_classifier_prompt(engineering_note):
    """Builds a prompt to classify a note as PUBLIC or INTERNAL."""
    triage_data = {
        "Summary": engineering_note.get("Summary", ""),
        "Issue Type": engineering_note.get("Issue Type", ""),
        "Description": (engineering_note.get("Description", "") or "")[:300]
    }
    
    return f"""
    You are an expert Release Manager at an enterprise software company. Analyze the following engineering ticket data to determine if it describes a customer-facing change or an internal-only task.

    - **PUBLIC** changes are new features, enhancements, or bug fixes that a customer would notice or benefit from. They affect the user interface, functionality, performance, or API. Examples: "Ability to Filter by ‘No Steward’", "Improve Search Performance".
    - **INTERNAL** changes are tasks like refactoring code, updating internal libraries, database migrations, or fixing unit tests with no direct, observable impact on the customer. Examples: "Fix the Unit Test Failures in Master Branch", "Refactor backend authentication service".

    Look for clues:
    - User-facing benefits in the summary/description suggest PUBLIC.
    - Terms like 'refactor', 'tech debt', 'internal testing', 'update dependency', 'unit test' suggest INTERNAL.
    - 'Bug', 'Story', 'Enhancement' are usually PUBLIC. 'Task', 'Spike', 'Sub-task' are usually INTERNAL.

    **Ticket Data:**
    ```json
    {json.dumps(triage_data, indent=2)}
    ```

    Is this change PUBLIC or INTERNAL? Your response must be a single word: PUBLIC or INTERNAL.
    """

def build_release_prompt(knowledge_base, engineering_note):
    """Dynamically builds a prompt to write the release note."""
    style_guide = knowledge_base['writing_style_guide']
    issue_type = engineering_note.get("Issue Type", "Feature").lower()

    if "bug" in issue_type or "defect" in issue_type:
        task_instruction = f"**Task:**\nThe engineering note describes a bug fix. Write a single sentence for a Markdown bullet point using this exact format:\n`{style_guide['bug_fix_writing']['format']}`"
    else:
        task_instruction = f"**Task:**\nThe engineering note describes a new feature or enhancement. Write the release note following this instruction:\n\"{style_guide['feature_enhancement_writing']['instruction']}\""
    
    prompt = f"""
    You are a Principal Technical Writer at Alation. Convert the raw engineering note into a formal, customer-facing release note.
    **CRITICAL RULE:** Remove all internal jargon (e.g., 'pid 1', 'master branch').
    **Crucial Instruction:** At the end of the note, you MUST append the Jira Key, enclosed in parentheses. Example: `(AL-12345)`.

    **Raw Engineering Note:**
    ```json
    {json.dumps(engineering_note, indent=2)}
    ```
    
    {task_instruction}
    """
    return prompt

# --- Main Application Logic ---
st.title("Intelligent Release Notes Assistant 🚀")

if 'final_report' not in st.session_state: st.session_state.final_report = None
if 'summary_data' not in st.session_state: st.session_state.summary_data = None

with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("The release notes style guide is embedded in the application.")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    release_version = st.text_input("Enter Release Version (e.g., 2025.3.1)", "2025.3.1")

st.header("Step 1: Upload Your Content")
uploaded_csv = st.file_uploader("Upload your engineering notes CSV file", type="csv")

st.header("Step 2: Generate Notes")
if uploaded_csv:
    if st.button("📝 Generate Release Notes Document"):
        st.session_state.final_report = None
        st.session_state.summary_data = None
        
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            df = pd.read_csv(uploaded_csv).fillna('')
            st.subheader("Processing Notes...")
            
            client = openai.OpenAI(api_key=api_key)
            processed_notes = []
            skipped_notes = []
            
            progress_bar = st.progress(0, text="Initializing...")
            total_rows = len(df)

            for index, row in df.iterrows():
                engineering_note = row.to_dict()
                summary = engineering_note.get('Summary', 'N/A')
                jira_key = engineering_note.get('Issue key', 'N/A')

                progress_text = f"Classifying '{summary[:40]}...'"
                progress_bar.progress((index + 1) / total_rows, text=progress_text)
                
                # --- AI Classifier Step ---
                classifier_prompt = build_classifier_prompt(engineering_note)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": classifier_prompt}],
                        max_tokens=5, temperature=0
                    )
                    classification = response.choices[0].message.content.strip().upper()
                except Exception:
                    classification = "INTERNAL" # Default to skipping on error
                
                if "PUBLIC" in classification:
                    # --- AI Writer Step ---
                    writer_prompt = build_release_prompt(RELEASE_KNOWLEDGE_BASE, engineering_note)
                    try:
                        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": writer_prompt}])
                        suggestion = response.choices[0].message.content.strip()
                        processed_notes.append((engineering_note, suggestion))
                    except Exception as e:
                        skipped_notes.append((jira_key, summary, f"AI writer failed: {e}"))
                else:
                    skipped_notes.append((jira_key, summary, "Classified as Internal"))

            progress_bar.progress(1.0, text="Assembling final document...")

            # --- Assemble the final report string and summary ---
            results = {"New Features": [], "Enhancements": [], "Bug Fixes": []}
            for eng_note, suggestion in processed_notes:
                issue_type = eng_note.get("Issue Type", "Feature").lower()
                if "bug" in issue_type or "defect" in issue_type:
                     results["Bug Fixes"].append(suggestion)
                elif "enhancement" in issue_type:
                    results["Enhancements"].append(suggestion)
                else:
                    results["New Features"].append(suggestion)

            month_year = datetime.now().strftime('%B %Y')
            report_parts = [f"# Release {release_version}", f"_{month_year}_"]
            for section in RELEASE_KNOWLEDGE_BASE['release_structure']['section_order']:
                if results.get(section):
                    report_parts.append(f"\n\n**{section}**\n")
                    if section == "Bug Fixes":
                        report_parts.append("\n".join(results[section]))
                    else:
                        report_parts.append("\n\n".join(results[section]))
            
            st.session_state.final_report = "\n".join(report_parts)
            st.session_state.summary_data = {
                "total": total_rows,
                "processed_count": len(processed_notes),
                "skipped_count": len(skipped_notes),
                "processed_list": [(note.get('Issue key'), note.get('Summary')) for note, _ in processed_notes],
                "skipped_list": skipped_notes
            }
            st.success("✅ Release notes document generated successfully!")

if st.session_state.summary_data:
    summary = st.session_state.summary_data
    st.info(f"**Processing Summary:** {summary['processed_count']} notes generated, {summary['skipped_count']} notes skipped.")
    
    with st.expander("Show Detailed Processing Report"):
        st.markdown("#### ✅ Notes Included in Document")
        for jira_key, summary_text in summary['processed_list']:
            st.text(f"- {jira_key}: {summary_text}")
            
        st.markdown("#### ⏩ Notes Skipped")
        for jira_key, summary_text, reason in summary['skipped_list']:
            st.text(f"- {jira_key}: {summary_text} (Reason: {reason})")

if st.session_state.final_report:
    st.header("Step 3: Download Report")
    st.markdown("### Preview")
    st.markdown(st.session_state.final_report, unsafe_allow_html=True)
    
    st.download_button(
        label="📥 Download Release Notes (.md)",
        data=st.session_state.final_report.encode('utf-8'),
        file_name=f"Release_Notes_{release_version}.md",
        mime="text/markdown",
    )
