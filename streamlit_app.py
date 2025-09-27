import streamlit as st
import pandas as pd
import json
import openai
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Release Notes Assistant 🚀",
    page_icon="📝",
    layout="wide"
)

# --- Embedded Knowledge Base (Style Guide) ---
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
  }
}

# --- Helper functions ---
def find_column_by_substring(df, substring):
    substring = substring.lower()
    for col in df.columns:
        if substring in col.lower():
            return col
    return None

def build_classifier_prompt(engineering_note):
    """Builds a more nuanced prompt to classify a note as PUBLIC or INTERNAL."""
    triage_data = {
        "Summary": engineering_note.get("Summary", ""),
        "Issue Type": engineering_note.get("Issue Type", ""),
        "Description": (engineering_note.get("Description", "") or "")[:400]
    }
    
    return f"""
    You are a discerning Principal Release Manager. Your goal is to accurately identify which engineering tasks represent customer-facing changes.

    **Analysis Framework:**
    Your decision must be based on the **ultimate outcome** for the end-user. Backend work or refactoring can still be part of a PUBLIC feature if the result is a new capability, a noticeable performance improvement, or a fixed bug.

    **How to Weigh Evidence:**
    1.  **Prioritize the 'Summary':** The summary often describes the user-facing goal. If it implies a new capability, lean towards PUBLIC.
    2.  **Analyze the 'Description' for Outcome:** Look past engineering keywords like 'refactor' or 'migrate'. Find the *reason* for the work. Does the refactor enable a new UI component? Does the migration improve speed? If so, it's likely PUBLIC.
    3.  **Do not over-rely on keywords:** A task like "Refactor Search Indexing" is PUBLIC if it makes search faster. A task like "Update Frontend Dependencies" is PUBLIC if it patches a security vulnerability.

    **Task:**
    Analyze the ticket data below. If the ultimate result is a direct, observable change or benefit for the customer, classify it as **PUBLIC**. Otherwise, classify it as **INTERNAL**.

    **Ticket Data to Classify:**
    ```json
    {json.dumps(triage_data, indent=2)}
    ```

    Your response must be a single word: PUBLIC or INTERNAL.
    """

def build_release_prompt(knowledge_base, engineering_note):
    """Dynamically builds a prompt to write the release note."""
    style_guide = knowledge_base['writing_style_guide']
    issue_type = engineering_note.get("Issue Type", "Feature").lower()

    if "bug" in issue_type or "escalation" in issue_type:
        task_instruction = f"**Task:**\nThe engineering note describes a bug fix. Write a single sentence for a Markdown bullet point using this exact format:\n`{style_guide['bug_fix_writing']['format']}`"
    else:
        task_instruction = f"**Task:**\nThe engineering note describes a new feature or enhancement. Write the release note following this instruction:\n\"{style_guide['feature_enhancement_writing']['instruction']}\""
    
    prompt = f"""
    You are a Principal Technical Writer at Alation. Convert the raw engineering note into a formal, customer-facing release note.
    **CRITICAL RULE:** Remove all internal jargon (e.g., 'pid 1', 'master branch').
    **Crucial Instruction:** At the end of the note, you MUST append the 'Key' (the Jira key), enclosed in parentheses. Example: `(AL-12345)`.

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

st.header("Step 1: Upload Your Content Files")
col1, col2, col3, col4 = st.columns(4)
with col1:
    epics_csv = st.file_uploader("1. Epics", type="csv")
with col2:
    stories_csv = st.file_uploader("2. Stories", type="csv")
with col3:
    bugs_csv = st.file_uploader("3. Bug Fixes", type="csv")
with col4:
    escalations_csv = st.file_uploader("4. Support Escalations", type="csv")

st.header("Step 2: Generate Notes")
if st.button("📝 Generate Release Notes Document"):
    st.session_state.final_report = None
    st.session_state.summary_data = None
    
    if not all([epics_csv, stories_csv, bugs_csv, escalations_csv]):
        st.error("Please upload all four CSV files to proceed.")
    elif not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        # --- Data Loading and Processing ---
        df_epics = pd.read_csv(epics_csv).fillna('')
        df_stories = pd.read_csv(stories_csv).fillna('')
        df_bugs = pd.read_csv(bugs_csv).fillna('')
        df_escalations = pd.read_csv(escalations_csv).fillna('')
        
        client = openai.OpenAI(api_key=api_key)
        processed_features = []
        processed_bugs = []
        skipped_items = []
        public_epic_keys = set()

        st.subheader("Processing Notes...")
        all_dfs = { "Epics": df_epics, "Stories": df_stories, "Bugs": df_bugs, "Escalations": df_escalations }
        progress_bar = st.progress(0, text="Initializing...")
        total_rows = sum(len(df) for df in all_dfs.values())
        processed_rows = 0

        # Process Epics First
        for index, row in df_epics.iterrows():
            processed_rows += 1
            progress_bar.progress(processed_rows / total_rows, text=f"Classifying Epic: {row.get('Summary', '')[:30]}...")
            eng_note = row.to_dict()
            classifier_prompt = build_classifier_prompt(eng_note)
            try:
                response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": classifier_prompt}], max_tokens=5, temperature=0)
                classification = response.choices[0].message.content.strip().upper()
                if "PUBLIC" in classification:
                    public_epic_keys.add(eng_note['Key'])
                    processed_features.append(eng_note)
                else:
                    skipped_items.append((eng_note.get('Key'), eng_note.get('Summary'), "Classified as Internal"))
            except Exception as e:
                skipped_items.append((eng_note.get('Key'), eng_note.get('Summary'), f"Classifier failed: {e}"))

        # Process Stories, checking against parent Epics
        for index, row in df_stories.iterrows():
            processed_rows += 1
            progress_bar.progress(processed_rows / total_rows, text=f"Classifying Story: {row.get('Summary', '')[:30]}...")
            eng_note = row.to_dict()
            if eng_note.get('parent') in public_epic_keys:
                skipped_items.append((eng_note.get('Key'), eng_note.get('Summary'), "Skipped (Parent Epic is public)"))
                continue
            
            classifier_prompt = build_classifier_prompt(eng_note)
            try:
                response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": classifier_prompt}], max_tokens=5, temperature=0)
                classification = response.choices[0].message.content.strip().upper()
                if "PUBLIC" in classification:
                    processed_features.append(eng_note)
                else:
                    skipped_items.append((eng_note.get('Key'), eng_note.get('Summary'), "Classified as Internal"))
            except Exception as e:
                skipped_items.append((eng_note.get('Key'), eng_note.get('Summary'), f"Classifier failed: {e}"))

        # Process Bugs and Escalations
        df_all_bugs = pd.concat([df_bugs, df_escalations], ignore_index=True)
        for index, row in df_all_bugs.iterrows():
            processed_rows += 1
            progress_bar.progress(processed_rows / total_rows, text=f"Classifying Bug: {row.get('Summary', '')[:30]}...")
            eng_note = row.to_dict()
            classifier_prompt = build_classifier_prompt(eng_note)
            try:
                response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": classifier_prompt}], max_tokens=5, temperature=0)
                classification = response.choices[0].message.content.strip().upper()
                if "PUBLIC" in classification:
                    processed_bugs.append(eng_note)
                else:
                    skipped_items.append((eng_note.get('Key'), eng_note.get('Summary'), "Classified as Internal"))
            except Exception as e:
                skipped_items.append((eng_note.get('Key'), eng_note.get('Summary'), f"Classifier failed: {e}"))

        # AI WRITING STAGE
        final_results = {"New Features": [], "Enhancements": [], "Bug Fixes": []}
        
        for note in processed_features:
            prompt = build_release_prompt(RELEASE_KNOWLEDGE_BASE, note)
            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            final_results["New Features"].append(response.choices[0].message.content.strip())
        
        for note in processed_bugs:
            prompt = build_release_prompt(RELEASE_KNOWLEDGE_BASE, note)
            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            final_results["Bug Fixes"].append(response.choices[0].message.content.strip())

        # Assemble Final Document
        progress_bar.progress(1.0, text="Assembling final document...")
        month_year = datetime.now().strftime('%B %Y')
        report_parts = [f"# Release {release_version}", f"_{month_year}_"]
        for section in RELEASE_KNOWLEDGE_BASE['release_structure']['section_order']:
            if final_results.get(section):
                report_parts.append(f"\n\n**{section}**\n")
                if section == "Bug Fixes":
                    report_parts.append("\n".join(final_results[section]))
                else:
                    report_parts.append("\n\n".join(final_results[section]))
        
        st.session_state.final_report = "\n".join(report_parts)
        st.session_state.summary_data = {
            "total": total_rows,
            "processed_count": len(processed_features) + len(processed_bugs),
            "skipped_count": len(skipped_items),
            "processed_list": [(note.get('Key'), note.get('Summary')) for note in processed_features] + [(note.get('Key'), note.get('Summary')) for note in processed_bugs],
            "skipped_list": skipped_items
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
