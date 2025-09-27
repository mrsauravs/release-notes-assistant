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

# --- Hardcoded URL for the Knowledge Base ---
KNOWLEDGE_BASE_URL = "https://raw.githubusercontent.com/mrsauravs/release-notes-assistant/refs/heads/main/release_knowledge_base.json"

@st.cache_data(ttl=3600)
def load_knowledge_base(url):
    """Fetches and loads a JSON knowledge base from a hardcoded URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        st.error(f"Fatal Error: Could not fetch knowledge base. Please check the URL in the code.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Fatal Error: Could not decode the knowledge base. Please ensure the file at the URL is valid JSON.")
        st.stop()

def build_classifier_prompt(engineering_note):
    """Builds a prompt to classify a note as PUBLIC or INTERNAL."""
    # Using a subset of the note for brevity and cost-effectiveness
    triage_data = {
        "Summary": engineering_note.get("Summary", ""),
        "Issue Type": engineering_note.get("Issue Type", ""),
        "Description": (engineering_note.get("Description", "") or "")[:300] # Truncate for efficiency
    }
    
    return f"""
    You are an expert Release Manager at an enterprise software company. Analyze the following engineering ticket data to determine if it describes a customer-facing change or an internal-only task.

    - **PUBLIC** changes are new features, enhancements, or bug fixes that a customer would notice or benefit from. They affect the user interface, functionality, performance, or API.
    - **INTERNAL** changes are tasks like refactoring code, updating internal libraries, database migrations, or technical debt with no direct, observable impact on the customer.

    Look for clues:
    - User-facing benefits in the summary/description suggest PUBLIC.
    - Terms like 'refactor', 'tech debt', 'internal testing', 'update dependency' suggest INTERNAL.
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
        task_instruction = f"**Task:**\nThe engineering note describes a bug fix. Write the final, polished release note using this exact format:\n`{style_guide['bug_fix_writing']['format']}`"
    else:
        task_instruction = f"**Task:**\nThe engineering note describes a new feature or enhancement. Write the release note following this instruction:\n\"{style_guide['feature_enhancement_writing']['instruction']}\""
    
    prompt = f"""
    You are a Principal Technical Writer at Alation. Convert the raw engineering note into a formal, customer-facing release note.

    **CRITICAL RULE: SANITIZE THE OUTPUT**
    Remove all internal details like process IDs (e.g., 'pid 1'), internal server names, or non-public feature names.

    **Crucial Instruction:**
    At the end of the generated note, you MUST append the Jira Key, enclosed in parentheses. Example: `(AL-12345)`.

    **Raw Engineering Note:**
    ```json
    {json.dumps(engineering_note, indent=2)}
    ```
    
    {task_instruction}
    """
    return prompt

# --- Main Application Logic ---
st.title("Intelligent Release Notes Assistant 🚀")

# Initialize session state
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
if 'summary_data' not in st.session_state:
    st.session_state.summary_data = None

release_kb = load_knowledge_base(KNOWLEDGE_BASE_URL)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Release notes style guide is loaded automatically.")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    release_version = st.text_input("Enter Release Version (e.g., 2025.3.1)", "2025.3.1")

st.header("Step 1: Upload Your Content")
uploaded_csv = st.file_uploader(
    "Upload your engineering notes CSV file",
    type="csv"
)

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
            
            progress_bar = st.progress(0, text="Classifying notes...")
            total_rows = len(df)

            for index, row in df.iterrows():
                engineering_note = row.to_dict()
                summary = engineering_note.get('Summary', 'N/A')
                jira_key = engineering_note.get('Issue key', 'N/A')

                progress_text = f"Classifying '{summary}'..."
                progress_bar.progress((index + 1) / total_rows, text=progress_text)
                
                # --- NEW: AI Classifier Step ---
                classifier_prompt = build_classifier_prompt(engineering_note)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": classifier_prompt}],
                        max_tokens=5,
                        temperature=0
                    )
                    classification = response.choices[0].message.content.strip().upper()
                except Exception as e:
                    classification = "INTERNAL" # Default to skipping on error
                    st.warning(f"Could not classify '{summary}': {e}")
                
                if "PUBLIC" in classification:
                    # --- AI Writer Step ---
                    with st.spinner(f"Writing note for '{summary}'..."):
                        writer_prompt = build_release_prompt(release_kb, engineering_note)
                        try:
                            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": writer_prompt}])
                            suggestion = response.choices[0].message.content.strip()
                            processed_notes.append((jira_key, summary, suggestion))
                        except Exception as e:
                            skipped_notes.append((jira_key, summary, f"AI writer failed: {e}"))
                else:
                    skipped_notes.append((jira_key, summary, "Classified as Internal"))

            progress_bar.progress(1.0, text="Assembling final document...")

            # --- NEW: Assemble the final report string and summary ---
            results = {"New Features": [], "Enhancements": [], "Bug Fixes": []}
            for jira_key, summary, suggestion in processed_notes:
                # Simple categorization for assembly
                if "bug" in summary.lower() or "fix" in summary.lower():
                     results["Bug Fixes"].append(suggestion)
                else: # Assume Feature/Enhancement
                     results["New Features"].append(suggestion)

            month_year = datetime.now().strftime('%B %Y')
            report_parts = [f"# Release {release_version}", f"_{month_year}_"]
            for section, notes in results.items():
                if notes:
                    report_parts.append(f"\n\n**{section}**\n")
                    if section == "Bug Fixes":
                        report_parts.append("\n".join(notes))
                    else:
                        report_parts.append("\n\n".join(notes))
            
            st.session_state.final_report = "\n".join(report_parts)
            st.session_state.summary_data = {
                "total": total_rows,
                "processed_count": len(processed_notes),
                "skipped_count": len(skipped_notes),
                "processed_list": processed_notes,
                "skipped_list": skipped_notes
            }
            st.success("✅ Release notes document generated successfully!")

# --- Display Results and Download ---
if st.session_state.summary_data:
    summary = st.session_state.summary_data
    st.info(f"**Processing Summary:** {summary['processed_count']} notes generated, {summary['skipped_count']} notes skipped.")
    
    with st.expander("Show Detailed Processing Report"):
        st.markdown("#### ✅ Processed Notes")
        for jira_key, summary_text, _ in summary['processed_list']:
            st.text(f"- {jira_key}: {summary_text}")
            
        st.markdown("####  skipped_notes")
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
