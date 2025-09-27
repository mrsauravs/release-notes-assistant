import streamlit as st
import pandas as pd
import json
import requests
import openai 
# Import other LLM libraries as needed

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Release Notes Assistant 🚀",
    page_icon="📝",
    layout="wide"
)

# --- Hardcoded URL for the Knowledge Base ---
KNOWLEDGE_BASE_URL = "https://raw.githubusercontent.com/mrsauravs/release-notes-assistant/refs/heads/main/release_knowledge_base.json"

# --- Helper function to find the release notes column ---
def find_release_notes_column(df):
    """Dynamically finds the column containing 'Release Notes', case-insensitive."""
    for col in df.columns:
        if 'release notes' in col.lower():
            return col
    return None

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

def build_release_prompt(knowledge_base, engineering_note):
    """Dynamically builds a prompt using the release knowledge base."""
    style_guide = knowledge_base['writing_style_guide']
    issue_type = engineering_note.get("Issue Type", "Feature").lower()
    
    if "bug" in issue_type or "defect" in issue_type:
        task_instruction = f"""
        **Task:**
        The engineering note describes a bug fix. Write the final, polished release note using this exact format:
        `{style_guide['bug_fix_writing']['format']}`
        """
    elif "enhancement" in issue_type:
        task_instruction = f"""
        **Task:**
        The engineering note describes an enhancement. Write the final, polished release note using this exact format:
        `{style_guide['enhancement_writing']['format']}`
        """
    else: # Default to feature
        task_instruction = f"""
        **Task:**
        The engineering note describes a new feature. Write the final, polished release note following this instruction:
        "{style_guide['feature_writing']['instruction']}"
        - Example Format: {style_guide['feature_writing']['example_format']}
        """

    prompt = f"""
    You are a Principal Technical Writer at Alation. Your task is to convert a raw engineering note into a formal, customer-facing release note.

    ---
    **CRITICAL RULE: SANITIZE THE OUTPUT**
    The raw engineering note may contain internal jargon, project codenames, or technical identifiers. You MUST rewrite the content to be fully customer-facing. **Remove all internal details like process IDs (e.g., 'pid 1'), internal server names, or non-public feature names.** The final output must only contain language that an external customer would understand.
    ---

    **Crucial Instruction:**
    At the end of the generated note, you MUST append the Jira Key provided in the raw input, enclosed in parentheses. For example: `(AL-12345)`.

    **Raw Engineering Note:**
    ```json
    {json.dumps(engineering_note, indent=2)}
    ```
    
    {task_instruction}
    """
    return prompt

# --- Main Application Logic ---
st.title("Intelligent Release Notes Assistant 🚀")

release_kb = load_knowledge_base(KNOWLEDGE_BASE_URL)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Release notes style guide is loaded automatically.")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")

st.header("Step 1: Upload Your Content")
uploaded_csv = st.file_uploader(
    "Upload your engineering notes CSV file",
    type="csv",
    help="CSV must contain columns like 'Summary', 'Issue key', and a 'Release Notes' column."
)

st.header("Step 2: Generate Notes")
if uploaded_csv:
    if st.button("📝 Generate Release Notes"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            df = pd.read_csv(uploaded_csv).fillna('')
            st.subheader("🤖 AI-Generated Release Notes")
            
            release_notes_column = find_release_notes_column(df)
            if release_notes_column is None:
                st.error("Error: Could not find a column containing 'Release Notes' in the uploaded CSV.")
                st.stop()
            
            st.success(f"Found release notes column: '{release_notes_column}'")

            skip_texts = ['internal only', 'na']
            process_df = df[~df[release_notes_column].str.strip().str.lower().isin(skip_texts) & (df[release_notes_column].str.strip() != '')].copy()
            skipped_rows_count = len(df) - len(process_df)
            
            if process_df.empty:
                st.warning("No public-facing release notes found to process in the uploaded file.")
            else:
                client = openai.OpenAI(api_key=api_key)
                for index, row in process_df.iterrows():
                    engineering_note = row.to_dict()

                    with st.spinner(f"Generating note for '{engineering_note.get('Summary', 'N/A')}'..."):
                        prompt = build_release_prompt(release_kb, engineering_note)
                        
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": prompt}]
                            )
                            suggestion = response.choices[0].message.content
                        except Exception as e:
                            suggestion = f"An error occurred while calling the API: {e}"

                        st.markdown(suggestion, unsafe_allow_html=True)
                        with st.expander("Show Raw Input"):
                            st.json(engineering_note)
                        st.divider()

            if skipped_rows_count > 0:
                st.info(f"✅ Processing complete. Skipped {skipped_rows_count} internal or empty row(s).")
            else:
                st.success("✅ Processing complete. All rows were processed.")
else:
    st.info("Please upload a CSV file to begin.")
