import streamlit as st
import pandas as pd
import json
import requests
import openai
from datetime import datetime
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Release Notes Assistant 🚀",
    page_icon="📝",
    layout="wide"
)

# --- Hardcoded URL for the Knowledge Base ---
KNOWLEDGE_BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/knowledge_base.json"

# --- Helper functions ---
def find_column_by_substring(df, substring):
    """Dynamically finds a column name containing a given substring."""
    substring = substring.lower()
    for col in df.columns:
        if substring in col.lower():
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
        The engineering note describes a bug fix. Write a single sentence for a Markdown bullet point using this exact format:
        `{style_guide['bug_fix_writing']['format']}`
        """
    else: # Default to feature/enhancement
        task_instruction = f"""
        **Task:**
        The engineering note describes a new feature or enhancement. Write the release note following this instruction:
        "{style_guide['feature_enhancement_writing']['instruction']}"
        """

    prompt = f"""
    You are a Principal Technical Writer at Alation. Your task is to convert a raw engineering note into a formal, customer-facing release note. You must strictly follow the format requested.

    **Raw Engineering Note:**
    ```json
    {json.dumps(engineering_note, indent=2)}
    ```
    
    {task_instruction}
    """
    return prompt

# --- Main Application Logic ---
st.title("Intelligent Release Notes Assistant 🚀")

if 'final_report' not in st.session_state:
    st.session_state.final_report = None

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
        
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            df = pd.read_csv(uploaded_csv).fillna('')
            st.subheader("Processing Notes...")
            
            release_notes_column = find_column_by_substring(df, 'release notes')
            if release_notes_column is None:
                st.error("Error: Could not find a column containing 'Release Notes' in the uploaded CSV.")
                st.stop()
            
            is_empty = df[release_notes_column].str.strip() == ''
            is_na = df[release_notes_column].str.strip().str.lower() == 'na'
            is_internal = df[release_notes_column].str.strip().str.lower().str.contains('internal', na=False)
            process_df = df[~(is_empty | is_na | is_internal)].copy()
            
            if process_df.empty:
                st.warning("No public-facing release notes found to process in the uploaded file.")
            else:
                client = openai.OpenAI(api_key=api_key)
                results = {"New Features": [], "Enhancements": [], "Bug Fixes": []}
                progress_bar = st.progress(0, text="Generating notes...")
                total_rows = len(process_df)
                processed_count = 0

                for index, row in process_df.iterrows():
                    processed_count += 1
                    engineering_note = row.to_dict()
                    summary = engineering_note.get('Summary', 'N A')
                    issue_type = engineering_note.get("Issue Type", "Feature").lower()
                    
                    progress_bar.progress(processed_count / total_rows, text=f"Processing: {summary}")
                    
                    prompt = build_release_prompt(release_kb, engineering_note)
                    try:
                        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
                        suggestion = response.choices[0].message.content.strip()

                        if "bug" in issue_type or "defect" in issue_type:
                            results["Bug Fixes"].append(suggestion)
                        elif "enhancement" in issue_type:
                            results["Enhancements"].append(suggestion)
                        else:
                            results["New Features"].append(suggestion)

                    except Exception as e:
                        st.warning(f"Could not process '{summary}': {e}")
                
                progress_bar.progress(1.0, text="Assembling final document...")

                month_year = datetime.now().strftime('%B %Y')
                report_parts = [f"# Release {release_version}", f"_{month_year}_"]
                for section in release_kb['release_structure']['section_order']:
                    if results.get(section):
                        report_parts.append(f"\n\n**{section}**\n")
                        if section == "Bug Fixes":
                            report_parts.append("\n".join(results[section]))
                        else:
                            report_parts.append("\n\n".join(results[section]))
                
                st.session_state.final_report = "\n".join(report_parts)
                st.success("✅ Release notes document generated successfully!")

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
