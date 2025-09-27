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
# IMPORTANT: Replace this placeholder with the actual raw URL of your release_knowledge_base.json file on GitHub
KNOWLEDGE_BASE_URL = "https://github.com/mrsauravs/release-notes-assistant/blob/main/release_knowledge_base.json"

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

    # Determine which instruction to use based on the issue type
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
    You are a Principal Technical Writer at Alation. Your task is to convert a raw engineering note into a formal, customer-facing release note, following Alation's established style and structure.

    **Categorization:**
    Before writing, mentally categorize the note into one of these product areas: {', '.join(knowledge_base['key_product_areas'])}. This will help you choose the right terminology.

    **Raw Engineering Note:**
    ```json
    {json.dumps(engineering_note, indent=2)}
    ```
    
    {task_instruction}
    """
    return prompt

# --- Main Application Logic ---
st.title("Intelligent Release Notes Assistant 🚀")

# Load the knowledge base from the hardcoded URL
release_kb = load_knowledge_base(KNOWLEDGE_BASE_URL)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Release notes style guide is loaded automatically.")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")

# --- File Uploader ---
st.header("Step 1: Upload Your Content")
uploaded_csv = st.file_uploader(
    "Upload your engineering notes CSV file",
    type="csv",
    help="CSV must contain columns like 'Summary', 'Description', and 'Issue Type'"
)

st.header("Step 2: Generate Notes")
if uploaded_csv:
    if st.button("📝 Generate Release Notes"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            df = pd.read_csv(uploaded_csv).fillna('')
            st.subheader("🤖 AI-Generated Release Notes")
            
            client = openai.OpenAI(api_key=api_key)

            for index, row in df.iterrows():
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
else:
    st.info("Please upload a CSV file to begin.")
