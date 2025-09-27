import streamlit as st
import pandas as pd
import google.generativeai as genai
import openai
import requests
from huggingface_hub import InferenceClient
from importlib import metadata
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-LLM Release Notes Assistant 🚀",
    page_icon="📝",
    layout="wide"
)

# --- Hardcoded URL for the Knowledge Base ---
# IMPORTANT: Replace this placeholder with the actual raw URL of your JSON file on GitHub
KNOWLEDGE_BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/knowledge_base.json"

# --- Knowledge Base and Prompt Generation ---
@st.cache_data(ttl=3600)
def load_knowledge_base(url):
    """Fetches and loads the knowledge base JSON from a raw GitHub URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Fatal Error: Could not fetch knowledge base from URL: {url}. Please check the URL in the code.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Fatal Error: Could not decode the knowledge base. Please ensure the file at the URL is valid JSON.")
        st.stop()

def build_prompt_with_knowledge(knowledge_base, data):
    """Dynamically builds a prompt using the external knowledge base and input data."""
    if not knowledge_base:
        return "Error: Knowledge base not loaded."

    # Create formatted lists of the valid terms to guide the AI
    deployment_options = ", ".join(f"'{item}'" for item in knowledge_base['deployment_types'])
    role_options = ", ".join(f"'{item}'" for item in knowledge_base['user_roles'])
    area_options = ", ".join(f"'{item}'" for item in knowledge_base['functional_areas'])
    topic_options = ", ".join(f"'{item}'" for item in knowledge_base['topics'])

    prompt = f"""
    You are a Principal Technical Writer at Alation responsible for producing and classifying release notes. Your task is two-fold:
    1.  **Classify the Input:** Analyze the raw input and determine the correct metadata.
    2.  **Write the Release Note:** Write a clear, customer-facing release note based on the input and classification.

    **Reference Information (Valid Options):**
    -   **Deployment Types:** {deployment_options}
    -   **User Roles:** {role_options}
    -   **Functional Areas:** {area_options}
    -   **Topics:** {topic_options}

    **Raw Input Data:**
    -   **Summary:** {data.get('summary', "")}
    -   **Description:** {data.get('description', "")}
    -   **Issue Type:** {data.get('issue_type', "Feature")}
    -   **Engineer's Notes:** {data.get('raw_notes', "")}

    **Your Task:**
    First, based on the raw input, provide a YAML block with the most appropriate classification. Choose ONLY from the valid options provided in the reference section.
    -   `Deployment Type`: Choose one. Infer based on keywords like 'Cloud', 'Customer Managed', 'Server', or 'Agent'. If unsure or applicable to both, choose 'Customer Managed, Alation Cloud Service'.
    -   `User Role`: Choose the one primary user role most affected by this change.
    -   `Functional Area`: Choose the one best-fitting functional area.
    -   `Topics`: Choose 1 to 3 relevant topics.

    Second, write the customer-facing release note below the YAML block, separated by '---'.
    -   For new features, use a structured format with a header, a "What's New?" section, and a "Why it Matters" section.
    -   For bug fixes, use a single, direct sentence.

    **Example Output Format:**
    ```yaml
    Deployment Type: Customer Managed
    User Role: Server Admin
    Functional Area: Server Maintenance
    Topics:
      - Back Up and Restore Usage
      - Logging
    ```
    ---
    ## 🚀 Enhanced Backup and Restore Monitoring
    **What's New?**
    You can now monitor the status of backup and restore jobs directly from the Alation UI.
    **Why it Matters**
    - **Improved Visibility:** Administrators can now easily check the health and history of their backups without accessing the server backend.
    - **Faster Troubleshooting:** Quick access to logs and status helps diagnose issues more efficiently.

    Now, process the provided raw input and generate the response in the specified format.
    """
    return prompt


# --- LLM API Call Functions (Unchanged) ---
def call_gemini_api(prompt, api_key):
    try:
        version = metadata.version("google-generativeai")
        st.success(f"Running with google-generativeai version: {version}")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API:")
        st.exception(e)
        return None

def call_openai_api(prompt, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI API: {e}"

def call_huggingface_api(prompt, api_key, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
    try:
        client = InferenceClient(token=api_key)
        response = client.text_generation(
            prompt, 
            model=model_id,
            max_new_tokens=512,
            return_full_text=False
        )
        return response
    except Exception as e:
        return f"Error with Hugging Face API: {e}"

# --- Main Note Generation Logic (Refactored) ---
def generate_note(model_provider, api_key, knowledge_base, data):
    """Dispatcher function to build a prompt and call the correct API."""
    
    prompt = build_prompt_with_knowledge(knowledge_base, data)

    if model_provider == "Gemini":
        return call_gemini_api(prompt, api_key)
    elif model_provider == "OpenAI":
        return call_openai_api(prompt, api_key)
    elif model_provider == "Hugging Face":
        return call_huggingface_api(prompt, api_key)
    else:
        return "Error: Invalid model provider selected."

# --- UI Layout ---
st.title("📝 Intelligent Release Notes Assistant 🚀")
st.markdown("This tool uses a knowledge base to generate and classify release notes from your engineering input.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Knowledge base is loaded automatically from a predefined URL.")
    model_provider = st.selectbox(
        "Choose your LLM Provider",
        ("OpenAI", "Gemini", "Hugging Face")
    )
    api_key = st.text_input(f"Enter your {model_provider} API Key", type="password")

# --- Main Application Logic ---
knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_URL)

st.header("Generate and Classify Release Notes")
uploaded_file = st.file_uploader(
    "Upload a CSV with the required columns.",
    key="main_uploader",
    type="csv"
)

if uploaded_file:
    if st.button("✨ Generate Release Notes", key="main_generate"):
        if not api_key:
            st.error("Please enter your API key in the sidebar.")
        elif not knowledge_base:
            # The load_knowledge_base function will already show an error and stop.
            # This is a fallback.
            st.error("Could not load Knowledge Base. See error message above.")
        else:
            try:
                df = pd.read_csv(uploaded_file, dtype={"Release Notes": str}).fillna('')
                
                st.subheader("🤖 AI-Generated Suggestions")
                progress_bar = st.progress(0, text="Starting generation...")

                for i, row in enumerate(df.itertuples(index=False)):
                    row_dict = row._asdict()
                    data_payload = {
                        'key': row_dict.get('Key', ""),
                        'summary': row_dict.get('Summary', ""),
                        'description': row_dict.get('Description', ""),
                        'raw_notes': row_dict.get('Release Notes', ""),
                        'components': row_dict.get('Components', ""),
                        'issue_type': row_dict.get('Issue Type', 'Feature')
                    }

                    progress_text = f"Processing {data_payload['key']} ({i+1}/{len(df)})..."
                    progress_bar.progress((i + 1) / len(df), text=progress_text)

                    with st.spinner(progress_text):
                        suggestion = generate_note(
                            model_provider, 
                            api_key, 
                            knowledge_base,
                            data_payload
                        )
                    
                    if suggestion:
                        st.markdown(suggestion, unsafe_allow_html=True)
                        st.code(f"Original Summary: {data_payload['summary']}\nIssue Type: {data_payload['issue_type']}\nComponents: {data_payload['components']}", language="text")
                        st.divider()

                progress_bar.progress(1.0, text="Generation complete!")
                st.success("All notes have been generated successfully!")

            except Exception as e:
                st.error(f"An error occurred during file processing: {e}")
