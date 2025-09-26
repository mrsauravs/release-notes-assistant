import streamlit as st
import pandas as pd
import google.generativeai as genai
import openai
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-LLM Release Notes Assistant 🚀",
    page_icon="📝",
    layout="wide"
)

# --- PROMPT TEMPLATES (Common for all models) ---
CORE_PROMPT_TEMPLATE = """
You are a seasoned technical writer at Alation, a leading data intelligence company. Your task is to rewrite raw, engineer-written notes into a polished, customer-facing release note for a bug fix.

**Style Rules:**
- The audience is data analysts, data stewards, and business users.
- The tone should be clear, professional, and user-focused.
- Start with a phrase like "Fixed an issue where..." or "Addressed a defect that...".
- Focus on the user's problem that was solved, not the internal technical details.
- The output must be a single, complete sentence.
- Format the final output as: **{key}**: {{Polished release note}}.

**Raw Input:**
- **Jira Key:** {key}
- **Summary:** {summary}
- **Description:** {description}
- **Engineer's Notes:** {raw_notes}

Rewrite this into a single, polished release note.
"""

API_PROMPT_TEMPLATE = """
You are a technical writer for the developer portal at Alation. Your task is to rewrite raw engineering notes into a clear, direct API change log entry.

**Style Rules:**
- The audience is software developers using the Alation API.
- The tone should be direct, technical, and unambiguous.
- If a component is provided, lead with it in bold brackets: **[{component}]**.
- Clearly state the change: what was added, updated, or deprecated.
- The output must be a single, complete sentence.
- Format the final output as: **{key}**: {{Polished API note}}.

**Raw Input:**
- **Jira Key:** {key}
- **Summary:** {summary}
- **Component:** {component}
- **Engineer's Notes:** {raw_notes}

Rewrite this into a single, technical API release note.
"""

# --- LLM API Call Functions ---

def call_gemini_api(prompt, api_key):
    try:
        # CORRECTED: Explicitly set the transport to 'rest' to avoid
        # auto-discovery of incorrect enterprise (Vertex AI) credentials.
        genai.configure(
            api_key=api_key,
            transport='rest'
        )
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error with Gemini API: {e}"

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
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(
            api_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {"return_full_text": False}
            }
        )
        response.raise_for_status()
        return response.json()[0]['generated_text']
    except Exception as e:
        return f"Error with Hugging Face API: {e}"

# --- Main Note Generation Logic ---
def generate_note(model_provider, api_key, note_type, data):
    """Dispatcher function to call the correct API based on user selection."""

    if note_type == "Core":
        prompt = CORE_PROMPT_TEMPLATE.format(**data)
    else: # API
        prompt = API_PROMPT_TEMPLATE.format(**data)

    if model_provider == "Gemini":
        return call_gemini_api(prompt, api_key)
    elif model_provider == "OpenAI":
        return call_openai_api(prompt, api_key)
    elif model_provider == "Hugging Face":
        return call_huggingface_api(prompt, api_key)
    else:
        return "Error: Invalid model provider selected."

# --- UI Layout ---
st.title("📝 Multi-LLM Release Notes Assistant 🚀")
st.markdown("This tool uses your chosen LLM to draft release notes from engineering input.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    model_provider = st.selectbox(
        "Choose your LLM Provider",
        ("Gemini", "OpenAI", "Hugging Face")
    )
    api_key = st.text_input(f"Enter your {model_provider} API Key", type="password")

    st.info(
        "**Where to find your API Key:**\n"
        "- **Gemini:** [Google AI Studio](https://aistudio.google.com/)\n"
        "- **OpenAI:** [OpenAI Platform](https://platform.openai.com/api-keys)\n"
        "- **Hugging Face:** [Access Tokens](https://huggingface.co/settings/tokens)"
    )

# --- Main Application Tabs ---
core_tab, dev_tab = st.tabs(["Alation Core Release Notes", "Developer Portal (API) Release Notes"])

def run_generation(note_type, uploader_key, button_key):
    """Generic function to handle the UI logic for file upload and generation."""
    uploaded_file = st.file_uploader(
        "Upload a CSV with Key, Summary, Description, Release Notes, and Component columns.",
        key=uploader_key,
        type="csv"
    )
    if uploaded_file:
        if st.button(f"Generate {note_type} Release Notes", key=button_key):
            if not api_key:
                st.error("Please enter your API key in the sidebar.")
                return

            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("🤖 AI-Generated Suggestions")
                progress_bar = st.progress(0, text="Starting generation...")

                for index, row in df.iterrows():
                    progress_text = f"Generating note for {row.get('Key', 'N/A')}..."
                    progress_bar.progress((index) / len(df), text=progress_text)

                    data_payload = {
                        'key': row.get('Key', ""),
                        'summary': row.get('Summary', ""),
                        'description': row.get('Description', ""),
                        'raw_notes': row.get('Release Notes', ""),
                        'component': row.get('Component', "")
                    }

                    with st.spinner(progress_text):
                        suggestion = generate_note(model_provider, api_key, note_type, data_payload)

                    st.markdown(suggestion)
                    st.code(f"Original Summary: {data_payload['summary']}\nComponent: {data_payload['component']}", language="text")
                    st.divider()

                progress_bar.progress(1.0, text="Generation complete!")
                st.success("All notes have been generated successfully!")

            except Exception as e:
                st.error(f"An error occurred during file processing: {e}")

with core_tab:
    st.header("Generate General Release Notes")
    run_generation("Core", "core_uploader", "core_generate")

with dev_tab:
    st.header("Generate API Release Notes")
    run_generation("API", "dev_uploader", "dev_generate")
