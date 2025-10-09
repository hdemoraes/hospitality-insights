# ============================================================
# 🤖 Hospitality AI Assistant – Streamlit + Gemini (Auto Model)
# ============================================================

import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------------------
# 1. Load environment variables
# ------------------------------
# Automatically search for the .env file in the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.set_page_config(page_title="Hospitality AI Assistant")
    st.error("❌ GOOGLE_API_KEY not found. Please check your .env file location.")
    st.stop()

# ------------------------------
# 2. Configure Gemini API
# ------------------------------
genai.configure(api_key=api_key)

# ------------------------------
# 3. Auto-detect available models
# ------------------------------
@st.cache_resource(show_spinner=False)
def get_first_available_model():
    try:
        models = genai.list_models()
        valid_models = [
            m for m in models if hasattr(m, "supported_generation_methods")
            and "generateContent" in m.supported_generation_methods
        ]
        if not valid_models:
            raise ValueError("No text-generation models available for this API key.")
        return valid_models[0].name
    except Exception as e:
        st.error(f"⚠️ Unable to fetch models: {e}")
        st.stop()

model_name = get_first_available_model()

# Initialize the model
model = genai.GenerativeModel(model_name)

# ------------------------------
# 4. Streamlit UI Layout
# ------------------------------
st.set_page_config(page_title="Hospitality AI Assistant", layout="centered")
st.title("🤖 Hospitality AI Assistant")
st.caption(f"✅ Connected to: `{model_name}`")

st.markdown(
    "Ask a question about hospitality, guests, or management insights. "
    "This demo uses Google's Gemini API to generate real-time responses."
)

# ------------------------------
# 5. Query Input + AI Response
# ------------------------------
query = st.text_input("💬 Enter your question:")

if query:
    with st.spinner("Thinking... 🤔"):
        try:
            response = model.generate_content(query)
            st.markdown(f"### 🧠 Answer:\n{response.text}")
        except Exception as e:
            st.error(f"⚠️ Error generating response: {e}")

# ------------------------------
# 6. Footer
# ------------------------------
st.divider()
st.markdown(
    "<small>Powered by Google Gemini & Streamlit | Built by FastAISolution</small>",
    unsafe_allow_html=True,
)
