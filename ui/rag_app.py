import os
import sys
import streamlit as st
import pandas as pd
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# --- Make sure src folder is visible ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embedding_utils import create_or_load_chroma, load_and_embed_csv, retrieve_similar

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="Hospitality Insights AI", page_icon="🧠", layout="wide")
st.title("🧠 Hospitality Insights AI")
st.caption("AI-powered feedback intelligence using ChromaDB + Gemini 2.5 🚀")

# ============================================================
# 2. LOAD ENVIRONMENT
# ============================================================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Google API key not found in .env.")
    st.stop()
genai.configure(api_key=api_key)

# Auto-select current Gemini 2.5 model
available_models = [m.name for m in genai.list_models()]
preferred_models = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
]
MODEL_NAME = next((m for m in preferred_models if m in available_models), None)
if not MODEL_NAME:
    st.error("❌ No Gemini model found.")
    st.stop()

# ============================================================
# 3. INITIALIZE DATABASE + LOG FILE
# ============================================================
collection = create_or_load_chroma()
os.makedirs("outputs", exist_ok=True)
LOG_PATH = "outputs/session_log.csv"

# ============================================================
# 4. STEP 1 – UPLOAD CSV
# ============================================================
st.markdown("### 📂 Step 1: Upload and Embed Dataset")
uploaded_file = st.file_uploader("Upload your `customer_reviews.csv`", type=["csv"])

if uploaded_file is not None:
    csv_path = os.path.join("data", uploaded_file.name)
    os.makedirs("data", exist_ok=True)
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["csv_path"] = csv_path
    st.success(f"✅ File `{uploaded_file.name}` uploaded successfully.")

    if st.button("📊 Embed Uploaded Dataset"):
        try:
            df, text_col = load_and_embed_csv(csv_path, collection)
            st.session_state["text_col"] = text_col
            st.success(f"✅ {len(df)} rows embedded.")
            st.caption(f"Detected text column: **{text_col}**")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"⚠️ Embedding failed: {e}")
else:
    st.info("📥 Upload your CSV to begin.")

# ============================================================
# 5. STEP 2 – ASK A QUESTION
# ============================================================
st.markdown("---")
st.markdown("### 💬 Step 2: Ask a Question")
query = st.text_input("Type your question based on the dataset 👇")

if st.button("🚀 Analyze"):
    if not query.strip():
        st.warning("Please enter a question before analyzing.")
    else:
        try:
            with st.spinner("🔍 Retrieving relevant reviews..."):
                similar_docs = retrieve_similar(collection, query)

            if not similar_docs:
                st.info("No relevant results found.")
            else:
                st.success("✅ Top reviews retrieved.")
                st.markdown("### 🔎 Top Retrieved Reviews (Expandable)")
                for i, doc in enumerate(similar_docs[:5], 1):
                    with st.expander(f"Review {i}"):
                        st.write(doc)

                # ====================================================
                # 6. HYBRID RAG – GEMINI ANALYSIS
                # ====================================================
                st.markdown("---")
                st.markdown("### 🤖 AI-Generated Insights (Gemini 2.5)")
                context = "\n\n".join(similar_docs[:10])
                prompt = f"""
                You are an AI analyst specializing in hospitality feedback.
                Summarize key insights and actionable recommendations based on the text below.
                Keep it concise (150–200 words) and professional.

                === Customer Reviews ===
                {context}

                === Query ===
                {query}
                """

                try:
                    model = genai.GenerativeModel(MODEL_NAME)
                    response = model.generate_content(prompt)
                    ai_text = response.text.strip() if hasattr(response, "text") else ""

                    if ai_text:
                        st.markdown(ai_text)

                        # Save to session log
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        entry = pd.DataFrame(
                            [[timestamp, query, ai_text]],
                            columns=["timestamp", "query", "response"]
                        )
                        if os.path.exists(LOG_PATH):
                            entry.to_csv(LOG_PATH, mode="a", header=False, index=False)
                        else:
                            entry.to_csv(LOG_PATH, index=False)
                        st.success("💾 Response saved to session log.")
                    else:
                        st.info("Gemini returned no text.")
                except Exception as e:
                    st.error(f"Gemini generation failed: {e}")
        except Exception as e:
            st.error(f"⚠️ Retrieval or analysis failed: {e}")

# ============================================================
# 7. SESSION LOG TOOLS
# ============================================================
st.markdown("---")
st.markdown("### 📜 Session Log Viewer & Export")

if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)
    st.dataframe(df_log.tail(10))
    st.download_button(
        label="⬇️ Download Full Log as CSV",
        data=df_log.to_csv(index=False),
        file_name="session_log.csv",
        mime="text/csv"
    )
else:
    st.info("No session log found yet.")

# ============================================================
# 8. FOOTER
# ============================================================
st.markdown("---")
st.caption("Powered by Google Gemini 2.5 + ChromaDB | FastAISolution 🧩")
