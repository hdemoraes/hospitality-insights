import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise EnvironmentError("❌ GOOGLE_API_KEY not found in .env file")

# Configure client (v1 endpoint)
genai.configure(api_key=api_key)

# Check available models
print("🔍 Available Gemini models:")
available = [m.name for m in genai.list_models()]
for m in available:
    print(" -", m)

# Select preferred model dynamically
PREFERRED_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
]
MODEL_NAME = next((m for m in PREFERRED_MODELS if m in available), None)

if not MODEL_NAME:
    raise ValueError("❌ No supported Gemini models found in your API list.")

print(f"\n⚙️ Using model: {MODEL_NAME}")

# Test generation
model = genai.GenerativeModel(MODEL_NAME)
prompt = "Write a 2-sentence welcome message for an AI-powered analytics app."
response = model.generate_content(prompt)

print("\n✅ Gemini response:")
print(response.text)
