import importlib
import sys

modules = [
    "chromadb",
    "google.generativeai",
    "pandas",
    "numpy",
    "sentence_transformers",
    "sklearn"
]

print("🔎 Importing modules...\n")
for m in modules:
    try:
        importlib.import_module(m)
        print(f"✅ {m}")
    except Exception as e:
        print(f"❌ {m} -> {e}")
        sys.exit(1)

print("\n🎉 Environment OK — all key dependencies loaded successfully!")
