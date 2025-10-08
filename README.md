# Hospitality Insights 🧠

AI-powered document & review intelligence.  
MVP: analyse guest reviews to extract top issues, trends, and exportable reports.

## 📂 Structure
data/ – input files (not tracked)
outputs/ – generated CSVs/reports
src/ – core python modules
ui/ – streamlit assets (week 4)
notebooks/ – experiments & demos


## 🚀 Quickstart (Windows)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

GOOGLE_API_KEY=your_key_here
ENV=dev

python src\smoke_test.py


---

## 4) (optional but recommended) add `CODEOWNERS`
**Where:** *Explorer* → **New File** → name: `CODEOWNERS`  
**What to paste:**

**Why:** marks you as default reviewer/owner for any PRs.

---

## 5) create a quick smoke test script

**Where:** *Explorer* → right-click **src/** → **New File** → name: `smoke_test.py`  
**What to paste:**
```python
import importlib, sys

mods = [
    "chromadb",
    "google.generativeai",
    "pandas",
    "numpy",
    "sentence_transformers",
    "sklearn"
]

print("🔎 Importing modules...")
for m in mods:
    try:
        importlib.import_module(m)
        print(f"✅ {m}")
    except Exception as e:
        print(f"❌ {m} -> {e}")
        sys.exit(1)

print("🎉 Environment OK")

