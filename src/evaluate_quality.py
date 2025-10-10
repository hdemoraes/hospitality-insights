import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Fix imports to src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embedding_utils import create_or_load_chroma, retrieve_similar

# ============================================================
# 1. Setup
# ============================================================
os.makedirs("outputs/metrics", exist_ok=True)
session_log_path = "outputs/session_log.csv"

if not os.path.exists(session_log_path):
    raise FileNotFoundError("❌ No session_log.csv found. Run some queries first.")

df_log = pd.read_csv(session_log_path)
collection = create_or_load_chroma()

print(f"✅ Loaded {len(df_log)} logged sessions for evaluation.")

# ============================================================
# 2. Helper Functions
# ============================================================

def compute_relevance(response_text, retrieved_docs):
    """
    Measures cosine similarity between response and retrieved context.
    """
    from chromadb.utils import embedding_functions
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    def flatten(vec):
        arr = np.array(vec)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    r_vec = flatten(embedder(response_text))
    d_vecs = np.vstack([flatten(embedder(d)) for d in retrieved_docs])
    sims = cosine_similarity(r_vec, d_vecs)[0]
    return float(np.mean(sims))

def compute_faithfulness(response_text, retrieved_docs):
    """
    Estimates how factually grounded the response is:
    Counts overlap of key phrases between response and context.
    """
    response_words = set(response_text.lower().split())
    context_words = set(" ".join(retrieved_docs).lower().split())
    overlap = len(response_words.intersection(context_words)) / max(1, len(response_words))
    return round(overlap, 3)

def compute_sentiment_alignment(response_text, retrieved_docs):
    """
    Compares sentiment polarity between response and retrieved context.
    """
    resp_polarity = TextBlob(response_text).sentiment.polarity
    context_polarities = [TextBlob(doc).sentiment.polarity for doc in retrieved_docs]
    avg_context_polarity = np.mean(context_polarities)
    diff = abs(resp_polarity - avg_context_polarity)
    return round(1 - diff, 3)  # higher = better alignment

# ============================================================
# 3. Evaluate Each Logged Response
# ============================================================

results = []
for _, row in df_log.iterrows():
    query = row["query"]
    response = str(row["response"])

    # Retrieve docs to use as ground truth
    retrieved = retrieve_similar(collection, query)[:5]

    relevance = compute_relevance(response, retrieved)
    faithfulness = compute_faithfulness(response, retrieved)
    sentiment = compute_sentiment_alignment(response, retrieved)
    avg_score = round(np.mean([relevance, faithfulness, sentiment]), 3)

    results.append({
        "timestamp": row["timestamp"],
        "query": query,
        "relevance": relevance,
        "faithfulness": faithfulness,
        "sentiment_alignment": sentiment,
        "overall_quality": avg_score
    })

# ============================================================
# 4. Save Results
# ============================================================
metrics_df = pd.DataFrame(results)
output_file = f"outputs/metrics/quality_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
metrics_df.to_csv(output_file, index=False)

print("\n📊 Quality Evaluation Complete:")
print(metrics_df.describe())
print(f"\n✅ Saved metrics to: {output_file}")
