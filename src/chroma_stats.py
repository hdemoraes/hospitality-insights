import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Make sure src imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embedding_utils import create_or_load_chroma, load_and_embed_csv, retrieve_similar

# ------------------------------------------------------------
# 1. Load the ChromaDB collection and dataset
# ------------------------------------------------------------
csv_path = "data/customer_reviews.csv"
collection = create_or_load_chroma()
df, text_col = load_and_embed_csv(csv_path, collection)

print(f"✅ Loaded {len(df)} rows from {csv_path}")
print(f"🔹 Text column detected: {text_col}")

# ------------------------------------------------------------
# 2. Extract basic collection metadata
# ------------------------------------------------------------
try:
    meta = collection.count()
    print(f"📦 Collection contains {meta} documents")
except Exception:
    meta = "Unavailable (using client version fallback)"

# ------------------------------------------------------------
# 3. Sample 10 random records for quick inspection
# ------------------------------------------------------------
sample_texts = df[text_col].sample(min(10, len(df))).tolist()

# ------------------------------------------------------------
# 4. Compute embedding health statistics
# ------------------------------------------------------------
from chromadb.utils import embedding_functions
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def flatten(vec):
    arr = np.array(vec)
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

sample_vecs = np.vstack([flatten(embedder(t)) for t in sample_texts])
pairwise_sims = cosine_similarity(sample_vecs)

avg_sim = np.mean(pairwise_sims[np.triu_indices_from(pairwise_sims, k=1)])
std_sim = np.std(pairwise_sims[np.triu_indices_from(pairwise_sims, k=1)])
dim = sample_vecs.shape[1]

print(f"📈 Embedding vector dimension: {dim}")
print(f"📊 Avg pairwise similarity among random samples: {avg_sim:.3f} ± {std_sim:.3f}")

# ------------------------------------------------------------
# 5. Evaluate recall baseline (rough)
# ------------------------------------------------------------
queries = [
    "room cleanliness",
    "breakfast quality",
    "customer service",
]

recall_scores = []
for q in queries:
    retrieved = retrieve_similar(collection, q)
    retrieved_texts = [r.lower() for r in retrieved[:5]]
    matched = [q.split()[0] in t for t in retrieved_texts]
    recall = sum(matched) / len(matched)
    recall_scores.append((q, recall))

recall_df = pd.DataFrame(recall_scores, columns=["query", "recall_score"])

# ------------------------------------------------------------
# 6. Build summary report
# ------------------------------------------------------------
report = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "collection_count": meta,
    "embedding_dimension": dim,
    "avg_similarity": round(avg_sim, 3),
    "std_similarity": round(std_sim, 3),
    "avg_recall": round(recall_df["recall_score"].mean(), 3),
}

print("\n📋 Summary Report:")
for k, v in report.items():
    print(f"   {k}: {v}")

# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
pd.DataFrame([report]).to_csv("outputs/chroma_stats_summary.csv", index=False)
recall_df.to_csv("outputs/chroma_recall_scores.csv", index=False)

print("\n✅ Saved:")
print("   - outputs/chroma_stats_summary.csv")
print("   - outputs/chroma_recall_scores.csv")
