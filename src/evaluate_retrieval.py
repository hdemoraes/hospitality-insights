import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Ensure imports from src work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embedding_utils import create_or_load_chroma, load_and_embed_csv, retrieve_similar

# ------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------
csv_path = "data/customer_reviews.csv"
collection = create_or_load_chroma()
df, text_col = load_and_embed_csv(csv_path, collection)
print(f"✅ {len(df)} reviews embedded from column: {text_col}")

# ------------------------------------------------------------
# 2. Define evaluation queries
# ------------------------------------------------------------
queries = [
    ("room cleanliness", "dirty, unclean, dusty, smelled bad"),
    ("customer service", "rude, unhelpful, slow response"),
    ("breakfast quality", "cold food, poor variety"),
    ("check-in experience", "fast, friendly, easy process"),
    ("location satisfaction", "close to beach, convenient, far from city")
]

# ------------------------------------------------------------
# 3. Run retrieval & compute similarity metrics
# ------------------------------------------------------------
k = 5  # top-k results
scores = []

from chromadb.utils import embedding_functions
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def flatten(vec):
    """Ensures embeddings are numpy arrays of shape (1, n_features)."""
    arr = np.array(vec)
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

for topic, example_text in queries:
    retrieved = retrieve_similar(collection, topic)
    retrieved_texts = [r for r in retrieved[:k]]

    # Compute embedding similarity safely
    q_vec = flatten(embedder(topic))
    r_vecs = np.vstack([flatten(embedder(t)) for t in retrieved_texts])

    sims = cosine_similarity(q_vec, r_vecs)[0]
    avg_sim = float(np.mean(sims))
    top_sim = float(np.max(sims))

    scores.append({
        "query": topic,
        "avg_similarity": round(avg_sim, 3),
        "top_similarity": round(top_sim, 3),
        "examples": retrieved_texts[:2]
    })

# ------------------------------------------------------------
# 4. Report results
# ------------------------------------------------------------
results_df = pd.DataFrame(scores)
print("\n📊 Retrieval Evaluation Summary:\n")
print(results_df[["query", "avg_similarity", "top_similarity"]])

print("\n🔍 Example retrieved texts:")
for _, row in results_df.iterrows():
    print(f"\n➡️ {row['query'].capitalize()}")
    for ex in row["examples"]:
        print(f"   - {ex[:150]}...")

# ------------------------------------------------------------
# 5. Save results
# ------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/retrieval_evaluation.csv", index=False)
print("\n✅ Saved metrics to outputs/retrieval_evaluation.csv")
