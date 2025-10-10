import os
import pandas as pd
from chromadb.utils import embedding_functions

# Existing create_or_load_chroma() remains unchanged

def load_and_embed_csv(csv_path, collection):
    """
    Loads a CSV, embeds its text column, and caches embeddings locally to speed up future runs.
    """
    df = pd.read_csv(csv_path)
    text_col = next((c for c in df.columns if "review" in c.lower() or "text" in c.lower()), None)

    if not text_col:
        raise ValueError("No text column found in CSV. Expected a column containing 'review' or 'text'.")

    cache_file = f"{csv_path}.embedded.parquet"

    if os.path.exists(cache_file):
        print(f"⚡ Using cached embeddings from {cache_file}")
        return pd.read_parquet(cache_file), text_col

    # --- Embed if not cached ---
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    df["embedding"] = df[text_col].apply(lambda x: embedder(str(x)))
    df.to_parquet(cache_file, index=False)
    print(f"✅ Cached embeddings saved to {cache_file}")

    # --- Store in ChromaDB ---
    for idx, row in df.iterrows():
        collection.add(
            ids=[str(row.get("review_id", idx))],
            documents=[str(row[text_col])],
            embeddings=[row["embedding"]]
        )

    return df, text_col
