import os
import sys

# Add the project root (one level above /src) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding_utils import create_or_load_chroma, load_and_embed_csv, retrieve_similar

# Initialize or load the vector store
collection = create_or_load_chroma()

# Load your dataset and embed
csv_path = "data/customer_reviews.csv"
df, text_col = load_and_embed_csv(csv_path, collection)

print(f"✅ {len(df)} reviews embedded from column: {text_col}")

# Test a few retrieval queries
test_queries = [
    "What are the most common customer complaints?",
    "How do guests describe the service?",
    "What do people like about the hotel breakfast?"
]

for q in test_queries:
    print(f"\n🔍 Query: {q}")
    results = retrieve_similar(collection, q)
    for r in results[:3]:
        print(f" - {r}")


from src.embedding_utils import create_or_load_chroma, load_and_embed_csv, retrieve_similar

# Initialize or load the vector store
collection = create_or_load_chroma()

# Load your dataset and embed
csv_path = "data/customer_reviews.csv"
df, text_col = load_and_embed_csv(csv_path, collection)

print(f"✅ {len(df)} reviews embedded from column: {text_col}")

# Test a few retrieval queries
test_queries = [
    "What are the most common customer complaints?",
    "How do guests describe the service?",
    "What do people like about the hotel breakfast?"
]

for q in test_queries:
    print(f"\n🔍 Query: {q}")
    results = retrieve_similar(collection, q)
    for r in results[:3]:
        print(f" - {r}")
