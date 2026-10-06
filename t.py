import bm25s
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------
# 1. Our hardcoded chunks
# -------------------------


def rescaling(scores):
    scores = np.array(scores).flatten()
    mini = min(scores)
    maxi = max(scores)

    if mini == maxi:
        return np.zeros_like(scores)
    return (scores - mini) / (maxi - mini)


chunks = [
    "the rat crazzy",
    "automobile of Ferrari is good",
    "Berserk is a manga by Kentaru Miuara from japan",
    "Berserk's main charachter is Guts",
    "The database connection timeout can be configured using environment variables.",
    "Authentication errors happen when the user credentials are invalid.",
    "The server stores database configuration settings in a configuration file.",
]


# -------------------------
# 2. Tokenize the chunks
# -------------------------

corpus_tokens = bm25s.tokenize(chunks)


# -------------------------
# 3. Build the BM25 index
# -------------------------

retriever = bm25s.BM25(method="lucene")

retriever.index(corpus_tokens)


# -------------------------
# 4. Search
# -------------------------

query = "the cat is Tom"

query_tokens = bm25s.tokenize(query)

results, scores = retriever.retrieve(
    query_tokens,
    k=len(chunks),
)


# -------------------------
# 5. Display results
# -------------------------

for i in range(3):
    chunk_id = results[0, i]
    score = scores[0, i]

    print(f"Score: {score:.4f}")
    print(f"Chunk: {chunks[chunk_id]}")
    print()


model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
)
query_embedding = model.encode(
    query,
    normalize_embeddings=True,
)
import numpy as np
s_scores = chunk_embeddings @  query_embedding

ranking = np.argsort(s_scores)[::-1]

for i in ranking[:3]:
    print(f"Score: {s_scores[i]:.4f}")
    print(f"Chunk: {chunks[i]}")
    print()

print("====hybrid====")

s_scores = rescaling(s_scores)
scores = rescaling(scores)
hybrid = 0.5 * s_scores + 0.5 * scores
hybrid_ranking = np.argsort(hybrid)[::-1]

for i in hybrid_ranking[:3]:
    print(f"Score: {hybrid[i]:.4f}")
    print(f"Chunk: {chunks[i]}")
    print()