from pathlib import Path
import Stemmer
import bm25s
import sentence_transformers
import numpy as np
import hashlib
import json

class Indexer():
    def __init__(self, chunks):
        self.chunks = chunks
        self.saving_path = Path("data/processed/")
        self.model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
        self.hash_path = Path("data/cache/corpus_hash.json")
        self.query_cache_path = Path("data/cache/query_cache.json")

    # CACHING THE INDEX METHODES -----
    def get_corpus_hash(self):
        texts = self.chunks
        corpus = "\n".join(texts)
        return hashlib.sha256(corpus.encode()).hexdigest()

    def corpus_changed(self):
        if not self.hash_path.exists():
            return True
        with open(self.hash_path, "r") as f:
            saved_hash = json.load(f)["hash"]
        return saved_hash != self.get_corpus_hash()

    def save_corpus_hash(self):
        self.hash_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_path, "w") as f:
            json.dump({"hash": self.get_corpus_hash()}, f)
    # CACHING THE INDEX METHODES -----

    #CACHING THE QUERY METHODES -----
    def load_query_cache(self):
        if not self.query_cache_path.exists():
            return {}
        with open(self.query_cache_path, "r") as f:
            cache = json.load(f)
        if cache["corpus_hash"] != self.get_corpus_hash():
            return {}
        return cache["queries"]

    def save_query_cache(self, query_cache):
        self.query_cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus_hash": self.get_corpus_hash(),
            "queries": query_cache
        }
        with open(self.query_cache_path, "w") as f:
            json.dump(data, f)
    #CACHING THE QUERY METHODES -----

    # encode chunks once
    def encode_chunks(self):
        self.encoded_chunks = self.model.encode(
            self.chunks, normalize_embeddings=True)
    # encode chunks once


    def bm25s_indexing(self):

        if not self.corpus_changed():
            print("Corpus unchanged, using cached index.")
            return
        print("Corpus changed, rebuilding index.")


        for chunk in self.chunks:
            text = ''
            for i, m in enumerate(chunk.metadata):
                if i != 0:
                    text += ' ' + m
                else:
                    text += m
            chunk.text = text + '\n' + chunk.text

        texts = [chunk.text for chunk in self.chunks] # the chunks text in order
        print(f"Tokenizing {len(texts)} chunks...")
        stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(texts, stemmer=stemmer)
        print(f"{type(corpus_tokens)} -> {corpus_tokens}")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        self.saving_path.mkdir(parents=True, exist_ok=True)
        retriever.save(self.saving_path)
        print(f"BM25 index successfully built and saved to {self.saving_path}")
        self.save_corpus_hash()

    def semantic_search(self, query, top_k):
        model = self.model
        chunks = self.encoded_chunks
        print(f"==={chunks.shape} ====")
        query = model.encode(query, normalize_embeddings=True)
        similarity = chunks @ query
        top_results = similarity.argsort()[::-1][:top_k]
        return top_results, similarity[top_results]

    def lexical_search(self, query: str, top_k: int = 5):
        """Load the BM25 index, query it, and map row IDs back to your in-memory chunks."""
        if not self.saving_path.exists():
            raise FileNotFoundError(f"Index not found at {self.saving_path}. Run 'index' first!")

        retriever = bm25s.BM25.load(str(self.saving_path), load_corpus=False)
        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize([query], stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=top_k)
        results = results[0]
        scores = scores[0]
        return results, scores

    def rescale_vectors(self, scores):
        scores = np.array(scores)
        maxi = max(scores)
        mini = min(scores)

        if maxi == mini:
            return np.zeros_like(scores)

        return (scores - mini) / (maxi - mini)

    def map_chunks(self, query, top_k):
        semantic_res = self.semantic_search(query, top_k)
        lexical_res = self.lexical_search(query, top_k)

        semantic_dict = {}
        for i, s in enumerate(semantic_res[0]):
            semantic_dict[s] = semantic_res[1][i]

        lexical_dict = {}
        for i, s in enumerate(lexical_res[0]):
            lexical_dict[s] = lexical_res[1][i]

        return semantic_dict, lexical_dict

    def fusion(self, query, top_k):

        query_cache = self.load_query_cache()
        cache_key = f"{query}|{top_k}"
        if cache_key in query_cache:
            print("CACHE HIT")
            return query_cache[cache_key]
        print("CACHE MISS")


        semantic_dict, lexical_dict = self.map_chunks(query, top_k)
        combined = set(semantic_dict) | set(lexical_dict)

        semantic_scores = self.rescale_vectors(list(semantic_dict.values()))
        lexical_scores = self.rescale_vectors(list(lexical_dict.values()))

        brandnew_semantic = {}
        for i, s in enumerate(semantic_dict):
            brandnew_semantic[s] = semantic_scores[i]

        brandnew_lexical = {}
        for i ,s in enumerate(lexical_dict):
            brandnew_lexical[s] = lexical_scores[i]

        brandnew_ranking = {}
        for c in combined:
            hybrid_score = 0.5 * brandnew_semantic.get(
                c, 0 ) + 0.5 * brandnew_lexical.get(c, 0)
            brandnew_ranking[c] = hybrid_score

        top_results = sorted(brandnew_ranking.items(),key=lambda x: x[1],
                             reverse=True)[:top_k]

        top_results = [
            [int(chunk_id), float(score)]
            for chunk_id, score in top_results
            ]
        query_cache[cache_key] = top_results
        self.save_query_cache(query_cache)

        return top_results


chunks = [
    "vLLM uses abb PagedAttention to efficiently manage GPU memory.",
    "Berserk is a manga",
    "Berserk cry baby",
    "rat is Jerry",
    "The scheduler manages requests waiting to be executed.",
    "vLLM supports continuous batching for efficient inference.",
    "The tokenizer converts text into tokens before inference.",
    "Python is a programming language used for machine learning.",
]

queries = ["what is cat", "Berserk manga?"]

indexer = Indexer(chunks)
indexer.bm25s_indexing()
indexer.encode_chunks()
for q in queries:
    top = indexer.fusion(q, 3)

    for t in top:
        print(f"with score of {t[1]} | {chunks[t[0]]}")