from .pyd_models import Chunk, MinimalSource
from pathlib import Path
import Stemmer
import bm25s
import sentence_transformers
import numpy as np
import math
import json

class Indexer():
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.saving_path = Path("data/processed/")
        # self.model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

    def encode_chunks(self):
        gather = []
        batch_size = 32

        for i in range(0, len(self.chunks), batch_size):
            batch = [chunk.text for chunk in self.chunks[i:i + batch_size]]

            res = self.model.encode(
                batch,
                normalize_embeddings=True
            )

            gather.extend(res)

            print(
                f"Encoded {min(i + batch_size, len(self.chunks))}"
                f"/{len(self.chunks)} chunks"
            )

        self.encoded_chunks = np.array(gather)

    def tokenize_chunks(self, chunks, query):
        en_stem = Stemmer.Stemmer("english")
        tokenized = bm25s.tokenize(chunks, stemmer=en_stem)
        query = bm25s.tokenize([query], stemmer=en_stem)

        vocab = tokenized.vocab
        query_tokens = [
            vocab[token]
            for token in query.vocab
            if token in vocab
        ]
        encoded_chunks = tokenized.ids

        return query_tokens, encoded_chunks

    def build_tf(self, query, chunks):
        chunks_infos = []
        for chunk in chunks:
            tf_dict = {}
            for token in query:
                tf = chunk.count(token)
                if not tf:
                    continue
                tf_dict[token] = tf

            chunks_infos.append((chunk, tf_dict))
        return chunks_infos

    def build_df(self, query, chunks):
        df_dict = {}

        for token in query:
            df = sum(token in chunk for chunk in chunks)
            df_dict[token] = df
        return df_dict
            
    def build_idf(self, query, df_dict, chunks):
        idf_dict = {}
        n = len(chunks)
        for token in query:
            df = df_dict[token]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            idf_dict[token] = idf
        return idf_dict

    def average_lenght(self, chunks):
        lenght = 0
        for chunk in chunks:
            lenght += len(chunk)
        avgdl = lenght / len(chunks) if len(chunks) else 0
        return len(chunks), avgdl

    def bm25_score(self, tf_dict, idf_dict, dl, avgdl, k1=1.5, b=0.75):
        bm25_score = 0
        for token , tf in tf_dict.items():
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl /avgdl)
            score = idf_dict[token] * numerator / denominator
            bm25_score += score
        return bm25_score

    def score_chunks(self, chunks_tf, chunk_tokens, idf_dict, avgdl, k1=1.5, b=0.75):
        scores = {}

        for i, (chunk, tf_dict) in enumerate(chunks_tf):
            dl = len(chunk_tokens[i])
            score = self.bm25_score(tf_dict, idf_dict, dl, avgdl, k1, b)
            scores[i] = score
        return scores

    def return_topk(self, scores, top_k):
        return sorted(scores, key=scores.get, reverse=True)[:top_k]

    def bm25s_indexing(self):
        self.saving_path.mkdir(parents=True, exist_ok=True)

        for chunk in self.chunks:
            text = ''
            for i, m in enumerate(chunk.metadata):
                if i != 0:
                    text += ' ' + m
                else:
                    text += m
            chunk.text = text + '\n' + chunk.text

        texts = [chunk.text for chunk in self.chunks]
        print(f"Tokenizing {len(texts)} chunks...")
        stemmer = Stemmer.Stemmer("english")
        tokenized_corpus = bm25s.tokenize(texts, stemmer=stemmer)

        index_data = {
            "tokenized_ids": [list(arr) for arr in tokenized_corpus.ids],
            "vocab": tokenized_corpus.vocab
        }

        index_file = self.saving_path / "custom_bm25_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False)

        print(f"Custom BM25 index successfully built and saved to {index_file}")

    def semantic_search(self, query, top_k):
        model = self.model
        query = model.encode(query, normalize_embeddings=True)
        similarity = self.encoded_chunks @ query
        top_results = similarity.argsort()[::-1][:top_k]
        return top_results, similarity[top_results]

    def lexical_search(self, query: str, top_k: int = 5):
            """Load the custom BM25 index from JSON, query it, and return top results."""
            index_file = self.saving_path / "custom_bm25_index.json"
            
            # Enforce that the pre-built index must exist
            if not index_file.exists():
                raise FileNotFoundError(
                    f"Custom BM25 index not found at {index_file}. Please run indexing first."
                )

            with open(index_file, "r", encoding="utf-8") as f:
                saved_index = json.load(f)
                en_chunks = saved_index["tokenized_ids"]

            # Allow self.chunks to be None by using a safe fallback length matching the saved index
            if self.chunks:
                chunks_text = [chunk.text for chunk in self.chunks]
            else:
                chunks_text = [""] * len(en_chunks)

            query_tokens, _ = self.tokenize_chunks(chunks_text, query)

            if not query_tokens:
                return np.array([]), np.array([])

            chunks_tf = self.build_tf(query_tokens, en_chunks)
            df_dict = self.build_df(query_tokens, en_chunks)
            idf_dict = self.build_idf(query_tokens, df_dict, en_chunks)
            _, avgdl = self.average_lenght(en_chunks)

            scores = self.score_chunks(chunks_tf, en_chunks, idf_dict, avgdl)
            
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results = np.array([item[0] for item in sorted_scores])
            retrieved_scores = np.array([item[1] for item in sorted_scores])

            return results, retrieved_scores

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
                c, 0 ) + 1.5 * brandnew_lexical.get(c, 0)
            brandnew_ranking[c] = hybrid_score

        top_results = sorted(brandnew_ranking.items(),key=lambda x: x[1],
                             reverse=True)[:top_k]
        return top_results


    def source_overlap(self, retrieved, ground_truth):
        if retrieved.file_path != ground_truth["file_path"]:
            return 0.0

        intersection_start = max(
            retrieved.first_character_index,
            ground_truth["first_character_index"]
        )

        intersection_end = min(
            retrieved.last_character_index,
            ground_truth["last_character_index"]
        )

        intersection = max(
            0,
            intersection_end - intersection_start
        )

        retrieved_length = (
            retrieved.last_character_index
            - retrieved.first_character_index
        )

        ground_truth_length = (
            ground_truth["last_character_index"]
            - ground_truth["first_character_index"]
        )

        union = retrieved_length + ground_truth_length - intersection

        if union <= 0:
            return 0.0

        return intersection / union


    def evaluate_dataset(self, questions, top_k=5):
        passed = 0

        for question in questions:
            query = question["question"]
            ground_truths = question["sources"]

            results = self.fusion(query, top_k)

            retrieved_chunks = [
                self.chunks[chunk_id]
                for chunk_id, score in results
            ]

            question_passed = False

            for chunk in retrieved_chunks:
                for ground_truth in ground_truths:
                    overlap = self.source_overlap(
                        chunk.source,
                        ground_truth
                    )

                    if overlap >= 0.05:
                        question_passed = True
                        break

                if question_passed:
                    break

            if question_passed:
                passed += 1

            print(
                f"{'PASS' if question_passed else 'FAIL'} "
                f"| {question['question_id']} "
                f"| {query}"
            )

        total = len(questions)

        if total > 0:
            score = passed / total * 100
            print(f"\nRecall@{top_k}: {score:.2f}%")