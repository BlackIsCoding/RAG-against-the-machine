from .pyd_models import Chunk, MinimalSource
from pathlib import Path
import Stemmer
import bm25s
import sentence_transformers
import numpy as np

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

    def bm25s_indexing(self):
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
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        self.saving_path.mkdir(parents=True, exist_ok=True)
        retriever.save(self.saving_path)
        print(f"BM25 index successfully built and saved to {self.saving_path}")

    def semantic_search(self, query, top_k):
        model = self.model
        query = model.encode(query, normalize_embeddings=True)
        similarity = self.encoded_chunks @ query
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