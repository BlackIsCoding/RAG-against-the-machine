from .pyd_models import Chunk, MinimalSource
from pathlib import Path
import Stemmer
import bm25s

def compute_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Compute intersection-over-union between two [start, end) ranges."""
    a_start, a_end = a
    b_start, b_end = b

    intersection: int = max(0, min(a_end, b_end) - max(a_start, b_start))
    union: int = (a_end - a_start) + (b_end - b_start) - intersection

    if union == 0:
        return 0.0

    return intersection / union

class Indexer():
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.saving_path = Path("data/processed/")

    def indexing_chunks(self):
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

    def search(self, query: str, top_k: int = 5):
        """Load the BM25 index, query it, and map row IDs back to your in-memory chunks."""
        if not self.saving_path.exists():
            raise FileNotFoundError(f"Index not found at {self.saving_path}. Run 'index' first!")

        retriever = bm25s.BM25.load(str(self.saving_path), load_corpus=False)
        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize([query], stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=top_k)
        print(f"\nQuery: '{query}'")
        print("=" * 40)

        self.last_results: list[Chunk] = []

        for i in range(results.shape[1]):
            doc_idx = results[0, i]
            score = scores[0, i]
            
            matched_chunk: Chunk = self.chunks[doc_idx]
            self.last_results.append(matched_chunk)
            source: MinimalSource = matched_chunk.source
            
            print(f"Rank {i+1} (Score: {score:.2f})")
            print(f"  File Path: {source.file_path}")
            print(f"  Character Range: {source.first_character_index} - {source.last_character_index}")
            print(f"  Snippet: {matched_chunk.text}...\n")

    def evaluate_all(self, questions: list[dict], top_k: int = 5) -> None:
        threshold = 0.05
        passed = 0

        for number, question in enumerate(questions, start=1):
            query = question["question"]

            source = question["sources"][0]

            file_path = source["file_path"]
            gt_range = (
                source["first_character_index"],
                source["last_character_index"],
            )

            self.search(query, top_k=top_k)

            matching = [
                chunk
                for chunk in self.last_results
                if chunk.source.file_path == file_path
            ]

            question_passed = False
            best_iou = 0.0

            for chunk in matching:
                chunk_range = (
                    chunk.source.first_character_index,
                    chunk.source.last_character_index,
                )

                iou = compute_iou(chunk_range, gt_range)
                best_iou = max(best_iou, iou)

                if iou >= threshold:
                    question_passed = True
                    break

            if question_passed:
                passed += 1

            status = "PASS" if question_passed else "FAIL"

            print(
                f"[{number}/{len(questions)}] "
                f"{status} | IoU: {best_iou:.4f} | "
                f"{query}"
            )

        total = len(questions)
        pass_rate = passed / total * 100 if total else 0

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Total questions : {total}")
        print(f"Passed          : {passed}")
        print(f"Failed          : {total - passed}")
        print(f"Pass rate       : {pass_rate:.2f}%")