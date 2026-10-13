from __future__ import annotations
import json
import math
import pathlib
import fire
import Stemmer
import bm25s
import sentence_transformers
import numpy as np
from pydantic import BaseModel, Field

# ==========================================
# 1. MODELS (from pyd_models.py equivalent)
# ==========================================

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class Chunk(BaseModel):
    source: MinimalSource
    text: str
    metadata: list[str] = Field(default_factory=list)

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]

class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


# ==========================================
# 2. INGESTER (from ingest.py equivalent)
# ==========================================

class Ingester:
    def __init__(self, repository: str | None):
        self.repository = repository

    def find_python_files(self) -> list[pathlib.Path]:
        if not self.repository:
            return []
        repository = pathlib.Path(self.repository)
        return list(repository.rglob("*.py"))

    def find_md_files(self) -> list[pathlib.Path]:
        if not self.repository:
            return []
        repository = pathlib.Path(self.repository)
        return list(repository.rglob("*.md"))

    def find_txt_files(self) -> list[pathlib.Path]:
        if not self.repository:
            return []
        repository = pathlib.Path(self.repository)
        return list(repository.rglob("*.txt"))

    def wrapping_chunks(self, chunks: list[dict]) -> tuple[list[MinimalSource], list[Chunk]]:
        wrapped_chunks = []
        with_text = []

        for chunk in chunks:
            source = MinimalSource(
                file_path=str(chunk["file_path"]),
                first_character_index=chunk["start_char"],
                last_character_index=chunk["end_char"]
            )
            wrapped_chunks.append(source)

            source_content = Chunk(source=source, text=chunk["content"], metadata=chunk.get('metadata', []))
            with_text.append(source_content)
            
        return wrapped_chunks, with_text

    def saving_chunks(self, chunks, output: str = "data/processed/chunks/all_chunks.json"):
        out_file = pathlib.Path(output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        _, with_text = self.wrapping_chunks(chunks)
        serialized_data = [chunk.model_dump() for chunk in with_text]

        with open(output, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, indent=4, ensure_ascii=False)
            
        return with_text


# ==========================================
# 3. INDEXER (from index.py equivalent)
# ==========================================

class Indexer:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks
        self.saving_path = pathlib.Path("data/processed/")

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
        length = sum(len(chunk) for chunk in chunks)
        avgdl = length / len(chunks) if len(chunks) else 0
        return len(chunks), avgdl

    def bm25_score(self, tf_dict, idf_dict, dl, avgdl, k1=1.5, b=0.75):
        bm25_score = 0
        for token, tf in tf_dict.items():
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)
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

    def bm25s_indexing(self):
        self.saving_path.mkdir(parents=True, exist_ok=True)
        if not self.chunks:
            return

        for chunk in self.chunks:
            text = ''
            for i, m in enumerate(chunk.metadata):
                text += (' ' if i != 0 else '') + m
            chunk.text = text + '\n' + chunk.text

        texts = [chunk.text for chunk in self.chunks]
        stemmer = Stemmer.Stemmer("english")
        tokenized_corpus = bm25s.tokenize(texts, stemmer=stemmer)

        index_data = {
            "tokenized_ids": [list(arr) for arr in tokenized_corpus.ids],
            "vocab": tokenized_corpus.vocab
        }

        index_file = self.saving_path / "custom_bm25_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False)

    def lexical_search(self, query: str, top_k: int = 5):
        index_file = self.saving_path / "custom_bm25_index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Custom BM25 index not found at {index_file}.")

        with open(index_file, "r", encoding="utf-8") as f:
            saved_index = json.load(f)
            en_chunks = saved_index["tokenized_ids"]

        chunks_text = [chunk.text for chunk in self.chunks] if self.chunks else [""] * len(en_chunks)
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


# ==========================================
# 4. CLI COMMANDS & UTILITIES (__main__.py)
# ==========================================

def index(max_chunk_size=2000):
    ingester = Ingester("data/raw/vllm-0.10.1")
    md = ingester.find_txt_files() + ingester.find_md_files()
    
    # Minimal stub parsers if external markdown/python parsers are missing in your environment; 
    # replace/hook up your custom chunkers here if they are in separate local files.
    chunks = [] 

    py_files = ingester.find_python_files()
    # py_parser setup if needed...

    objected_chunks = ingester.saving_chunks(chunks)
    indexer = Indexer(objected_chunks)
    indexer.bm25s_indexing()

def revert_saved_chunks(saved_chunks: list[dict]) -> list[dict]:
    flat_chunks = []
    for item in saved_chunks:
        source = item.get("source", {})
        flat_chunks.append({
            "file_path": source.get("file_path"),
            "start_char": source.get("first_character_index"),
            "end_char": source.get("last_character_index"),
            "content": item.get("text"),
            "metadata": item.get("metadata", [])
        })
    return flat_chunks

def search(query='', k=5):
    with open("data/processed/chunks/all_chunks.json", "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    flat_chunks = revert_saved_chunks(chunks_data)
    ingest = Ingester(None)
    _, chunks_with_text = ingest.wrapping_chunks(flat_chunks)
    
    index_obj = Indexer(chunks_with_text)

    if query:
        results, scores = index_obj.lexical_search(query, k)
        if len(results) == 0:
            print("No matching chunks found.")
            return

        for r, score in zip(results, scores):
            chunk = chunks_with_text[int(r)]
            print(f"[{score:.2f}] {chunk.source.file_path} [{chunk.source.first_character_index}:{chunk.source.last_character_index}]")

def search_dataset(dataset_path, k, save_directory):
    path = pathlib.Path(dataset_path)
    if not path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data.get("rag_questions", [])
    if not questions:
        print("Error: No questions found in the dataset.")
        return

    chunks_file = pathlib.Path("data/processed/chunks/all_chunks.json")
    if not chunks_file.exists():
        print(f"Error: Chunks file not found at {chunks_file}.")
        return

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    chunks_with_text = [
        Chunk(
            source=MinimalSource(
                file_path=item["source"]["file_path"],
                first_character_index=item["source"]["first_character_index"],
                last_character_index=item["source"]["last_character_index"]
            ),
            text=item["text"],
            metadata=item["metadata"]
        )
        for item in chunks_data
    ]
    
    index_obj = Indexer(chunks_with_text)
    min_search_result_list = []
    
    for question in questions:
        q_id = question['question_id']
        q_text = question['question']

        results, _ = index_obj.lexical_search(q_text, k)

        minimal_list = []
        for r in results:
            chunk = chunks_with_text[int(r)]
            source = MinimalSource(
                file_path=str(chunk.source.file_path),
                first_character_index=chunk.source.first_character_index,
                last_character_index=chunk.source.last_character_index
            )
            minimal_list.append(source)
        
        min_search_result = MinimalSearchResults(
            question_id=q_id,
            question=q_text,
            retrieved_sources=minimal_list
        )
        min_search_result_list.append(min_search_result)

    student_results = StudentSearchResults(
        search_results=min_search_result_list,
        k=k
    )

    save_dir = pathlib.Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = save_dir / path.name
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(student_results.model_dump(), f, indent=4)

    print(f"Saved student_search_results to {output_file}")

def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    student_path = pathlib.Path(student_search_results_path)
    gt_path = pathlib.Path(dataset_path)

    if not student_path.exists() or not gt_path.exists():
        print("Error: Path does not exist.")
        return

    with open(student_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    student_results_dict = {
        item["question_id"]: item
        for item in student_data.get("search_results", [])
    }
    gt_questions = gt_data.get("rag_questions", [])

    k_values = [1, 3, 5, 10]
    recall_counts = {k: 0 for k in k_values}
    total_evaluated = 0

    for gt_q in gt_questions:
        q_id = gt_q.get("question_id")
        gt_sources = gt_q.get("sources", [])

        if not gt_sources or q_id not in student_results_dict:
            continue

        total_evaluated += 1
        student_sources = student_results_dict[q_id].get("retrieved_sources", [])

        for k in k_values:
            top_k = student_sources[:k]
            found = False

            for gt_src in gt_sources:
                gt_file = gt_src["file_path"]
                gt_start = gt_src["first_character_index"]
                gt_end = gt_src["last_character_index"]

                for st_src in top_k:
                    if gt_file != st_src["file_path"]:
                        continue

                    st_start = st_src["first_character_index"]
                    st_end = st_src["last_character_index"]

                    inter_start = max(gt_start, st_start)
                    inter_end = min(gt_end, st_end)
                    intersection = max(0, inter_end - inter_start)

                    gt_len = max(0, gt_end - gt_start)
                    st_len = max(0, st_end - st_start)
                    union = gt_len + st_len - intersection

                    iou = intersection / union if union > 0 else 0.0

                    if iou >= 0.05:
                        found = True
                        break

                if found:
                    break

            if found:
                recall_counts[k] += 1

    print(f"Total evaluated questions: {total_evaluated}")
    print("Evaluation Results (IoU >= 0.05):")

    for k in k_values:
        score = recall_counts[k] / total_evaluated if total_evaluated > 0 else 0.0
        print(f"Recall@{k}: {score:.3f} ({score * 100:.1f}%)")

if __name__ == "__main__":
    fire.Fire({
        "index": index, 
        "search": search,
        "search_dataset": search_dataset,
        "evaluate": evaluate
    })