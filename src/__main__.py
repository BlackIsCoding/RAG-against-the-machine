"""Command-line interface for the RAG pipeline (run: python -m src <cmd>)."""
import functools
import sys
from pathlib import Path
from typing import Any, Callable

import fire  # type: ignore
from tqdm import tqdm

from .markdown import Header
from .chunking_pyy import Python  # ADAPT to your chunker
from .index import Indexer
from .ingest import Ingester
from .pyd_models import (
    AnsweredQuestion,
    Chunk,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

RAW_DIR: str = "data/raw"
PROCESSED_DIR: str = "data/processed"
MAX_CHUNK_LIMIT: int = 2000
IOU_THRESHOLD: float = 0.05
RECALL_KS: tuple[int, ...] = (1, 3, 5, 10)
MODEL_NAME: str = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARS: int = 8000

_MODEL_CACHE: dict[str, Any] = {}


def _safe(func: Callable[..., Any]) -> Callable[..., Any]:
    """Turn any exception into a clean error message (no traceback)."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None
    return wrapper


def _load_index(processed_dir: str) -> tuple[Indexer, list[MinimalSource]]:
    """Load the saved sources and an Indexer pointing at the saved index."""
    chunks_file: Path = Path(processed_dir) / "chunks" / "all_chunks.json"
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"{chunks_file} not found, run the `index` command first"
        )
    sources: list[MinimalSource] = [
        MinimalSource.model_validate(item)
        for item in __import__("json").loads(
            chunks_file.read_text(encoding="utf-8")
        )
    ]
    indexer: Indexer = Indexer([])
    indexer.saving_path = Path(processed_dir)
    return indexer, sources


def _retrieve(indexer: Indexer, sources: list[MinimalSource],
              query: str, k: int) -> list[MinimalSource]:
    """Return the top-k sources for one query."""
    ids, _scores = indexer.lexical_search(query, top_k=k)
    return [sources[int(i)] for i in ids if int(i) < len(sources)]


def _check_query(query: Any, k: int) -> str:
    """Validate a single query and k, return the cleaned query."""
    cleaned: str = str(query).strip()
    if not cleaned:
        raise ValueError("the query is empty")
    if k <= 0:
        raise ValueError("k must be greater than 0")
    return cleaned


def _read_span(source: MinimalSource) -> str:
    """Read the text covered by a source from disk."""
    try:
        text: str = Path(source.file_path).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return ""
    return text[source.first_character_index:source.last_character_index]


def _generate(question: str, sources: list[MinimalSource]) -> str:
    """Answer a question from the retrieved sources with Qwen3-0.6B."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "model" not in _MODEL_CACHE:
        _MODEL_CACHE["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL_CACHE["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype="auto"
        )
    tokenizer: Any = _MODEL_CACHE["tokenizer"]
    model: Any = _MODEL_CACHE["model"]

    context: str = ""
    for source in sources:
        piece: str = f"[{source.file_path}]\n{_read_span(source)}\n\n"
        if len(context) + len(piece) > MAX_CONTEXT_CHARS:
            break
        context += piece

    messages: list[dict[str, str]] = [
        {"role": "system",
         "content": "Answer the question using only the provided context. "
                    "If the context does not contain the answer, say so. "
                    "Be concise."},
        {"role": "user",
         "content": f"Context:\n{context}\nQuestion: {question}"},
    ]
    prompt: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs: Any = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output: Any = model.generate(
            **inputs, max_new_tokens=256, do_sample=False
        )
    new_tokens: Any = output[0][inputs["input_ids"].shape[1]:]
    return str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


class Cli:
    """RAG against the machine: index, search, answer, evaluate."""

    @_safe
    def index(self, max_chunk_size: int = MAX_CHUNK_LIMIT,
              raw_dir: str = RAW_DIR,
              processed_dir: str = PROCESSED_DIR) -> None:
        """Chunk data/raw and build the BM25 index under data/processed."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than 0")
        if max_chunk_size > MAX_CHUNK_LIMIT:
            print(f"max_chunk_size capped to {MAX_CHUNK_LIMIT}")
            max_chunk_size = MAX_CHUNK_LIMIT

        ingester: Ingester = Ingester(raw_dir)
        files: list[Path] = (
            ingester.find_python_files()
            + ingester.find_md_files()
            + ingester.find_txt_files()
        )
        if not files:
            raise FileNotFoundError(f"no .py/.md/.txt files under {raw_dir}")

        all_chunks: list[dict[str, Any]] = []
        for path in tqdm(files, desc="Chunking", unit="file"):
            chunker: Callable[[Path, int], list[dict[str, Any]]] = (
                chunk_python if path.suffix == ".py" else chunk_markdown
            )
            all_chunks.extend(chunker(path, max_chunk_size))

        too_long: int = sum(
            1 for c in all_chunks
            if c["end_char"] - c["start_char"] > max_chunk_size
        )
        if too_long:
            print(f"Warning: {too_long} chunks exceed {max_chunk_size} chars")

        chunks_file: Path = Path(processed_dir) / "chunks" / "all_chunks.json"
        chunks: list[Chunk] = ingester.saving_chunks(
            all_chunks, output=str(chunks_file)
        )
        indexer: Indexer = Indexer(chunks)
        indexer.saving_path = Path(processed_dir)
        indexer.bm25s_indexing()
        print(f"Ingestion complete! Indexed {len(chunks)} chunks "
              f"under {processed_dir}/")

    @_safe
    def search(self, query: str, k: int = 10,
               processed_dir: str = PROCESSED_DIR) -> None:
        """Print the top-k sources for a single query."""
        cleaned: str = _check_query(query, k)
        indexer, sources = _load_index(processed_dir)
        for src in _retrieve(indexer, sources, cleaned, k):
            print(f"{src.file_path} "
                  f"[{src.first_character_index}:{src.last_character_index}]")

    @_safe
    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results",
                       processed_dir: str = PROCESSED_DIR) -> None:
        """Run search over a dataset and save a StudentSearchResults JSON."""
        if k <= 0:
            raise ValueError("k must be greater than 0")
        dataset: RagDataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        indexer, sources = _load_index(processed_dir)

        results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching",
                             unit="question"):
            found: list[MinimalSource] = (
                _retrieve(indexer, sources, question.question, k)
                if question.question.strip() else []
            )
            results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=found,
            ))

        output: StudentSearchResults = StudentSearchResults(
            search_results=results, k=k
        )
        out_file: Path = Path(save_directory) / Path(dataset_path).name
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(output.model_dump_json(indent=2),
                            encoding="utf-8")
        print(f"Saved student_search_results to {out_file}")

    @_safe
    def answer(self, query: str, k: int = 5,
               processed_dir: str = PROCESSED_DIR) -> None:
        """Retrieve for one query and generate an answer."""
        cleaned: str = _check_query(query, k)
        indexer, sources = _load_index(processed_dir)
        found: list[MinimalSource] = _retrieve(indexer, sources, cleaned, k)
        print(_generate(cleaned, found))

    @_safe
    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str = (
                           "data/output/search_results_and_answer"),
                       ) -> None:
        """Generate answers from saved search results."""
        student: StudentSearchResults = (
            StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(encoding="utf-8")
            )
        )
        answers: list[MinimalAnswer] = []
        for res in tqdm(student.search_results, desc="Answering",
                        unit="question"):
            text: str = (
                _generate(res.question, res.retrieved_sources)
                if res.question.strip() else ""
            )
            answers.append(MinimalAnswer(
                question_id=res.question_id,
                question=res.question,
                retrieved_sources=res.retrieved_sources,
                answer=text,
            ))

        output: StudentSearchResultsAndAnswer = StudentSearchResultsAndAnswer(
            search_results=answers, k=student.k
        )
        out_file: Path = (
            Path(save_directory) / Path(student_search_results_path).name
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(output.model_dump_json(indent=2),
                            encoding="utf-8")
        print(f"Saved student_search_results_and_answer to {out_file}")

    @_safe
    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print recall@k against a ground-truth (answered) dataset."""
        student: StudentSearchResults = (
            StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(encoding="utf-8")
            )
        )
        truth: RagDataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        truth_by_id: dict[str, AnsweredQuestion] = {
            q.question_id: q for q in truth.rag_questions
            if isinstance(q, AnsweredQuestion)
        }
        overlap: Indexer = Indexer([])

        print("Student data is valid: True")
        for k_val in RECALL_KS:
            if k_val > student.k:
                continue
            total: float = 0.0
            counted: int = 0
            for res in student.search_results:
                gt: AnsweredQuestion | None = truth_by_id.get(res.question_id)
                if gt is None or not gt.sources:
                    continue
                top: list[MinimalSource] = res.retrieved_sources[:k_val]
                hits: int = sum(
                    1 for g in gt.sources
                    if any(overlap.source_overlap(r, g.model_dump())
                           >= IOU_THRESHOLD for r in top)
                )
                total += hits / len(gt.sources)
                counted += 1
            if counted:
                print(f"Recall@{k_val}: {total / counted:.3f}")


if __name__ == "__main__":
    fire.Fire(Cli)