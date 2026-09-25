"""AST-based source chunker for RAG indexing.

Design notes
------------
`split_big` never composes final chunk text. It emits `_RawChunk` records
carrying only a source span plus the *full* context header chain that applies
to that span. Text is assembled exactly once, in `_finalize_chunk`, after
`merge_smalls` has decided the final boundaries.

Sizing policy
-------------
Chunk boundaries are evaluated *strictly* against the raw source span
(`last_idx - first_idx`). Header/prefix length is never added to or subtracted
from the budget. Prefixes are structural context for display and retrieval
only; they never throttle boundary decisions.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
import json

from .pyd_models import Chunk, MinimalSource


@dataclass
class _RawChunk:
    """A chunk boundary plus its context header chain, before text assembly."""

    file_path: str
    first_idx: int
    last_idx: int
    prefix: str
    block_start: int = -1


class Ingester:
    def __init__(self, repository: str):
        self.repository = repository

    def find_python_files(self) -> list[Path]:
        """Find all Python files recursively in a repository."""
        repository = Path(self.repository)
        return list(repository.rglob("*.py"))

    def find_md_files(self) -> list[Path]:
        """Find all documents files recursively in a repository."""
        repository = Path(self.repository)
        return list(repository.rglob("*.md"))

    def find_txt_files(self) -> list[Path]:
        """Find all text files recursively in a repository."""
        repository = Path(self.repository)
        return list(repository.rglob("*.txt"))

    def wrapping_chunks(self, chunks: list[dict]) -> list[Chunk]:
            wrapped_chunks = []
            with_text = []

            for chunk in chunks:

                source = MinimalSource(
                    file_path=str(chunk["file_path"]),
                    first_character_index=chunk["start_char"],
                    last_character_index=chunk["end_char"]
                )
                wrapped_chunks.append(source)

                source_content = Chunk(source=source, text=chunk["content"], metadata=chunk['metadata'])
                with_text.append(source_content)
                
            return (wrapped_chunks, with_text)

    def saving_chunks(self, chunks, output: str = "data/processed/chunks/all_chunks.json"):
        out_file = Path(output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        wrapped_chunks = self.wrapping_chunks(chunks)
        serialized_data = [chunk.model_dump() for chunk in wrapped_chunks[0]]

        with open(output, 'w') as f:
            json.dump(serialized_data, f, indent=4)
            print("donneee")
        return wrapped_chunks[1]

class Markdown:
    def __init__(self, md_files, max_chunk_size):
        self.md_files = md_files
        self.max_chunk_size = max_chunk_size

    def split_by_headers(self):
        chunks: list[dict] = []

        for file in self.md_files:
            headers: list[str | None] = [None] * 7  # index 1..6 used, level -> title
            content: list[str] = []
            offset: int = 0
            section_start: int = 0  # raw offset right after the header line (untrimmed)

            def build_metadata() -> dict:
                return {
                    f"Header {level}": title
                    for level, title in enumerate(headers)
                    if title is not None
                }

            def add_chunk() -> None:
                if content:
                    raw_text: str = "".join(content)
                    stripped: str = raw_text.lstrip()
                    if stripped:
                        lstrip_len: int = len(raw_text) - len(raw_text.lstrip())
                        start_char: int = section_start + lstrip_len
                        chunks.append({
                            "content": stripped,
                            "metadata": build_metadata(),
                            "start_char": start_char,
                            "end_char": start_char + len(stripped),
                            "file_path": file
                        })
                    content.clear()

            with open(file, "r") as f:
                for line in f:
                    start: int = offset
                    offset += len(line)

                    stripped_line: str = line.strip()

                    if stripped_line.startswith("#"):
                        parts: list[str] = stripped_line.split(" ", 1)
                        hashes: str = parts[0]

                        is_header: bool = (
                            len(parts) == 2
                            and 1 <= len(hashes) <= 6
                            and hashes.count("#") == len(hashes)
                        )

                        if is_header:
                            add_chunk()

                            level: int = len(hashes)
                            headers[level] = parts[1].strip()
                            headers[level + 1:] = [None] * (6 - level)

                            section_start = offset
                            continue

                    content.append(line)

                add_chunk()

        return chunks

    def split_paragraphs(self, chunks: list[dict]) -> list[dict]:
        new_chunks: list[dict] = []

        for chunk in chunks:
            content: str = chunk["content"]
            chunk_base: int = chunk["start_char"]

            if len(content) < self.max_chunk_size:
                new_chunks.append(chunk)
                continue

            paragraphs: list[str] = re.split(r"\n\s*\n+", content)
            search_pos: int = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                local_start: int = content.find(para, search_pos)
                local_end: int = local_start + len(para)
                search_pos = local_end

                new_chunks.append({
                    "content": para,
                    "metadata": chunk["metadata"].copy(),
                    "start_char": chunk_base + local_start,
                    "end_char": chunk_base + local_end,
                    "file_path": chunk['file_path']
                })

        return new_chunks

    def split_lines(self, chunks: list[dict]) -> list[dict]:
            new_chunks: list[dict] = []

            for chunk in chunks:
                content: str = chunk["content"]
                chunk_start: int = chunk["start_char"]

                # If it fits within the limit, keep it as is
                if len(content) <= self.max_chunk_size:
                    new_chunks.append(chunk)
                    continue

                # Otherwise, apply greedy line accumulation
                current_chunk_lines: list[str] = []
                current_length: int = 0
                sub_chunk_start: int = chunk_start
                
                lines = content.splitlines(keepends=True)
                
                for line in lines:
                    line_len = len(line)
                    
                    # If adding this line exceeds the limit, save the current chunk
                    if current_length + line_len > self.max_chunk_size and current_chunk_lines:
                        chunk_text = "".join(current_chunk_lines)
                        new_chunks.append({
                            "content": chunk_text,
                            "metadata": chunk["metadata"].copy(),
                            "start_char": sub_chunk_start,
                            "end_char": sub_chunk_start + len(chunk_text),
                            "file_path": chunk["file_path"]
                        })
                        
                        # Move start forward and start fresh with the current line
                        sub_chunk_start += len(chunk_text)
                        current_chunk_lines = [line]
                        current_length = line_len
                    else:
                        current_chunk_lines.append(line)
                        current_length += line_len

                # Save the final remaining lines
                if current_chunk_lines:
                    chunk_text = "".join(current_chunk_lines)
                    new_chunks.append({
                        "content": chunk_text,
                        "metadata": chunk["metadata"].copy(),
                        "start_char": sub_chunk_start,
                        "end_char": sub_chunk_start + len(chunk_text),
                        "file_path": chunk["file_path"]
                    })

            return new_chunks


    def split_charachters(self, chunks: list[dict]) -> list[dict]:
            new_chunks: list[dict] = []

            # Tiers searched in priority order: sentence enders first, then
            # clause punctuation, then plain whitespace. Hard-cut if none found.
            checkpoint_tiers: list[str] = [
                ".!?…",
                ";:,",
                " \t",
            ]

            for chunk in chunks:
                content: str = chunk["content"]
                chunk_start: int = chunk["start_char"]
                content_len: int = len(content)

                # If it fits within the limit, keep it as is
                if content_len <= self.max_chunk_size:
                    new_chunks.append(chunk)
                    continue

                pos: int = 0

                while pos < content_len:
                    remaining: int = content_len - pos

                    if remaining <= self.max_chunk_size:
                        piece_end: int = content_len
                    else:
                        window_end: int = pos + self.max_chunk_size
                        cut_index: int = -1

                        # Search backward from the window end for the highest-priority checkpoint
                        for tier in checkpoint_tiers:
                            search_pos: int = window_end - 1
                            while search_pos > pos:
                                if content[search_pos] in tier:
                                    cut_index = search_pos
                                    break
                                search_pos -= 1
                            if cut_index != -1:
                                break

                        # Checkpoint char stays with the current piece; fallback to hard slice
                        piece_end = cut_index + 1 if cut_index != -1 else window_end

                    piece: str = content[pos:piece_end]
                    new_chunks.append({
                        "content": piece,
                        "metadata": chunk["metadata"].copy(),
                        "start_char": chunk_start + pos,
                        "end_char": chunk_start + piece_end,
                        "file_path": chunk["file_path"]
                    })

                    pos = piece_end

            return new_chunks


import bm25s
import Stemmer


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
            for m in chunk.metadata:
                text += f"{m} -> {chunk.metadata[m]}\n"
            chunk.text = text + chunk.text

        texts = [chunk.text for chunk in self.chunks]
        print(f"Tokenizing {len(texts)} chunks...")
        stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(texts, stemmer=stemmer)
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