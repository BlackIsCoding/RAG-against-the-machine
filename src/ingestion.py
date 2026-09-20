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
                    file_path=chunk["file_path"],
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
        return wrapped_chunks[1]

class Markdown:
    def __init__(self, md_files):
        self.md_files = md_files

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
                    stripped: str = raw_text.strip()
                    if stripped:
                        lstrip_len: int = len(raw_text) - len(raw_text.lstrip())
                        start_char: int = section_start + lstrip_len
                        chunks.append({
                            "content": stripped,
                            "metadata": build_metadata(),
                            "start_char": start_char,
                            "end_char": start_char + len(stripped),
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

    def split_paragraphs(self, chunks: list[dict], max_chunk_size: int = 20) -> list[dict]:
        new_chunks: list[dict] = []

        for chunk in chunks:
            content: str = chunk["content"]
            chunk_base: int = chunk["start_char"]

            if len(content) < max_chunk_size:
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
                })

        return new_chunks

import bm25s
import Stemmer
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


if __name__ == "__main__":
    # 1. Create a dummy markdown file for testing
    test_filename = "t.md"

    # 2. Instantiate and run your splitter
    splitter = Markdown(md_files=[test_filename])
    extracted_chunks = splitter.split_by_headers()
    extracted_chunks = splitter.split_paragraphs(extracted_chunks)

    # 3. Print results to inspect
    print(f"Total Chunks Extracted: {len(extracted_chunks)}\n" + "="*50)
    for i, chunk in enumerate(extracted_chunks):
        print(f"Chunk #{i + 1}")
        print(f"Metadata : {chunk['metadata']}")
        print(f"Offsets  : Chars [{chunk['start_char']} -> {chunk['end_char']}]")
        print(f"Content  :\n{chunk['content']}")
        print("-" * 50)