"""AST-based source chunker for RAG indexing, aligned with Ingester contract."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class _RawChunk:
    """A chunk boundary plus its context header chain, before text assembly."""

    file_path: str | Path
    first_idx: int
    last_idx: int
    prefix: str
    block_start: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


class PythonStructuralChunker:
    """Chunks Python source files structurally using AST node boundaries."""

    def __init__(self, file_path: str | Path, source_code: str, max_chunk_size: int = 1500):
        self.file_path = file_path
        self.source_code = source_code
        self.max_chunk_size = max_chunk_size
        self._line_offsets = self._compute_line_offsets(source_code)

    @staticmethod
    def _compute_line_offsets(source: str) -> list[int]:
        """Compute the starting character index for each line."""
        offsets = [0]
        for line in source.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return offsets

    def _pos_to_char(self, lineno: int, col_offset: int) -> int:
        """Convert 1-based (lineno, col_offset) to an absolute character index."""
        if lineno <= 0:
            return 0
        line_idx = lineno - 1
        if line_idx >= len(self._line_offsets):
            return len(self.source_code)
        return min(self._line_offsets[line_idx] + col_offset, len(self.source_code))

    def _get_node_span(self, node: ast.AST) -> tuple[int, int]:
        """Extract absolute start and end character indexes for an AST node."""
        start_line = getattr(node, "lineno", 1)
        start_col = getattr(node, "col_offset", 0)
        end_line = getattr(node, "end_lineno", start_line)
        end_col = getattr(node, "end_col_offset", start_col)

        start_idx = self._pos_to_char(start_line, start_col)
        end_idx = self._pos_to_char(end_line, end_col)
        return start_idx, end_idx

    def _finalize_chunk(self, raw_chunk: _RawChunk) -> dict[str, Any]:
        """Assemble the final text and return a dictionary matching Markdown chunk structure."""
        content = self.source_code[raw_chunk.first_idx : raw_chunk.last_idx]
        stripped_content = content.strip()

        lstrip_offset = content.find(stripped_content) if stripped_content else 0
        actual_start = raw_chunk.first_idx + lstrip_offset
        actual_end = actual_start + len(stripped_content)

        metadata = dict(raw_chunk.metadata)
        if raw_chunk.prefix:
            metadata["AST Context"] = raw_chunk.prefix
        if raw_chunk.block_start != -1:
            metadata["block_start"] = raw_chunk.block_start

        return {
            "content": stripped_content,
            "metadata": metadata,
            "start_char": actual_start,
            "end_char": actual_end,
            "file_path": self.file_path,
        }

    def _split_big_span(self, start_idx: int, end_idx: int, prefix: str, metadata: dict) -> list[_RawChunk]:
        """Recursively split spans that exceed max_chunk_size by lines."""
        span_length = end_idx - start_idx
        if span_length <= self.max_chunk_size or start_idx >= end_idx:
            return [_RawChunk(
                file_path=self.file_path,
                first_idx=start_idx,
                last_idx=end_idx,
                prefix=prefix,
                metadata=metadata
            )]

        sub_chunks = []
        current_start = start_idx
        
        while current_start < end_idx:
            current_end = min(current_start + self.max_chunk_size, end_idx)
            if current_end < end_idx:
                newline_pos = self.source_code.rfind('\n', current_start, current_end)
                if newline_pos > current_start:
                    current_end = newline_pos + 1

            sub_chunks.append(_RawChunk(
                file_path=self.file_path,
                first_idx=current_start,
                last_idx=current_end,
                prefix=prefix,
                metadata=metadata
            ))
            current_start = current_end

        return sub_chunks

    def chunk(self) -> list[dict[str, Any]]:
        """Parse source code and return a list of chunk dictionaries matching Ingester."""
        try:
            tree = ast.parse(self.source_code)
        except SyntaxError:
            raw = _RawChunk(
                file_path=self.file_path,
                first_idx=0,
                last_idx=len(self.source_code),
                prefix="module",
                metadata={"Type": "fallback"}
            )
            return [self._finalize_chunk(raw)]

        raw_chunks: list[_RawChunk] = []

        for node in tree.body:
            start_idx, end_idx = self._get_node_span(node)
            prefix = ""
            metadata: dict[str, Any] = {}

            if isinstance(node, ast.FunctionDef):
                prefix = f"def {node.name}"
                metadata["Type"] = "function"
                metadata["Name"] = node.name
            elif isinstance(node, ast.ClassDef):
                prefix = f"class {node.name}"
                metadata["Type"] = "class"
                metadata["Name"] = node.name
            elif isinstance(node, ast.AsyncFunctionDef):
                prefix = f"async def {node.name}"
                metadata["Type"] = "async_function"
                metadata["Name"] = node.name
            else:
                metadata["Type"] = type(node).__name__

            if (end_idx - start_idx) > self.max_chunk_size:
                raw_chunks.extend(self._split_big_span(start_idx, end_idx, prefix, metadata))
            else:
                raw_chunks.append(_RawChunk(
                    file_path=self.file_path,
                    first_idx=start_idx,
                    last_idx=end_idx,
                    prefix=prefix,
                    metadata=metadata
                ))

        if not raw_chunks and self.source_code.strip():
            raw_chunks.append(_RawChunk(
                file_path=self.file_path,
                first_idx=0,
                last_idx=len(self.source_code),
                prefix="module",
                metadata={"Type": "module"}
            ))

        return [self._finalize_chunk(rc) for rc in raw_chunks if rc.last_idx > rc.first_idx]

    def chunk_file(self) -> list[dict[str, Any]]:
        """Alias for chunk() to prevent attribute errors on execution."""
        return self.chunk()