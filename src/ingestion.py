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

    def find_docs_files(self) -> list[Path]:
        """Find all documents files recursively in a repository."""
        return []


class Indexer:
    _FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
    _LOOP_TYPES = (ast.If, ast.For, ast.AsyncFor, ast.While)
    _WITH_TYPES = (ast.With, ast.AsyncWith)
    _TRY_TYPES = tuple(t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None)

    _DEF_RE = re.compile(r"^\s*(?:async\s+def|def)\s+(\w+)")
    _CLASS_RE = re.compile(r"^\s*class\s+(\w+)")

    _MAX_PREFIX_CHARS = 400

    def __init__(self, py_files: list[str]):
        self.py_files = py_files

    # ------------------------------------------------------------------ #
    # Offset helpers -- exact spans from AST position metadata
    # ------------------------------------------------------------------ #

    def _build_line_index(self, source_code: str) -> tuple[list[int], list[bytes]]:
        """Build the position index used to convert AST coordinates to char offsets."""
        starts: list[int] = [0]
        raw_lines: list[bytes] = []
        for line in source_code.splitlines(keepends=True):
            raw_lines.append(line.encode("utf-8"))
            starts.append(starts[-1] + len(line))
        return starts, raw_lines

    def _pos(self, lineno: int, col: int, line_index: tuple[list[int], list[bytes]]) -> int:
        """Convert a 1-indexed (line, utf8-byte-column) pair to a char offset."""
        starts, raw_lines = line_index
        i: int = lineno - 1
        if i < 0:
            return 0
        if i >= len(raw_lines):
            return starts[-1]
        prefix_bytes: bytes = raw_lines[i][:col]
        return starts[i] + len(prefix_bytes.decode("utf-8", errors="ignore"))

    def _node_own_span(
        self, node: ast.AST, line_index: tuple[list[int], list[bytes]]
    ) -> tuple[int, int]:
        """Span of the node itself, excluding any decorators."""
        start: int = self._pos(node.lineno, node.col_offset, line_index)
        end: int = self._pos(node.end_lineno, node.end_col_offset, line_index)
        return start, end

    def _node_full_span(self, node: ast.AST, line_index: tuple[list[int], list[bytes]]) -> tuple[int, int]:
        """Span including decorators, when present."""
        decorators: list[ast.expr] = getattr(node, "decorator_list", None) or []
        if not decorators:
            return self._node_own_span(node, line_index)
        first_dec = decorators[0]
        start: int = max(self._pos(first_dec.lineno, first_dec.col_offset, line_index) - 1, 0)
        _, end = self._node_own_span(node, line_index)
        return start, end

    def _get_header_line(self, node: ast.AST, source_code: str, line_index: tuple[list[int], list[bytes]]) -> str:
        """Header of a compound statement collapsed onto one line."""
        start, end = self._node_own_span(node, line_index)
        segment: str = source_code[start:end]

        depth: int = 0
        for i, ch in enumerate(segment):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == ":" and depth == 0:
                header: str = segment[: i + 1]
                return " ".join(l.strip() for l in header.splitlines() if l.strip())

        return " ".join(l.strip() for l in segment.splitlines() if l.strip())

    def _get_block_header(
        self, source_code: str, window_start: int, window_end: int, keyword: str
    ) -> tuple[str, int]:
        """Locate a bare keyword header (`else:`, `finally:`) that owns no AST node."""
        window: str = source_code[window_start:window_end]
        offset: int = window_start
        for line in window.splitlines(keepends=True):
            stripped: str = line.strip()
            if stripped == f"{keyword}:" or (
                stripped.startswith(keyword) and stripped.endswith(":")
            ):
                return stripped, offset + (len(line) - len(line.lstrip()))
            offset += len(line)
        return f"{keyword}:", -1

    # ------------------------------------------------------------------ #
    # Prefix / scope helpers
    # ------------------------------------------------------------------ #

    def _prefix_lines(self, prefix: str) -> list[str]:
        return [l for l in prefix.splitlines() if l.strip()]

    def _join_prefix(self, parent_prefix: str, header: str) -> str:
        header = header.strip()
        if not header:
            return parent_prefix
        return f"{parent_prefix}\n{header}" if parent_prefix else header

    def _common_prefix_lines(self, a: str, b: str) -> list[str]:
        """Longest shared leading run of header lines between two chains."""
        lines_a: list[str] = self._prefix_lines(a)
        lines_b: list[str] = self._prefix_lines(b)
        common: list[str] = []
        for la, lb in zip(lines_a, lines_b):
            if la.strip() != lb.strip():
                break
            common.append(la)
        return common

    def _get_scope_key(self, prefix: str) -> str:
        """Innermost enclosing def/class of a header chain."""
        for line in reversed(self._prefix_lines(prefix)):
            m = self._DEF_RE.match(line)
            if m:
                return f"def:{m.group(1)}"
            m = self._CLASS_RE.match(line)
            if m:
                return f"class:{m.group(1)}"
        return "<module>"

    def _sanitize_prefix(self, prefix: str) -> str:
        """Render-time trimming only."""
        if not prefix or len(prefix) <= self._MAX_PREFIX_CHARS:
            return prefix

        lines: list[str] = self._prefix_lines(prefix)
        headers: list[str] = [l for l in lines if l.rstrip().endswith(":")]
        candidate: str = "\n".join(headers)
        if headers and len(candidate) <= self._MAX_PREFIX_CHARS:
            return candidate

        if not headers:
            return lines[-1] if lines else ""

        if len(headers) >= 2:
            return f"{headers[0]}\n{headers[-1]}"
        return headers[-1]

    def _strip_redundant_header(self, prefix_lines: list[str], raw_slice: str) -> list[str]:
        """Drop any header line from prefix if it already exists in raw body slice."""
        if not prefix_lines:
            return prefix_lines

        raw_lines: list[str] = [l.strip() for l in raw_slice.splitlines()]
        filtered_headers: list[str] = []
        for header in prefix_lines:
            header_clean = header.strip()
            if not any(r == header_clean or r.startswith(header_clean) for r in raw_lines):
                filtered_headers.append(header)
                
        return filtered_headers

    def _get_body_children(self, node: ast.AST) -> list[ast.stmt]:
        """Body statements, bypassing a leading docstring Expr."""
        body: list[ast.stmt] = list(getattr(node, "body", []))
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def _get_function_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_code: str,
        line_index: tuple[list[int], list[bytes]],
    ) -> str:
        """Decorators + signature + docstring, folded into one header chain."""
        decorator_lines: list[str] = []
        for dec in getattr(node, "decorator_list", []):
            dec_start, dec_end = self._node_own_span(dec, line_index)
            dec_text: str = source_code[dec_start:dec_end].strip()
            if dec_text:
                decorator_lines.append(f"@{dec_text}")

        sig: str = self._get_header_line(node, source_code, line_index)

        doc: str | None = ast.get_docstring(node, clean=False)
        if doc:
            sig = f'{sig}\n    """{doc.strip()}"""'

        if decorator_lines:
            return "\n".join(decorator_lines) + "\n" + sig
        return sig

    def _compose_class_header(
        self, node: ast.ClassDef, source_code: str, line_index: tuple[list[int], list[bytes]]
    ) -> str:
        header: str = self._get_header_line(node, source_code, line_index)
        doc: str | None = ast.get_docstring(node, clean=False)
        if doc:
            header = f'{header}\n    """{doc.strip()}"""'
        return header

    def _compound_header_end(
        self,
        node: ast.AST,
        source_code: str,
        line_index: tuple[list[int], list[bytes]],
    ) -> int:
        """Return the end offset of the node's own compound header."""
        start, node_end = self._node_own_span(node, line_index)
        segment = source_code[start:node_end]
        depth = 0
        for i, ch in enumerate(segment):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == ":" and depth == 0:
                return start + i + 1
        return min(node_end, start)

    def _header_chunk(
        self,
        first_idx: int,
        header_end: int,
        file_path: str,
        parent_prefix: str,
        max_chunk_size: int,
        source_code: str,
        block_start: int = -1,
    ) -> list[_RawChunk]:
        """Emit the node-owned header as counted content, not metadata."""
        if header_end <= first_idx:
            return []
        if header_end - first_idx <= max_chunk_size:
            return [_RawChunk(file_path, first_idx, header_end, parent_prefix, block_start)]
        return self._split_lines(
            first_idx, header_end, file_path, parent_prefix,
            max_chunk_size, source_code, block_start,
        )

    # ------------------------------------------------------------------ #
    # Recursive decomposition
    # ------------------------------------------------------------------ #

    def _process_children(
        self,
        children: list[ast.stmt],
        source_code: str,
        file_path: str,
        start_cursor: int,
        prefix: str,
        max_chunk_size: int,
        line_index: tuple[list[int], list[bytes]],
        block_start: int = -1,
    ) -> list[_RawChunk]:
        """Shared statement-list walker used for every compound block type."""
        result: list[_RawChunk] = []
        cursor: int = start_cursor
        for child in children:
            sub: list[_RawChunk] = self.split_big(
                child, source_code, file_path, cursor, prefix, max_chunk_size,
                line_index, block_start,
            )
            if sub:
                result.extend(sub)
                cursor = sub[-1].last_idx
        return result

    def _split_lines(
        self,
        first_idx: int,
        last_idx: int,
        file_path: str,
        prefix: str,
        max_chunk_size: int,
        source_code: str,
        block_start: int = -1,
    ) -> list[_RawChunk]:
        """Line-aware fallback slicing for indivisible oversized nodes."""
        code_budget: int = max(max_chunk_size, 1)
        chunks: list[_RawChunk] = []
        pos: int = first_idx
        segment: str = source_code[first_idx:last_idx]
        
        lines = segment.splitlines(keepends=True)
        if len(lines) > 1:
            line_pos = first_idx
            current_start = line_pos
            current_len = 0
            
            for line in lines:
                line_len = len(line)
                if current_len + line_len > code_budget and current_len > 0:
                    chunks.append(_RawChunk(file_path, current_start, line_pos, prefix, block_start))
                    current_start = line_pos
                    current_len = 0
                
                if line_len > code_budget:
                    sub_pos = line_pos
                    line_end = line_pos + line_len
                    while sub_pos < line_end:
                        sub_end = min(sub_pos + code_budget, line_end)
                        chunks.append(_RawChunk(file_path, sub_pos, sub_end, prefix, block_start))
                        sub_pos = sub_end
                    current_start = line_end
                    current_len = 0
                else:
                    current_len += line_len
                
                line_pos += line_len
                
            if current_start < last_idx:
                chunks.append(_RawChunk(file_path, current_start, last_idx, prefix, block_start))
            return chunks

        while pos < last_idx:
            end: int = min(pos + code_budget, last_idx)
            chunks.append(_RawChunk(file_path, pos, end, prefix, block_start))
            pos = end
        return chunks

    def _split_branching(
        self,
        node: ast.If | ast.For | ast.AsyncFor | ast.While,
        source_code: str,
        file_path: str,
        first_idx: int,
        parent_prefix: str,
        max_chunk_size: int,
        line_index: tuple[list[int], list[bytes]],
        parent_block_start: int = -1,
    ) -> list[_RawChunk]:
        """Split branching nodes while counting their own headers as content."""
        node_start, _ = self._node_own_span(node, line_index)
        header_end = self._compound_header_end(node, source_code, line_index)
        header_chunks = self._header_chunk(
            max(first_idx, node_start), header_end, file_path, parent_prefix,
            max_chunk_size, source_code, parent_block_start,
        )
        own_header = self._get_header_line(node, source_code, line_index)
        body_prefix = self._join_prefix(parent_prefix, own_header)
        chunks = self._process_children(
            list(node.body), source_code, file_path, header_end, body_prefix,
            max_chunk_size, line_index, node_start,
        )

        if node.orelse:
            body_end = self._node_full_span(node.body[-1], line_index)[1] if node.body else header_end
            orelse_start = self._node_full_span(node.orelse[0], line_index)[0]
            is_elif = (
                len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
                and source_code[orelse_start:orelse_start + 4] == "elif"
            )
            if is_elif:
                chunks.extend(self.split_big(
                    node.orelse[0], source_code, file_path, orelse_start,
                    parent_prefix, max_chunk_size, line_index, parent_block_start,
                ))
            else:
                header_text, header_off = self._get_block_header(
                    source_code, body_end, orelse_start, "else"
                )
                header_end_else = orelse_start
                header_chunks.extend(self._header_chunk(
                    header_off, header_end_else, file_path, parent_prefix,
                    max_chunk_size, source_code, parent_block_start,
                ))
                orelse_prefix = self._join_prefix(parent_prefix, header_text)
                chunks.extend(self._process_children(
                    list(node.orelse), source_code, file_path, orelse_start,
                    orelse_prefix, max_chunk_size, line_index, header_off,
                ))
        return header_chunks + chunks

    def _split_try(
        self, node: ast.Try, source_code: str, file_path: str, first_idx: int,
        parent_prefix: str, max_chunk_size: int,
        line_index: tuple[list[int], list[bytes]], parent_block_start: int = -1,
    ) -> list[_RawChunk]:
        """Split try statements and count try/except/else/finally headers as content."""
        chunks: list[_RawChunk] = []
        node_start, _ = self._node_own_span(node, line_index)
        first_body_start = (self._node_full_span(node.body[0], line_index)[0]
                            if node.body else self._compound_header_end(node, source_code, line_index))
        header_chunks = self._header_chunk(
            max(first_idx, node_start), first_body_start, file_path, parent_prefix,
            max_chunk_size, source_code, parent_block_start,
        )
        try_prefix = self._join_prefix(parent_prefix, "try:")
        chunks.extend(self._process_children(
            list(node.body), source_code, file_path, first_body_start, try_prefix,
            max_chunk_size, line_index, node_start,
        ))
        cursor = chunks[-1].last_idx if chunks else first_body_start

        for handler in node.handlers:
            h_start = self._node_full_span(handler, line_index)[0]
            h_body_start = (self._node_full_span(handler.body[0], line_index)[0]
                            if handler.body else self._node_own_span(handler, line_index)[1])
            handler_header_chunks = self._header_chunk(
                max(cursor, h_start), h_body_start, file_path, parent_prefix,
                max_chunk_size, source_code, h_start,
            )
            header_chunks.extend(handler_header_chunks)
            handler_header = self._get_header_line(handler, source_code, line_index)
            chunks.extend(self._process_children(
                list(handler.body), source_code, file_path, h_body_start,
                self._join_prefix(parent_prefix, handler_header), max_chunk_size,
                line_index, h_start,
            ))
            if chunks:
                cursor = chunks[-1].last_idx

        for body, keyword in ((node.orelse, "else"), (node.finalbody, "finally")):
            if not body:
                continue
            branch_start = self._node_full_span(body[0], line_index)[0]
            text, off = self._get_block_header(source_code, cursor, branch_start, keyword)
            header_chunks.extend(self._header_chunk(
                off, branch_start, file_path, parent_prefix, max_chunk_size,
                source_code, off,
            ))
            chunks.extend(self._process_children(
                list(body), source_code, file_path, branch_start,
                self._join_prefix(parent_prefix, text), max_chunk_size, line_index, off,
            ))
            if chunks:
                cursor = chunks[-1].last_idx
        return header_chunks + chunks

    def split_big(
        self,
        node: ast.AST,
        source_code: str,
        file_path: str,
        cursor: int,
        parent_prefix: str,
        max_chunk_size: int,
        line_index: tuple[list[int], list[bytes]],
        parent_block_start: int = -1,
    ) -> list[_RawChunk]:
        first_idx, last_idx = self._node_full_span(node, line_index)
        if first_idx < cursor:
            first_idx = cursor
        if last_idx <= first_idx:
            return []

        # If the entire node fits in budget, emit it directly with its span
        if last_idx - first_idx <= max_chunk_size:
            return [_RawChunk(file_path, first_idx, last_idx, parent_prefix, parent_block_start)]

        if isinstance(node, ast.ClassDef):
            header: str = self._compose_class_header(node, source_code, line_index)
            header_start: int = self._node_own_span(node, line_index)[0]
            header_end: int = self._compound_header_end(node, source_code, line_index)
            header_chunks = self._header_chunk(
                first_idx, header_end, file_path, parent_prefix,
                max_chunk_size, source_code, parent_block_start,
            )
            current_prefix: str = self._join_prefix(parent_prefix, header)
            result: list[_RawChunk] = self._process_children(
                self._get_body_children(node), source_code, file_path,
                header_end, current_prefix, max_chunk_size, line_index, header_start,
            )
            if result or header_chunks:
                return header_chunks + result
            return self._split_lines(
                first_idx, last_idx, file_path, parent_prefix, max_chunk_size, source_code, parent_block_start
            )

        if isinstance(node, self._FUNC_TYPES):
            header = self._get_function_signature(node, source_code, line_index)
            header_start = self._node_own_span(node, line_index)[0]
            header_end = self._compound_header_end(node, source_code, line_index)
            header_chunks = self._header_chunk(
                first_idx, header_end, file_path, parent_prefix,
                max_chunk_size, source_code, parent_block_start,
            )
            current_prefix = self._join_prefix(parent_prefix, header)
            result = self._process_children(
                self._get_body_children(node), source_code, file_path,
                header_end, current_prefix, max_chunk_size, line_index, header_start,
            )
            if result or header_chunks:
                return header_chunks + result
            return self._split_lines(
                first_idx, last_idx, file_path, parent_prefix, max_chunk_size, source_code, parent_block_start
            )

        if isinstance(node, self._TRY_TYPES):
            result = self._split_try(
                node, source_code, file_path, first_idx, parent_prefix,
                max_chunk_size, line_index, parent_block_start,
            )
            if result:
                return result

        elif isinstance(node, self._LOOP_TYPES):
            result = self._split_branching(
                node, source_code, file_path, first_idx, parent_prefix,
                max_chunk_size, line_index, parent_block_start,
            )
            if result:
                return result

        elif isinstance(node, self._WITH_TYPES):
            header = self._get_header_line(node, source_code, line_index)
            header_start = self._node_own_span(node, line_index)[0]
            header_end = self._compound_header_end(node, source_code, line_index)
            header_chunks = self._header_chunk(
                max(first_idx, header_start), header_end, file_path, parent_prefix,
                max_chunk_size, source_code, parent_block_start,
            )
            current_prefix = self._join_prefix(parent_prefix, header)
            result = self._process_children(
                list(node.body), source_code, file_path,
                header_end, current_prefix, max_chunk_size, line_index, header_start,
            )
            if result or header_chunks:
                return header_chunks + result

        return self._split_lines(
            first_idx, last_idx, file_path, parent_prefix, max_chunk_size, source_code, parent_block_start
        )

    # ------------------------------------------------------------------ #
    # Merging and final text assembly
    # ------------------------------------------------------------------ #

    def _finalize_chunk(self, rc: _RawChunk, source_code: str) -> Chunk:
        raw_slice: str = source_code[rc.first_idx : rc.last_idx].strip()

        prefix_lines: list[str] = self._strip_redundant_header(
            self._prefix_lines(rc.prefix), raw_slice
        )
        effective_prefix: str = self._sanitize_prefix("\n".join(prefix_lines)).strip()

        text: str = f"{effective_prefix}\n{raw_slice}" if effective_prefix else raw_slice

        return Chunk(
            source=MinimalSource(
                file_path=rc.file_path,
                first_character_index=rc.first_idx,
                last_character_index=rc.last_idx,
            ),
            text=text,
        )

    def _can_merge(
        self, acc: _RawChunk, rc: _RawChunk, max_chunk_size: int, floor: int
    ) -> tuple[list[str], int] | None:
        """Decide whether two chunks may merge."""
        if acc.file_path != rc.file_path:
            return None
        if rc.first_idx < acc.last_idx:
            return None

        common: list[str] = self._common_prefix_lines(acc.prefix, rc.prefix)
        acc_lines: list[str] = self._prefix_lines(acc.prefix)

        if not common and not (not acc.prefix.strip() and not rc.prefix.strip()):
            return None

        start: int = acc.first_idx
        if len(common) < len(acc_lines) and 0 <= acc.block_start < acc.first_idx:
            start = max(acc.block_start, floor)

        if rc.last_idx - start > max_chunk_size:
            return None

        return common, start

    def merge_smalls(
        self, raw_chunks: list[_RawChunk], source_code: str, max_chunk_size: int
    ) -> list[Chunk]:
        """Greedily merge contiguous chunks sharing a context ancestor."""
        if not raw_chunks:
            return []

        ordered: list[_RawChunk] = sorted(raw_chunks, key=lambda c: (c.file_path, c.first_idx))
        merged: list[Chunk] = []
        acc: _RawChunk = ordered[0]
        floor: int = 0

        for rc in ordered[1:]:
            outcome: tuple[list[str], int] | None = self._can_merge(
                acc, rc, max_chunk_size, floor
            )
            if outcome is not None:
                common, start = outcome
                widened: bool = start < acc.first_idx
                acc = _RawChunk(
                    file_path=acc.file_path,
                    first_idx=start,
                    last_idx=max(acc.last_idx, rc.last_idx),
                    prefix="\n".join(common),
                    block_start=-1 if widened else acc.block_start,
                )
            else:
                merged.append(self._finalize_chunk(acc, source_code))
                floor = acc.last_idx
                acc = rc

        merged.append(self._finalize_chunk(acc, source_code))
        return merged

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def index(self, max_chunk_size: int = 2000) -> list[Chunk]:
        max_chunk_size = max(int(max_chunk_size), 1)
        all_chunks: list[Chunk] = []

        for py_file in self.py_files:
            file_path: Path = Path(py_file)
            if not file_path.exists():
                continue

            try:
                source_code: str = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            try:
                tree: ast.Module = ast.parse(source_code)
            except (SyntaxError, ValueError, RecursionError):
                continue

            line_index: tuple[list[int], list[bytes]] = self._build_line_index(source_code)
            cursor: int = 0
            raw_chunks: list[_RawChunk] = []

            for node in tree.body:
                sub: list[_RawChunk] = self.split_big(
                    node=node,
                    source_code=source_code,
                    file_path=str(py_file),
                    cursor=cursor,
                    parent_prefix="",
                    max_chunk_size=max_chunk_size,
                    line_index=line_index,
                )
                if sub:
                    raw_chunks.extend(sub)
                    cursor = sub[-1].last_idx

            all_chunks.extend(self.merge_smalls(raw_chunks, source_code, max_chunk_size))

        return all_chunks