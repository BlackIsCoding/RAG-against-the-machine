# AST-Aware Python RAG Ingestion Pipeline Strategy

## 1. Pipeline Architecture Overview

The ingestion strategy combines **AST-driven structural decomposition** with a **line-aligned fallback slicer** to produce syntax-aware, variable-sized chunks.

Every chunk, regardless of its depth in the AST, maintains its full structural hierarchy:

- `class` declaration
- `@decorator` declarations
- `def` signature, including parameters and type hints

```text
                              [ Source Python File ]
                                        │
                                  ast.parse()
                                        │
                                [ Top-Level AST Node ]
                                        │
                                   split_big()
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
   [ Fits <= max_chunk_size ]   [ Oversized ClassDef ]   [ Oversized Function / Leaf ]
             │                          │                          │
      Return Single Chunk        Decompose .body          Extract Decorators & Signature
             │                   Pass 'class Name:'             Strip Header from Body
             │                       Prefix                        │
             │                          │                   _split_lines Fallback
             │                          └────────┐                 │
             ▼                                   ▼                 ▼
      [ Output Chunk ]                     Recursive        Line-Aligned Windowing
                                           split_big()      + Character Hard-Cap Failsafe
                                                 │                 │
                                                 └────────┬────────┘
                                                          ▼
                                                   [ Output Chunks ]
                                                          │
                                                Phase 2: merge_smalls
```

---

## 2. Pipeline Flow

### Step 1 — Parse the Python File

The source file is parsed using Python's `ast` module:

```python
tree = ast.parse(source)
```

This provides a structured representation of the Python source code and allows the ingestion pipeline to distinguish between classes, functions, and other syntax nodes.

### Step 2 — Process Top-Level AST Nodes

Each relevant top-level AST node is passed to:

```python
split_big()
```

The function determines whether the node fits within `max_chunk_size`.

### Step 3 — Keep Small Nodes Intact

If the node already fits within the maximum chunk size, it is returned as a single chunk.

```text
AST Node
   │
   ├── size <= max_chunk_size
   │
   └──> Single Chunk
```

This avoids unnecessarily splitting small, logically coherent pieces of code.

### Step 4 — Recursively Decompose Oversized Classes

If an oversized node is a `ClassDef`, its `.body` is recursively processed.

The class name is preserved as a structural prefix:

```text
class ComplexClass:
```

The recursive calls then process the class's methods and other child nodes.

### Step 5 — Handle Oversized Functions and Leaf Nodes

For oversized functions or nodes that cannot be meaningfully decomposed further, the pipeline extracts:

1. Decorators
2. The exact function signature
3. The remaining function body

The extracted information becomes the `effective_prefix`.

Example:

```python
@staticmethod
def decorated_method(x: int, y: int) -> int:
```

The prefix preserves both the decorator and the complete signature.

### Step 6 — Fall Back to Line-Based Splitting

If the function body is still too large to fit into one chunk, `_split_lines` is used.

The body is divided along natural `\n` boundaries rather than arbitrarily cutting through lines.

A character-level hard cap is also used as a final failsafe:

```python
chunk[:max_chunk_size]
```

This guarantees that chunks do not exceed the configured maximum size.

### Step 7 — Preserve Structural Context

Every generated sub-chunk receives the appropriate structural prefix.

For example:

```python
class ComplexClass:
    @staticmethod
    def decorated_method(x: int, y: int) -> int:
        # Long body...
```

Can produce:

```text
Chunk 1:
class ComplexClass:
@staticmethod
def decorated_method(x: int, y: int) -> int:
    # Long body block statement 1...

Chunk 2:
class ComplexClass:
@staticmethod
def decorated_method(x: int, y: int) -> int:
    # Long body block statement 2...
```

This means that even when a function is split across multiple chunks, each chunk retains enough context to understand where the code belongs.

---

# 3. Documented Problems and Resolved Solutions

| # | Problem | Root Cause | Implemented Solution |
|---|---|---|---|
| **1** | **Dummy Parameter Signatures** (`def method(...):`) | Function signatures were hardcoded using generic placeholder strings instead of exact AST metadata. | Implemented `_get_function_signature` using AST line-boundary inspection and `ast.get_source_segment` to retrieve exact parameter lists and type hints. |
| **2** | **Dropped Decorators** (`@staticmethod`, `@cache`) | AST stores `decorator_list` separately from the function body and signature. | Extracted decorator AST segments through `node.decorator_list` and prepended them to the function signature prefix. |
| **3** | **Unbounded Chunk Sizes** | Leaf nodes, multiline dictionaries, or monolithic functions could exceed `max_chunk_size` when AST body splitting was insufficient. | Implemented `_split_lines` fallback windowing to split oversized text along natural line boundaries, with a character-level hard-cap failsafe. |
| **4** | **Missing Decorators in Sliced Chunks** | When an oversized `FunctionDef` went directly into `_split_lines`, `parent_prefix` only contained the enclosing `ClassDef`. | Updated `split_big` to construct an `effective_prefix` containing the function signature and decorators before calling `_split_lines`. |
| **5** | **Header Duplication in Sub-Chunks** | Function source segments included the `def func():` header at the beginning of the body. This caused the header to appear twice when `effective_prefix` was prepended. | Added header-stripping logic to remove the leading `def` and decorator lines before passing the remaining function body to `_split_lines`. |

---

# 4. Structural Hierarchy Preservation

The central principle of the ingestion pipeline is:

> **Splitting a code block must not remove the structural context required to understand that code.**

For a normal Python function:

```python
def example(x: int) -> str:
    ...
```

the generated chunks should retain the exact signature rather than replacing it with a generic placeholder.

For a class method:

```python
class ComplexClass:
    @staticmethod
    def decorated_method(x: int, y: int) -> int:
        ...
```

the hierarchy is:

```text
class ComplexClass:
    └── @staticmethod
        └── def decorated_method(x: int, y: int) -> int:
```

Therefore, when the method is split, every resulting chunk retains:

```text
class ComplexClass:
@staticmethod
def decorated_method(x: int, y: int) -> int:
```

followed by its respective body section.

---

# 5. Example

## Original Source

```python
class ComplexClass:
    @staticmethod
    def decorated_method(x: int, y: int) -> int:
        # Long body block statement 1...
        # Long body block statement 2...
```

## Generated Chunk #1

```python
class ComplexClass:
@staticmethod
def decorated_method(x: int, y: int) -> int:
    # Long body block statement 1...
```

## Generated Chunk #2

```python
class ComplexClass:
@staticmethod
def decorated_method(x: int, y: int) -> int:
    # Long body block statement 2...
```

Both chunks contain the complete structural hierarchy and the exact function signature.

---

# 6. Key Implementation Components

### `split_big()`

Responsible for deciding how an AST node should be split.

Its main responsibilities are:

- Check whether the node fits within `max_chunk_size`.
- Recursively decompose oversized classes.
- Build structural prefixes.
- Handle oversized functions and leaf nodes.
- Trigger `_split_lines` when AST decomposition is no longer sufficient.

### `_get_function_signature()`

Responsible for extracting the exact function signature from the original source.

It preserves:

- Function name
- Parameter names
- Parameter types
- Default values
- Return type annotations

Example:

```python
def calculate(value: int, multiplier: float = 1.0) -> float:
```

The exact signature is retained instead of generating:

```python
def calculate(...):
```

### `_split_lines()`

Responsible for splitting oversized text into manageable chunks.

Its strategy is:

1. Prefer natural line boundaries.
2. Keep chunks within `max_chunk_size`.
3. Use a character-level hard cap when necessary.

---

# 7. Design Goals

The ingestion pipeline is designed around four main goals:

### 1. Structural Awareness

Use the Python AST to understand the structure of the code rather than splitting the file blindly.

### 2. Context Preservation

Keep class names, decorators, and exact function signatures attached to their corresponding code.

### 3. Bounded Chunk Size

Guarantee that generated chunks do not exceed `max_chunk_size`.

### 4. Graceful Fallback

When AST-based decomposition cannot split a node any further, fall back to line-based slicing instead of returning an oversized chunk.

---

# 8. Current Pipeline Status

## Phase 1 — Complete and Verified

The following components are implemented and verified:

- [x] Variable-sized AST decomposition
- [x] Recursive splitting of oversized classes
- [x] Decorator preservation
- [x] Exact function parameter/signature extraction
- [x] Structural context prefixing
- [x] Line-boundary slicing
- [x] Character-level hard size cap
- [x] Function header stripping
- [x] Prevention of duplicated headers
- [x] Fallback handling for oversized leaf nodes

## Phase 2 — Next Step

### `merge_smalls`

The next stage is to implement `merge_smalls`.

Its purpose is to **coalesce adjacent undersized chunks** where appropriate.

The goal is to improve vector-search context density while avoiding unnecessary fragmentation.

```text
Phase 1
Source → AST decomposition → Split oversized nodes → Output chunks
                                                       │
                                                       ▼
Phase 2                                      merge_smalls()
                                                       │
                                                       ▼
                                             Final chunks
```

---

# 9. Final Pipeline

The complete ingestion flow is therefore:

```text
Python Source File
        │
        ▼
    ast.parse()
        │
        ▼
 Top-Level AST Nodes
        │
        ▼
    split_big()
        │
        ├── Small node
        │      └──> Keep intact
        │
        ├── Oversized ClassDef
        │      └──> Recursively split .body
        │
        └── Oversized Function / Leaf
               │
               ├── Extract decorators
               ├── Extract exact signature
               ├── Build structural prefix
               ├── Strip duplicated header
               └──> _split_lines()
                          │
                          ├── Line-aligned chunks
                          └── Character hard-cap failsafe
                                     │
                                     ▼
                               Phase 1 Chunks
                                     │
                                     ▼
                              merge_smalls()
                                     │
                                     ▼
                              Final RAG Chunks
```

The resulting chunks are **syntax-aware, size-bounded, and structurally contextualized**, making them better suited for downstream embedding and vector search in the RAG pipeline.
