from pathlib import Path
from .pyd_models import MinimalSource
import ast

class Ingester():
    def __init__(self, repository):
        self.repository = repository

    def find_python_files(self) -> list[Path]:
        """Find all Python files recursively in a repository."""
        repository = Path(self.repository)
        return list(repository.rglob("*.py"))

    def find_docs_files(self) -> list[Path]:
        """Find all documents files recursively in a repository."""
        pass

class Indexer():
    def __init__(self, py_files):
        self.py_files = py_files

    def merge_small():
        pass

    def split_big():
        pass

    def index(self, max_chunk_size=2000):
        for py_file in self.py_files:

            file_path = Path(py_file)
            source_code = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source_code)

            chunks_objects = []
            cursor = 0
            for node in tree.body:

                chunk_text = ast.get_source_segment(source_code, node)
                first_char_index = source_code.find(chunk_text, cursor)
                last_char_index = first_char_index + len(chunk_text)
                cursor = last_char_index

                chunks_objects.append(
                    MinimalSource(
                    file_path=str(py_file),
                    first_character_index=first_char_index,
                    last_character_index = last_char_index))
        return chunks_objects
