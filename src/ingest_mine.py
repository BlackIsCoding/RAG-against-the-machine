from pathlib import Path

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