from pathlib import Path
from .pyd_models import Chunk, MinimalSource
import json

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