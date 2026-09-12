from pathlib import Path


def find_python_files(repository_path: str) -> list[Path]:
    """Find all Python files recursively in a repository."""
    repository = Path(repository_path)
    print(type(repository.rglob("*.py")))

    return list(repository.rglob("*.py"))

files = find_python_files("../data/raw/vllm-0.10.1")

# for file in files:
#     print(file)
# print(len(files))