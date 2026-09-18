try:
    from .ingestion import Ingester, Indexer
    from .pyd_models import MinimalSource
    import fire
except Exception as e:
    print("Module level:", e)
    exit(1)

# ingester = Ingester("data/raw/vllm-0.10.1")
# py_files = ingester.find_python_files()

py_files = ["test_chunks.py"]


py_files = ["test_chunks.py"]


def run_test(max_chunk_size: int = 120):
    max_chunk_size = int(max_chunk_size)
    indexer = Indexer(py_files=py_files)
    chunks = indexer.index(max_chunk_size=max_chunk_size)

    print(f"\nTotal Chunks: {len(chunks)}")
    print("=" * 60)

    for idx, chunk in enumerate(chunks, 1):
        raw_chars = chunk.source.last_character_index - chunk.source.first_character_index
        total_chars = len(chunk.text)
        
        print(f"--- Chunk #{idx} ---")
        print(f"File: {chunk.source.file_path} | Offsets: [{chunk.source.first_character_index}:{chunk.source.last_character_index}]")
        print(f"Raw Body Chars: {raw_chars} | Total Rendered Chars: {total_chars}")
        print("Text:")
        print(chunk.text)
        print("-" * 60)


if __name__ == "__main__":
    fire.Fire({"run_test": run_test})