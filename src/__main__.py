try:
    from .ingestion import Ingester, Markdown, Indexer
    from .pyd_models import MinimalSource
    import fire
except Exception as e:
    print("Module level:", e)
    exit(1)

def index(max_chunk_size = 2000):
    ingester = Ingester("data/raw/vllm-0.10.1")
    test_file = "t.md"

    md_parser = Markdown(md_files=[test_file])
    chunks = md_parser.split_by_headers()
    chunks = md_parser.split_paragraphs(chunks)
    for chunk in chunks:
        chunk["file_path"] = test_file
    chunks_text = ingester.saving_chunks(chunks)

    indexer = Indexer(chunks_text)
    indexer.indexing_chunks()

fire.Fire({"index": index})