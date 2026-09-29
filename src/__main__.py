from __future__ import annotations
try:
    from .ingest import Ingester
    from .index import Indexer
    from .markdown import Header
    from .python_ch_paragraph import PythonStructuralChunker
    import json
    import fire
except Exception as e:
    print("Module level:", e)
    exit(1)


def index(max_chunk_size = 2000, query=''):
    ingester = Ingester("data/raw/vllm-0.10.1")
    md = ingester.find_txt_files()
    md += ingester.find_md_files()
    
    md_parser = Header(md, max_chunk_size)
    chunks = md_parser.split()
    chunks = md_parser.split_para()
    chunks = md_parser.split_lines()
    chunks = md_parser.split_sentence()

    py_files = ingester.find_python_files()
    for file_path in py_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()
            
            chunker = PythonStructuralChunker(file_path, code_content, max_chunk_size=max_chunk_size)
            py_chunks = chunker.chunk_file()
            
            # Convert Python dictionary metadata into a list of strings to match Markdown chunk expectations
            for chunk in py_chunks:
                meta = chunk.get("metadata", {})
                if isinstance(meta, dict):
                    chunk["metadata"] = [f"{k} -> {v}" for k, v in meta.items()]
            
            chunks.extend(py_chunks)
        except Exception as e:
            print(f"Error parsing python file {file_path}: {e}")
            
    objected_chunks = ingester.saving_chunks(chunks)

    indexer = Indexer(objected_chunks)
    indexer.indexing_chunks()
    if not query:
        with open("data/datasets/public/AnsweredQuestions/dataset_docs_public.json") as f:
            questions = json.load(f)
        indexer.evaluate_all(questions['rag_questions'], 5)
        with open("data/datasets/public/AnsweredQuestions/dataset_code_public.json") as f:
            questions = json.load(f)
        indexer.evaluate_all(questions['rag_questions'], 5)
    else:
        indexer.search(query, 5)

fire.Fire({"index": index})