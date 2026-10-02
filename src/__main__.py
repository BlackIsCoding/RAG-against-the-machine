from __future__ import annotations
try:
    from .ingest import Ingester
    from .index import Indexer
    from .markdown import Header
    from .chunking_pyy import Python
    import json
    import fire
except Exception as e:
    print("Module level:", e)
    exit(1)


def index(max_chunk_size = 2000, query=''):
    print(1)
    ingester = Ingester("data/raw/vllm-0.10.1")
    md = ingester.find_txt_files()
    md += ingester.find_md_files()
    
    md_parser = Header(md, max_chunk_size)
    chunks = md_parser.split()
    chunks = md_parser.split_para()
    chunks = md_parser.split_lines()
    chunks = md_parser.split_sentence()

    py_files = ingester.find_python_files()
    py_parser = Python(py_files, max_chunk_size)
    py_parser.split_toplvl()
    print("finish")
    py_parser.merge_toplvl()
    print(2)
    py_parser.split_internals()
    print(3)
    print("finish 2")
    py_parser.split_lines()
    print(4)
    py_parser.split_sentence()
    print(5)
    chunks += py_parser.chunks_adapt()
            
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