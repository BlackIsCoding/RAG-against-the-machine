from __future__ import annotations
try:
    from .ingest import Ingester
    from .index import Indexer
    from .markdown import Header
    from .chunking_pyy import Python
    import json
    import fire
    import time
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

    start = time.perf_counter()
    py_parser.split_toplvl()
    print("python toplvl", time.perf_counter() - start)
    start = time.perf_counter()
    py_parser.merge_toplvl()
    print("merge toplvl", time.perf_counter() - start)
    start = time.perf_counter()
    py_parser.split_internals()
    print("split internals", time.perf_counter() - start)
    start = time.perf_counter()
    py_parser.split_lines()
    print("split lines", time.perf_counter() - start)
    start = time.perf_counter()
    py_parser.split_sentence()
    print("python split sentence", time.perf_counter() - start)
    start = time.perf_counter()
    chunks += py_parser.chunks_adapt()
    print("python adapting", time.perf_counter() - start)
            
    objected_chunks = ingester.saving_chunks(chunks)

    indexer = Indexer(objected_chunks)
    indexer.bm25s_indexing()
    
    # start = time.perf_counter()
    # indexer.encode_chunks()
    # print("encoding chunks:", time.perf_counter() - start)

    # if not query:
    #     with open("data/datasets/public/AnsweredQuestions/dataset_docs_public.json") as f:
    #         questions = json.load(f)
    #     indexer.evaluate_dataset(questions['rag_questions'], 5)
    #     with open("data/datasets/public/AnsweredQuestions/dataset_code_public.json") as f:
    #         questions = json.load(f)
    #     indexer.evaluate_dataset(questions['rag_questions'], 5)
    # else:
    #     indexer.fusion(query, 5)

fire.Fire({"index": index})