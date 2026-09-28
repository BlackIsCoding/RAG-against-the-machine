from __future__ import annotations
try:
    from .ingest import Ingester
    from .index import Indexer
    from .markdown import Header
    # from .python_ch import PythonStructuralChunker
    import json
    import fire
except Exception as e:
    print("Module level:", e)
    exit(1)


def index(max_chunk_size = 2000, query=''):
    ingester = Ingester("data/raw/vllm-0.10.1")
    md = ingester.find_txt_files()
    md += ingester.find_md_files()
    md = ["data/raw/vllm-0.10.1/t.md"]
    # # md = ["t.md"]
    md_parser = Header(md, max_chunk_size)
    chunks = md_parser.split()
    chunks = md_parser.split_para()
    chunks = md_parser.split_lines()
    chunks = md_parser.split_sentence()
    objected_chunks = ingester.saving_chunks(chunks)

    # py_files = ingester.find_python_files()
    # for file_path in py_files:
    #     try:
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             code_content = f.read()
            
    #         chunker = PythonStructuralChunker(file_path, code_content, max_chunk_size=max_chunk_size)
    #         py_chunks = chunker.chunk_file()
            
    #         # Extend directly since keys match the Markdown dictionaries perfectly!
    #         chunks.extend(py_chunks)
    #     except Exception as e:
    #         print(f"Error parsing python file {file_path}: {e}")
    # chunks_text = ingester.saving_chunks(chunks)

    indexer = Indexer(objected_chunks)
    indexer.indexing_chunks()
    if not query:
        with open("data/datasets/public/AnsweredQuestions/dataset_docs_public.json") as f:
            questions = json.load(f)
        indexer.evaluate_all(questions['rag_questions'], 5)
    else:
        indexer.search(query, 5)

fire.Fire({"index": index})

