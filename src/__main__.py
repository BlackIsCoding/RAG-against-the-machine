from __future__ import annotations
try:
    from .ingest import Ingester
    from .index import Indexer
    from .markdown import Header
    from .chunking_pyy import Python
    from .pyd_models import MinimalSearchResults, MinimalSource, StudentSearchResults
    import json
    import fire
    import time
    import pathlib
except Exception as e:
    print("Module level:", e)
    exit(1)


def index(max_chunk_size = 2000):
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
    py_parser.merge_toplvl()
    py_parser.split_internals()
    py_parser.split_lines()
    py_parser.split_sentence()
    chunks += py_parser.chunks_adapt()

            
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

def search(query='', k=5):
    index = Indexer(None)

    if query:
        results, _ = index.lexical_search(query, k)
        with open("data/processed/chunks/all_chunks.json") as f:
            chunks = json.load(f)
        for r in results:
            chunk = chunks[r]
            first = chunk['first_character_index']
            last = chunk['last_character_index']
            print(f"{chunk['file_path']} [{first}:{last}]")

def search_dataset(dataset_path, k, save_directory):
    path = pathlib.Path(dataset_path)
    if not path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # 1. Load the question dataset
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data.get("rag_questions", [])
    if not questions:
        print("Error: No questions found in the dataset.")
        return

    saving_path = pathlib.Path("data/processed")
    if not saving_path.exists():
        print("Error: Index not found. Please run 'index' first.")
        return

    with open("data/processed/chunks/all_chunks.json") as f:
        chunks = json.load(f)

    index = Indexer(None)
    min_search_result_list = []
    for question in questions:
        q_id = question['question_id']
        q_text = question['question']


        results, _ = index.lexical_search(q_text, k)

        minimal_list = []

        for r in results:
            chunk = chunks[r]
            
            source = MinimalSource(
                file_path = str(chunk['file_path']),
                first_character_index = chunk['first_character_index'],
                last_character_index = chunk['last_character_index'])
            
            minimal_list.append(source)
        
        
        min_search_result = MinimalSearchResults(
                question_id=q_id,
                question=q_text,
                retrieved_sources=minimal_list
            )
        min_search_result_list.append(min_search_result)

    student_results = StudentSearchResults(
        search_results=min_search_result_list,
        k=k
    )

    save_dir = pathlib.Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = save_dir / path.name
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(student_results.model_dump(), f, indent=4)

    print(f"Saved student_search_results to {output_file}")



def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Report recall@k using IoU overlap with ground-truth sources."""
    student_path = pathlib.Path(student_search_results_path)
    gt_path = pathlib.Path(dataset_path)

    if not student_path.exists() or not gt_path.exists():
        print("Error: Path does not exist.")
        return

    with open(student_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    student_results_dict = {
        item["question_id"]: item
        for item in student_data.get("search_results", [])
    }
    gt_questions = gt_data.get("rag_questions", [])

    k_values = [1, 3, 5, 10]
    recall_counts = {k: 0 for k in k_values}
    total_evaluated = 0

    for gt_q in gt_questions:
        q_id = gt_q.get("question_id")
        gt_sources = gt_q.get("sources", [])

        if not gt_sources or q_id not in student_results_dict:
            continue

        total_evaluated += 1
        student_sources = student_results_dict[q_id].get(
            "retrieved_sources", []
        )

        for k in k_values:
            top_k = student_sources[:k]
            found = False

            for gt_src in gt_sources:
                gt_file = gt_src["file_path"]
                gt_start = gt_src["first_character_index"]
                gt_end = gt_src["last_character_index"]

                for st_src in top_k:
                    if gt_file != st_src["file_path"]:
                        continue

                    st_start = st_src["first_character_index"]
                    st_end = st_src["last_character_index"]

                    # Intersection
                    inter_start = max(gt_start, st_start)
                    inter_end = min(gt_end, st_end)
                    intersection = max(0, inter_end - inter_start)

                    # Union = GT length + chunk length - intersection
                    gt_len = max(0, gt_end - gt_start)
                    st_len = max(0, st_end - st_start)
                    union = gt_len + st_len - intersection

                    iou = intersection / union if union > 0 else 0.0

                    # One qualifying chunk is enough to pass this recall@k
                    if iou >= 0.05:
                        found = True
                        break

                if found:
                    break

            if found:
                recall_counts[k] += 1

    print(f"Total evaluated questions: {total_evaluated}")
    print("Evaluation Results (IoU >= 0.05):")

    for k in k_values:
        score = (
            recall_counts[k] / total_evaluated
            if total_evaluated > 0
            else 0.0
        )
        print(
            f"Recall@{k}: {score:.3f} ({score * 100:.1f}%)"
        )



fire.Fire({"index": index, "search": search,
           "search_dataset": search_dataset,
           "evaluate": evaluate})