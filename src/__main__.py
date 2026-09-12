try:
    from .ingestion import Ingester, Indexer
    from .pyd_models import MinimalSource
    import fire
except Exception as e:
    print("Module level:", e)
    exit(1)

ingester = Ingester("data/raw/vllm-0.10.1")
py_files = ingester.find_python_files()
py_files = ["data/raw/vllm-0.10.1/setup.py",]
fire.Fire(Indexer(py_files))