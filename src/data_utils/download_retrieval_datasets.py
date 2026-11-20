from datasets import load_dataset

def get_fiqa_dataset(base_dir: str):
    revision = "979c07a7cb5ccc6ca009792241fa1250b98055dd"
    corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus", revision=revision, cache_dir=base_dir)
    queries = load_dataset("BeIR/fiqa", "queries", split="queries", revision=revision, cache_dir=base_dir)
    qrels = load_dataset("BeIR/fiqa-qrels", revision="252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
                         cache_dir=base_dir)
    return corpus, queries, qrels

def get_squad_dataset(base_dir: str):
    ds = load_dataset("rajpurkar/squad", cache_dir=base_dir)
    return ds