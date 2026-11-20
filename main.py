import hydra

from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.evaluation.metrics import retrieval_metrics
from src.retrievers.bm25 import BM25
from src.retrievers.tf_idf import TFIDF
from src.retrievers.utils import retrieveQueryAndGetRelevancies


@hydra.main(config_path="config", config_name="fiqa_experiment.yaml", version_base=None)
def main(cfg):
    methods = {"tf_idf": TFIDF, "bm25": BM25}
    if cfg.retriever.method not in methods:
        raise ValueError("main.py supports lexical baselines; use evaluate.py for DPR checkpoints.")
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f"{cfg.base_path}/retrieval_data")
    queries = [Query(row["question"], row["query_id"], row["document_id"]) for row in corpus.val]
    model = methods[cfg.retriever.method](documents=corpus.all_documents)
    relevancies = retrieveQueryAndGetRelevancies(model, queries, k=cfg.top_k)
    print(f"Validation results ({corpus.version}): {retrieval_metrics(relevancies, queries, cfg.top_k)}")


if __name__ == "__main__":
    main()
