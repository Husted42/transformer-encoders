"""Evaluate one retrieval method against a shared benchmark corpus."""

import argparse
import json

from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.evaluation.metrics import retrieval_metrics
from src.retrievers.bm25 import BM25
from src.retrievers.checkpoint import load_checkpoint
from src.retrievers.tf_idf import TFIDF
from src.retrievers.utils import retrieveQueryAndGetRelevancies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("tfidf", "bm25", "dpr"), required=True)
    parser.add_argument("--dataset", choices=("fiqa", "squad"), default="fiqa")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--base-path", default=".")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    if args.method == "dpr" and not args.checkpoint:
        parser.error("--checkpoint is required for DPR")
    corpus = Corpus(args.dataset, f"{args.base_path}/retrieval_data")
    rows = {"train": corpus.train, "validation": corpus.val, "test": corpus.test}[args.split]
    if rows is None or not len(rows):
        parser.error(f"{args.dataset} does not have a nonempty {args.split} split")
    queries = [Query(row["question"], row["query_id"], row["document_id"]) for row in rows]
    if args.method == "dpr":
        retriever, metadata, _ = load_checkpoint(args.checkpoint, args.device)
        if metadata.get("corpus_version") != corpus.version:
            raise ValueError("Checkpoint corpus version does not match the evaluation corpus.")
        retriever.build_index(corpus.all_documents)
    else:
        retriever = (TFIDF if args.method == "tfidf" else BM25)(corpus.all_documents)
    relevancies = retrieveQueryAndGetRelevancies(retriever, queries, args.k)
    print(json.dumps({"dataset": corpus.version, "split": args.split, "queries": len(queries),
                      "method": args.method, **retrieval_metrics(relevancies, queries, args.k)}, indent=2))


if __name__ == "__main__":
    main()
