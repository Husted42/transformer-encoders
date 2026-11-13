"""NOTE : Debugging was build with AI """
"""Compare pooling and learning rate on fixed, small FIQA train/validation samples."""

import random
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from huggingface_hub import try_to_load_from_cache
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_utils.query import Query
from src.evaluation.metrics import Evaluate
from src.retrievers.dpr import DPR
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.training.trainer import Trainer


TRAIN_QUERIES = 128
VALIDATION_QUERIES = 64
EPOCHS = 8
SEED = 42


def sample_rows(split, count):
    indices = list(range(len(split)))
    random.Random(SEED).shuffle(indices)
    rows = []
    for index in indices:
        row = split[index]
        if len(row["text"]) != len(row["document_id"]):
            raise ValueError(f"Unaligned passages and IDs for query {row['query_id']}.")
        if any(not isinstance(text, str) for text in row["text"]):
            raise ValueError(f"Non-string passage for query {row['query_id']}.")
        pairs = [(text, doc_id) for text, doc_id in zip(row["text"], row["document_id"])
                 if text.strip()]
        if pairs and isinstance(row["question"], str) and row["question"].strip():
            rows.append({
                "question": row["question"], "query_id": row["query_id"],
                "text": [text for text, _ in pairs], "document_id": [doc_id for _, doc_id in pairs],
            })
        if len(rows) == count:
            return rows
    raise ValueError(f"FIQA split does not contain {count} usable queries.")


def recall(retriever, rows, k):
    queries = [Query(row["question"], row["query_id"], row["document_id"]) for row in rows]
    return Evaluate(retrieveQueryAndGetRelevancies(retriever, queries, k), queries)["recall"]


def main():
    dataset_path = ROOT / "retrieval_data" / "fiqa"
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Cached FIQA data not found: {dataset_path}")
    tokenizer_config = try_to_load_from_cache("bert-base-uncased", "tokenizer_config.json")
    if not isinstance(tokenizer_config, str):
        raise FileNotFoundError("The bert-base-uncased tokenizer must be available in the Hugging Face cache.")

    data = load_from_disk(str(dataset_path))
    train_rows = sample_rows(data["train"], TRAIN_QUERIES)
    validation_rows = sample_rows(data["validation"], VALIDATION_QUERIES)
    documents = train_rows + validation_rows
    print(f"Candidates: {sum(len(row['text']) for row in documents)} passages from sampled train/validation rows")
    print(f"Queries: {len(train_rows)} train, {len(validation_rows)} validation; test split is not used")
    results = []
    torch.set_num_threads(1)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        for pooling in ("cls", "mean"):
            for learning_rate in (0.001, 0.0001):
                torch.manual_seed(0)
                cfg = compose(config_name="fiqa_experiment", overrides=[
                    "retriever=dpr",
                    f"model.pooling_type={pooling}",
                    "model.query_encoder_weights_path=null",
                    "model.document_encoder_weights_path=null",
                    f"training.learning_rate={learning_rate}",
                    f"training.epochs={EPOCHS}",
                ])
                cfg.device = "cpu"
                cfg.model.tokenizer_name = str(Path(tokenizer_config).parent)
                print(f"\n--- pooling={pooling}, learning_rate={learning_rate} ---")
                retriever = DPR(cfg)
                losses = Trainer(retriever, cfg.training).train(train_rows)
                retriever.build_index(documents)
                train_recall = recall(retriever, train_rows, k=10)
                validation_recall = recall(retriever, validation_rows, k=10)
                results.append((pooling, learning_rate, losses[-1], train_recall, validation_recall))

    print("\nPooling  Learning rate  Loss      Train recall@10  Validation recall@10")
    for pooling, learning_rate, loss, train_recall, validation_recall in results:
        print(f"{pooling:<8} {learning_rate:<14g} {loss:<9.4f} {train_recall:<16.4f} {validation_recall:.4f}")
    print("Recall uses the project's existing metric and the restricted candidate pool described above.")


if __name__ == "__main__":
    main()
