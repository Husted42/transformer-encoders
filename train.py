import csv
import json
import random
from pathlib import Path

import hydra
import torch
from matplotlib.figure import Figure
from omegaconf import OmegaConf

from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.evaluation.metrics import mean_recall
from src.retrievers.dpr import DPR
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.training.trainer import Trainer


def save_training_history(output_dir, history, top_k):
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "epoch", "training_loss", "validation_loss", "positive_score",
            "negative_score", "score_gap", "validation_recall",
            "learning_rate", "epoch_seconds",
        ])
        writer.writeheader()
        writer.writerows(history)

    figure = Figure(figsize=(8, 5), layout="constrained")
    axes = figure.subplots()
    axes.plot([row["epoch"] for row in history], [row["training_loss"] for row in history],
              marker="o", label="Training loss")
    validation = [row for row in history if row["validation_loss"] is not None]
    if validation:
        axes.plot([row["epoch"] for row in validation], [row["validation_loss"] for row in validation],
                  marker="o", label="Validation loss")
    axes.set(xlabel="Epoch", ylabel="In-batch loss", title="Training and validation loss")
    axes.legend()
    axes.grid(True, alpha=0.3)
    figure.savefig(output_dir / "loss.png")

    figure = Figure(figsize=(8, 7), layout="constrained")
    scores_axes, gap_axes = figure.subplots(2, 1, sharex=True)
    epochs = [row["epoch"] for row in history]
    scores_axes.plot(epochs, [row["positive_score"] for row in history], marker="o", label="Positive")
    scores_axes.plot(epochs, [row["negative_score"] for row in history], marker="o", label="Negative")
    scores_axes.set(ylabel="Mean training score", title="Positive and negative training scores")
    scores_axes.legend()
    scores_axes.grid(True, alpha=0.3)
    gap_axes.plot(epochs, [row["score_gap"] for row in history], marker="o", label="Positive - negative")
    gap_axes.axhline(0, color="black", linewidth=0.8)
    gap_axes.set(xlabel="Epoch", ylabel="Score gap")
    gap_axes.legend()
    gap_axes.grid(True, alpha=0.3)
    figure.savefig(output_dir / "training_scores.png")

    recall = [row for row in history if row["validation_recall"] is not None]
    if recall:
        figure = Figure(figsize=(8, 5), layout="constrained")
        axes = figure.subplots()
        axes.bar([str(row["epoch"]) for row in recall], [row["validation_recall"] for row in recall])
        axes.set(xlabel="Epoch", ylabel=f"Validation recall@{top_k}", title="Validation recall")
        axes.set_ylim(bottom=0)
        figure.savefig(output_dir / "validation_recall.png")


@hydra.main(config_path="config", config_name="fiqa_experiment.yaml", version_base=None)
def main(cfg):
    if cfg.retriever.method != "dpr":
        raise ValueError("Training requires retriever=dpr.")

    output_dir = Path(cfg.training.output_dir)
    query_path = output_dir / "query_encoder.pt"
    document_path = output_dir / "document_encoder.pt"
    config_path = output_dir / "config.yaml"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training output already exists in {output_dir}; choose a new training.output_dir.")
    if cfg.training.checkpoint_every_epochs < 1 or cfg.training.validation_every_epochs < 1:
        raise ValueError("Checkpoint and validation intervals must be positive.")
    if cfg.training.validation_max_queries < 1:
        raise ValueError("validation_max_queries must be positive.")

    torch.manual_seed(cfg.training.seed)
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f"{cfg.base_path}/retrieval_data")
    dpr = DPR(cfg)
    if corpus.val is None or len(corpus.val) == 0:
        raise ValueError("DPR training requires a nonempty validation split.")
    indices = random.Random(cfg.training.seed).sample(
        range(len(corpus.val)), min(cfg.training.validation_max_queries, len(corpus.val))
    )
    validation_queries = []
    validation_rows = []
    for index in indices:
        row = corpus.val[index]
        relevant_ids = corpus.relevant_ids[row["query_id"]]
        if not relevant_ids:
            raise ValueError(f"Validation query {row['query_id']} has no relevant documents.")
        validation_queries.append(Query(row["question"], row["query_id"], relevant_ids))
        validation_rows.append(row)
    documents = corpus.all_documents
    trainer = Trainer(dpr, cfg.training)
    validation_batches = trainer.prepare_validation(validation_rows)
    history = []
    best_recall = None

    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)

    def on_epoch_end(epoch):
        nonlocal best_recall
        if epoch == 1 or epoch % cfg.training.checkpoint_every_epochs == 0 or epoch == cfg.training.epochs:
            checkpoint_dir = output_dir / f"epoch_{epoch:03d}"
            checkpoint_dir.mkdir()
            torch.save(dpr.query_encoder.state_dict(), checkpoint_dir / "query_encoder.pt")
            torch.save(dpr.document_encoder.state_dict(), checkpoint_dir / "document_encoder.pt")
            print(f"Saved epoch {epoch} encoder checkpoints to {checkpoint_dir}")
        validation_loss = trainer.validation_loss(validation_batches) if validation_batches else None
        if validation_loss is not None:
            print(f"Epoch {epoch}: validation loss {validation_loss:.4f}")
        score = None
        if epoch == 1 or epoch % cfg.training.validation_every_epochs == 0 or epoch == cfg.training.epochs:
            print(f"Building the document index for validation at epoch {epoch}...")
            dpr.build_index(documents)
            relevancies = retrieveQueryAndGetRelevancies(dpr, validation_queries, k=cfg.top_k)
            score = mean_recall(relevancies, validation_queries)
            print(f"Epoch {epoch}: validation recall@{cfg.top_k}={score:.4f} "
                  f"({len(validation_queries)} queries)")
            if best_recall is None or score > best_recall:
                best_recall = score
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save(dpr.query_encoder.state_dict(), best_dir / "query_encoder.pt")
                torch.save(dpr.document_encoder.state_dict(), best_dir / "document_encoder.pt")
                metadata = {
                    "epoch": epoch, "validation_recall": score, "top_k": cfg.top_k,
                    "validation_queries": len(validation_queries),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
                with (best_dir / "metadata.json").open("w", encoding="utf-8") as file:
                    json.dump(metadata, file, indent=2)
                print(f"Saved best checkpoint at epoch {epoch} to {best_dir}")
        positive_score = trainer.positive_scores[-1]
        negative_score = trainer.negative_scores[-1]
        history.append({
            "epoch": epoch,
            "training_loss": trainer.losses[-1],
            "validation_loss": validation_loss,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "score_gap": positive_score - negative_score if negative_score is not None else None,
            "validation_recall": score,
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            "epoch_seconds": trainer.epoch_durations[-1],
        })
        save_training_history(output_dir, history, cfg.top_k)

    trainer.train(corpus.train, on_epoch_end=on_epoch_end)

    torch.save(dpr.query_encoder.state_dict(), query_path)
    torch.save(dpr.document_encoder.state_dict(), document_path)
    print(f"Saved encoder checkpoints and configuration to {output_dir}")


if __name__ == "__main__":
    main()
