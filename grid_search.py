import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from train import main as train_main


ROOT = Path(__file__).resolve().parent
POOLING_TYPES = ("cls", "mean")
LEARNING_RATES = (0.001, 0.0001, 0.0003, 0.00001)
MAX_LENGTHS = (128, 256)
FIELDS = ("run", "pooling_type", "learning_rate", "max_len", "status",
          "best_epoch", "validation_recall", "top_k", "validation_queries", "checkpoint", "error")


def write_summary(output_dir, results):
    ranked = sorted(results, key=lambda row: (
        row["status"] != "completed",
        -row["validation_recall"] if row["status"] == "completed" else 0,
        row["run"],
    ))
    temporary = output_dir / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ranked)
    temporary.replace(output_dir / "summary.csv")


def run_grid(cfg, output_dir, epochs=24, pooling_types=POOLING_TYPES,
             learning_rates=LEARNING_RATES, max_lengths=MAX_LENGTHS):
    if epochs < 1:
        raise ValueError("Grid epochs must be positive.")
    if cfg.retriever.method != "dpr":
        raise ValueError("Grid search requires retriever=dpr.")
    combinations = list(itertools.product(pooling_types, learning_rates, max_lengths))
    if not combinations or len(set(combinations)) != len(combinations):
        raise ValueError("Grid must contain distinct combinations.")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for number, (pooling, learning_rate, max_len) in enumerate(combinations, start=1):
        run_name = f"run-{number:03d}"
        run_dir = output_dir / run_name
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        run_cfg.model.pooling_type = pooling
        run_cfg.model.max_len = max_len
        run_cfg.training.learning_rate = learning_rate
        run_cfg.training.epochs = epochs
        run_cfg.training.output_dir = str(run_dir)
        row = {
            "run": run_name, "pooling_type": pooling, "learning_rate": learning_rate,
            "max_len": max_len, "status": "failed", "best_epoch": "",
            "validation_recall": "", "top_k": cfg.top_k, "validation_queries": "",
            "checkpoint": str(run_dir / "best"), "error": "",
        }
        print(f"\nGrid run {number}/{len(combinations)}: {pooling=}, {learning_rate=}, {max_len=}",
              flush=True)
        try:
            train_main.__wrapped__(run_cfg)
            with (run_dir / "best" / "metadata.json").open(encoding="utf-8") as file:
                metadata = json.load(file)
            score = float(metadata["validation_recall"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid validation recall: {score}")
            if metadata["top_k"] != cfg.top_k:
                raise ValueError("Checkpoint recall k does not match the grid configuration.")
            if not all((run_dir / "best" / name).is_file()
                       for name in ("query_encoder.pt", "document_encoder.pt")):
                raise ValueError("Best checkpoint is missing encoder weights.")
            row.update(status="completed", best_epoch=metadata["epoch"], validation_recall=score,
                       validation_queries=metadata["validation_queries"])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            results.append(row)
            write_summary(output_dir, results)
            raise
        results.append(row)
        write_summary(output_dir, results)
    print(f"\nGrid summary: {output_dir / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Sequential DPR grid search ranked by validation recall.")
    parser.add_argument("--output", default="outputs/dpr/grid-01",
                        help="New directory for all runs and summary.csv.")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Hydra override applied to all runs; may be supplied multiple times.")
    args = parser.parse_args()
    controlled = ("retriever", "training.output_dir", "training.epochs", "training.learning_rate",
                  "model.pooling_type", "model.max_len")
    for override in args.override:
        key = override.split("=", 1)[0].lstrip("+~")
        if key in controlled:
            parser.error(f"{key} is controlled by the grid search")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        cfg = compose(config_name="fiqa_experiment", overrides=["retriever=dpr", *args.override])
    run_grid(cfg, args.output, args.epochs)


if __name__ == "__main__":
    main()
