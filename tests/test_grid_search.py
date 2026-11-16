import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from transformers import BertTokenizer

from grid_search import run_grid


class GridSearchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cfg = OmegaConf.create({
            "device": "cpu", "base_path": str(self.root), "top_k": 1,
            "dataset": {"name": "fiqa"}, "retriever": {"method": "dpr", "batch_size": 4},
            "model": {"tokenizer_name": "", "vocab_size": None, "d_model": 16, "n_heads": 2,
                      "n_layers": 1, "feed_forward_dim": 32, "max_len": 8,
                      "pooling_type": "mean", "dropout": 0.0},
            "training": {"batch_size": 4, "epochs": 4, "learning_rate": 0.01,
                         "weight_decay": 0.0, "seed": 0, "output_dir": "",
                         "checkpoint_every_epochs": 1, "validation_every_epochs": 1,
                         "validation_max_queries": 4},
        })

    def test_grid_runs_all_combinations_and_ranks_best(self):
        calls = []

        def fake_train(cfg):
            calls.append((cfg.model.pooling_type, cfg.training.learning_rate,
                          cfg.model.max_len, cfg.training.epochs, cfg.training.seed))
            best = Path(cfg.training.output_dir) / "best"
            best.mkdir(parents=True)
            for name in ("query_encoder.pt", "document_encoder.pt"):
                (best / name).write_bytes(b"weights")
            with (best / "metadata.json").open("w", encoding="utf-8") as file:
                json.dump({"validation_recall": len(calls) / 10, "epoch": 2,
                           "top_k": 1, "validation_queries": 4}, file)

        with patch("grid_search.train_main.__wrapped__", side_effect=fake_train), \
                contextlib.redirect_stdout(io.StringIO()):
            run_grid(self.cfg, self.root / "grid")
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(set(calls)), 8)
        self.assertTrue(all(call[3:] == (24, 0) for call in calls))
        with (self.root / "grid" / "summary.csv").open(newline="", encoding="utf-8") as file:
            results = list(csv.DictReader(file))
        self.assertEqual([r["run"] for r in results], [f"run-{i:03d}" for i in range(8, 0, -1)])
        self.assertEqual(float(results[0]["validation_recall"]), 0.8)
        with self.assertRaises(FileExistsError):
            run_grid(self.cfg, self.root / "grid")

    def test_failure_is_recorded_and_propagated(self):
        with patch("grid_search.train_main.__wrapped__", side_effect=RuntimeError("broken")), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "broken"):
                run_grid(self.cfg, self.root / "grid", pooling_types=("cls",),
                         learning_rates=(0.0001,), max_lengths=(128,))
        with (self.root / "grid" / "summary.csv").open(newline="", encoding="utf-8") as file:
            result = next(csv.DictReader(file))
        self.assertEqual(result["status"], "failed")
        self.assertIn("RuntimeError: broken", result["error"])

    def test_real_training_two_runs(self):
        vocab = self.root / "vocab.txt"
        vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\ngamma\ndelta\n",
                         encoding="utf-8")
        BertTokenizer(str(vocab)).save_pretrained(self.root)
        self.cfg.model.tokenizer_name = str(self.root)
        rows = [{"question": word, "text": [word], "document_id": [f"d{i}"], "query_id": f"q{i}"}
                for i, word in enumerate(("alpha", "beta", "gamma", "delta"))]
        corpus = SimpleNamespace(train=rows, val=rows, all_documents=rows,
                                 relevant_ids={row["query_id"]: row["document_id"] for row in rows})
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with patch("train.Corpus", return_value=corpus), contextlib.redirect_stdout(io.StringIO()):
                run_grid(self.cfg, self.root / "grid", epochs=1, pooling_types=("cls", "mean"),
                         learning_rates=(0.001,), max_lengths=(8,))
        finally:
            torch.set_num_threads(old_threads)
        with (self.root / "grid" / "summary.csv").open(newline="", encoding="utf-8") as file:
            results = list(csv.DictReader(file))
        self.assertEqual({row["pooling_type"] for row in results}, {"cls", "mean"})
        for result in results:
            self.assertEqual(result["status"], "completed")
            checkpoint = Path(result["checkpoint"])
            self.assertTrue((checkpoint / "query_encoder.pt").is_file())
            self.assertTrue((checkpoint / "document_encoder.pt").is_file())
            self.assertTrue((checkpoint.parent / "loss.png").is_file())


if __name__ == "__main__":
    unittest.main()
