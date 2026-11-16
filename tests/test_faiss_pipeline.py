import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import BertTokenizer

from src.data_utils.query import Query
from src.evaluation.metrics import mean_recall
from src.retrievers.build_index import build_index
from src.retrievers.search import search
from src.retrievers.vector_store import FaissVectorStore
from train import main as train_main


class FaissPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_recall_uses_all_relevant_documents(self):
        query = Query("example", "q", ["a", "b", "c"])
        self.assertAlmostEqual(mean_recall([[True, True]], [query]), 2 / 3)
        with self.assertRaisesRegex(ValueError, "at least one relevant"):
            mean_recall([[False]], [Query("missing", "q2", [])])

    def test_exact_search_survives_reload_and_validates_input(self):
        documents = [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]
        store = FaissVectorStore(2, {"dataset": "example"})
        with self.assertRaisesRegex(ValueError, "shape"):
            store.build(documents, [[1, 0]])
        store.build(documents, [[1, 0], [0, 1]])
        with tempfile.TemporaryDirectory() as directory:
            store.save(directory)
            restored = FaissVectorStore.load(directory)
            self.assertEqual(restored.index.ntotal, 2)
            results = restored.search([[0.1, 0.9]], 5)[0]
            self.assertEqual([result["id"] for result in results], ["b", "a"])
            self.assertEqual(results[0]["text"], "second")
            np.testing.assert_allclose([result["score"] for result in results], [0.9, 0.1])
            with self.assertRaisesRegex(ValueError, "positive k"):
                restored.search([[1, 0]], 0)
            with self.assertRaises(FileExistsError):
                store.save(directory)

    def test_training_best_checkpoint_faiss_build_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = root / "tokenizer"
            tokenizer_path.mkdir()
            vocab = tokenizer_path / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\ngamma\ndelta\n",
                             encoding="utf-8")
            BertTokenizer(str(vocab)).save_pretrained(tokenizer_path)
            cfg = OmegaConf.create({
                "device": "cpu", "base_path": directory, "top_k": 1,
                "dataset": {"name": "fiqa"}, "retriever": {"method": "dpr", "batch_size": 4},
                "model": {"tokenizer_name": str(tokenizer_path), "vocab_size": None,
                          "d_model": 16, "n_heads": 2, "n_layers": 1, "feed_forward_dim": 32,
                          "max_len": 8, "pooling_type": "mean", "dropout": 0.0},
                "training": {"batch_size": 4, "epochs": 40, "learning_rate": 0.01,
                             "weight_decay": 0.0, "seed": 0, "output_dir": str(root / "run"),
                             "checkpoint_every_epochs": 20, "validation_every_epochs": 10,
                             "validation_max_queries": 4},
            })
            rows = [{"question": word, "text": [word], "document_id": [f"d{i}"], "query_id": f"q{i}"}
                    for i, word in enumerate(("alpha", "beta", "gamma", "delta"))]
            corpus = SimpleNamespace(train=rows, val=rows, all_documents=rows,
                                     relevant_ids={row["query_id"]: row["document_id"] for row in rows})
            checkpoint = root / "run" / "best"
            index = root / "index"
            with contextlib.redirect_stdout(io.StringIO()):
                with patch("train.Corpus", return_value=corpus):
                    train_main.__wrapped__(cfg)
                with patch("src.retrievers.build_index.Corpus", return_value=corpus):
                    build_index(checkpoint, index, "cpu")
            with (checkpoint / "metadata.json").open(encoding="utf-8") as file:
                metadata = json.load(file)
            self.assertEqual(metadata["validation_recall"], 1.0)
            self.assertEqual(FaissVectorStore.load(index).index.ntotal, len(rows))
            for row in rows:
                result = search(row["question"], checkpoint, index, k=1, device="cpu")[0]
                self.assertEqual(result["id"], row["document_id"][0])
                self.assertEqual(result["text"], row["text"][0])
            with (index / "metadata.json").open(encoding="utf-8") as file:
                index_metadata = json.load(file)
            index_metadata["checkpoint_hashes"]["document"] = "different"
            with (index / "metadata.json").open("w", encoding="utf-8") as file:
                json.dump(index_metadata, file)
            with self.assertRaisesRegex(ValueError, "does not match"):
                search("alpha", checkpoint, index, device="cpu")


if __name__ == "__main__":
    unittest.main()
