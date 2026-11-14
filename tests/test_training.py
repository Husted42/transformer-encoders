import contextlib
import csv
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from matplotlib.figure import Figure
from transformers import BertTokenizer

from src.retrievers.dpr import DPR
from src.training.loss import DPRLoss
from src.training.trainer import Trainer
from train import main as train_main, save_training_history


class DPRLossTests(unittest.TestCase):
    def test_paired_loss_matches_cross_entropy(self):
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        documents = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        loss = DPRLoss()(queries, documents)
        self.assertTrue(torch.allclose(loss, F.cross_entropy(queries @ documents.T, torch.tensor([0, 1]))))
        self.assertTrue(torch.allclose(loss, DPRLoss()(queries, documents, torch.eye(2, dtype=torch.bool))))
        loss.backward()
        self.assertIsNotNone(queries.grad)
        self.assertIsNotNone(documents.grad)

    def test_multiple_positives_and_invalid_masks(self):
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        documents = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
        mask = torch.tensor([[True, True, False], [False, True, True]])
        scores = queries @ documents.T
        expected = -(torch.logsumexp(scores.masked_fill(~mask, -torch.inf), dim=1)
                     - torch.logsumexp(scores, dim=1)).mean()
        self.assertTrue(torch.allclose(DPRLoss()(queries, documents, mask), expected))
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            DPRLoss()(queries, documents, torch.zeros((2, 3), dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "boolean"):
            DPRLoss()(queries, documents, torch.ones((2, 3)))


class DPRTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        vocab = Path(self.directory.name) / "vocab.txt"
        vocab.write_text(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\ngamma\ndelta\n",
            encoding="utf-8",
        )
        BertTokenizer(str(vocab)).save_pretrained(self.directory.name)
        self.cfg = OmegaConf.create({
            "device": "cpu",
            "retriever": {"batch_size": 4},
            "model": {
                "tokenizer_name": self.directory.name,
                "vocab_size": None,
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "feed_forward_dim": 32,
                "max_len": 8,
                "pooling_type": "mean",
                "dropout": 0.0,
            },
            "training": {"batch_size": 4, "epochs": 40, "learning_rate": 0.01, "seed": 0},
        })
        self.rows = [
            {"question": word, "text": [word], "document_id": [f"d{i}"]}
            for i, word in enumerate(("alpha", "beta", "gamma", "delta"))
        ]
        torch.manual_seed(0)
        self.dpr = DPR(self.cfg)

    def test_training_overfits_and_reindexes(self):
        with self.assertWarns(UserWarning):
            self.dpr.build_index(self.rows)
        trainer = Trainer(self.dpr, self.cfg.training)
        query_weights = self.dpr.query_encoder.token_embedding.embedding.weight.detach().clone()
        document_weights = self.dpr.document_encoder.token_embedding.embedding.weight.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            losses = trainer.train(self.rows)
        self.assertLess(losses[-1], losses[0] * 0.25)
        self.assertFalse(torch.equal(query_weights, self.dpr.query_encoder.token_embedding.embedding.weight))
        self.assertFalse(torch.equal(document_weights, self.dpr.document_encoder.token_embedding.embedding.weight))
        with self.assertRaisesRegex(RuntimeError, "Build a DPR index"):
            self.dpr.Search(["alpha"], 1)
        self.dpr.build_index(self.rows)
        ids, _ = self.dpr.Search([row["question"] for row in self.rows], 1)
        self.assertEqual(ids, [[row["document_id"][0]] for row in self.rows])

    def test_known_other_positives_are_not_negatives(self):
        rows = [
            {"question": "alpha", "text": ["alpha", "beta"], "document_id": ["d1", "d2"]},
            {"question": "beta", "text": ["alpha", "beta"], "document_id": ["d3", "d4"]},
        ]
        self.cfg.training.epochs = 1
        trainer = Trainer(self.dpr, self.cfg.training)
        with patch.object(trainer.criterion, "forward", wraps=trainer.criterion.forward) as forward:
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train(rows)
        self.assertTrue(forward.call_args.args[2].all())
        self.assertEqual(trainer.negative_scores, [None])

    def test_score_plot_displays_scores_and_gaps_without_inventing_missing_negatives(self):
        history = [
            {"epoch": 1, "training_loss": 2.0, "validation_loss": 2.1, "validation_recall": 0.1,
             "positive_score": 4.0, "negative_score": 1.0, "score_gap": 3.0},
            {"epoch": 2, "training_loss": 1.0, "validation_loss": 1.1, "validation_recall": None,
             "positive_score": 5.0, "negative_score": None, "score_gap": None},
            {"epoch": 3, "training_loss": 0.5, "validation_loss": 0.6, "validation_recall": 0.2,
             "positive_score": 6.0, "negative_score": 8.0, "score_gap": -2.0},
        ]
        with patch.object(Figure, "savefig", autospec=True) as savefig:
            save_training_history(Path(self.directory.name), history, top_k=10)
        score_figure = next(call.args[0] for call in savefig.call_args_list
                            if call.args[1].name == "training_scores.png")
        scores_axes, gap_axes = score_figure.axes
        self.assertEqual(list(scores_axes.lines[0].get_ydata()), [4.0, 5.0, 6.0])
        self.assertEqual(list(scores_axes.lines[1].get_ydata()), [1.0, None, 8.0])
        self.assertEqual(list(gap_axes.lines[0].get_ydata()), [3.0, None, -2.0])
        with (Path(self.directory.name) / "training_history.csv").open(newline="", encoding="utf-8") as file:
            saved = list(csv.DictReader(file))
        self.assertEqual(saved[1]["negative_score"], "")
        self.assertEqual(saved[1]["score_gap"], "")
        save_training_history(Path(self.directory.name), [history[1]], top_k=10)
        self.assertEqual((Path(self.directory.name) / "training_scores.png").read_bytes()[:8],
                         b"\x89PNG\r\n\x1a\n")

    def test_known_document_ids_are_not_negatives(self):
        rows = [
            {"question": "alpha", "text": ["alpha"], "document_id": ["shared"]},
            {"question": "beta", "text": ["beta"], "document_id": ["shared"]},
        ]
        self.cfg.training.epochs = 1
        trainer = Trainer(self.dpr, self.cfg.training)
        with patch.object(trainer.criterion, "forward", wraps=trainer.criterion.forward) as forward:
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train(rows)
        self.assertTrue(forward.call_args.args[2].all())

    def test_singleton_last_batch_is_merged(self):
        self.cfg.training.epochs = 1
        self.cfg.training.batch_size = 2
        trainer = Trainer(self.dpr, self.cfg.training)
        with patch.object(trainer.criterion, "forward", wraps=trainer.criterion.forward) as forward:
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train(self.rows + self.rows[:1])
        self.assertEqual([call.args[0].shape[0] for call in forward.call_args_list], [2, 3])

    def test_invalid_training_data_and_settings(self):
        self.cfg.training.batch_size = 1
        with self.assertRaisesRegex(ValueError, "at least 2"):
            Trainer(self.dpr, self.cfg.training)
        self.cfg.training.batch_size = 2
        trainer = Trainer(self.dpr, self.cfg.training)
        with self.assertRaisesRegex(ValueError, "at least two"):
            trainer.train(self.rows[:1])
        with self.assertRaisesRegex(ValueError, "aligned"):
            trainer.train([self.rows[0], {"question": "beta", "text": ["beta"], "document_id": []}])
        with self.assertRaisesRegex(ValueError, "string text"):
            trainer.train([self.rows[0], {"question": "beta", "text": [None], "document_id": ["d1"]}])

    def test_empty_passages_and_rows_are_reported_and_excluded(self):
        self.cfg.training.epochs = 1
        rows = [
            {"question": "alpha", "text": ["alpha", " ", ""], "document_id": ["d0", "d1", "d2"]},
            {"question": "beta", "text": [""], "document_id": ["d3"]},
            self.rows[2],
        ]
        trainer = Trainer(self.dpr, self.cfg.training)
        with patch.object(self.dpr, "tokenize", wraps=self.dpr.tokenize) as tokenize:
            with self.assertWarnsRegex(UserWarning, "Excluded 3 empty passages and 1 training rows"):
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.train(rows)
        tokenized_texts = [text for call in tokenize.call_args_list for text in call.args[0]]
        self.assertNotIn("", tokenized_texts)
        self.assertNotIn(" ", tokenized_texts)
        self.assertNotIn("beta", tokenized_texts)
        self.assertEqual(len(tokenized_texts), 4)

    def test_validation_loss_is_fixed_and_preserves_training_state(self):
        self.cfg.training.batch_size = 2
        self.cfg.model.dropout = 0.3
        dpr = DPR(self.cfg)
        trainer = Trainer(dpr, self.cfg.training)
        rows = [
            {"question": "alpha", "text": ["alpha", "beta"], "document_id": ["d1", "d2"]},
            {"question": "beta", "text": ["alpha", "beta"], "document_id": ["d3", "d4"]},
            *self.rows[1:4],
        ]
        batches = trainer.prepare_validation(rows)
        repeated_batches = trainer.prepare_validation(rows)
        self.assertEqual([len(batch[0]) for batch in batches], [2, 3])
        self.assertEqual([batch[:2] for batch in batches], [batch[:2] for batch in repeated_batches])
        self.assertTrue(batches[0][2].all())
        dpr.query_encoder.train()
        dpr.document_encoder.eval()
        state = torch.random.get_rng_state()
        weights = [parameter.detach().clone() for encoder in (dpr.query_encoder, dpr.document_encoder)
                   for parameter in encoder.parameters()]
        first = trainer.validation_loss(batches)
        self.assertAlmostEqual(first, trainer.validation_loss(batches))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(dpr.query_encoder.training)
        self.assertFalse(dpr.document_encoder.training)
        for before, parameter in zip(weights, (p for encoder in (dpr.query_encoder, dpr.document_encoder)
                                               for p in encoder.parameters())):
            self.assertTrue(torch.equal(before, parameter))
            self.assertIsNone(parameter.grad)
        dpr.query_encoder.eval()
        with torch.no_grad():
            losses = [
                trainer.criterion(dpr.query_encoder(dpr.tokenize(questions)),
                                  dpr.document_encoder(dpr.tokenize(passages)), mask).item()
                for questions, passages, mask in batches
            ]
        self.assertAlmostEqual(first, (2 * losses[0] + 3 * losses[1]) / 5)

    def test_validation_loss_is_omitted_when_fewer_than_two_rows_are_usable(self):
        trainer = Trainer(self.dpr, self.cfg.training)
        with self.assertWarnsRegex(UserWarning, "at least two"):
            self.assertEqual(trainer.prepare_validation(self.rows[:1]), [])

    def test_validation_loss_does_not_change_training_trajectory(self):
        self.cfg.model.dropout = 0.3
        self.cfg.training.epochs = 2
        torch.manual_seed(123)
        first = Trainer(DPR(self.cfg), self.cfg.training)
        torch.manual_seed(123)
        second = Trainer(DPR(self.cfg), self.cfg.training)
        batches = second.prepare_validation(self.rows)
        with contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(456)
            first_losses = first.train(self.rows)
            torch.manual_seed(456)
            second_losses = second.train(self.rows, on_epoch_end=lambda epoch: second.validation_loss(batches))
        self.assertEqual(first_losses, second_losses)
        for a, b in zip(first.dpr.query_encoder.parameters(), second.dpr.query_encoder.parameters()):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(first.dpr.document_encoder.parameters(), second.dpr.document_encoder.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_training_refuses_to_overwrite_checkpoints(self):
        path = Path(self.directory.name) / "query_encoder.pt"
        path.write_bytes(b"existing checkpoint")
        self.cfg.retriever.method = "dpr"
        self.cfg.training.output_dir = self.directory.name
        with self.assertRaisesRegex(FileExistsError, "choose a new"):
            train_main.__wrapped__(self.cfg)
        self.assertEqual(path.read_bytes(), b"existing checkpoint")

    def test_validation_and_epoch_checkpoints(self):
        self.cfg.retriever.method = "dpr"
        self.cfg.dataset = {"name": "fiqa"}
        self.cfg.base_path = self.directory.name
        self.cfg.top_k = 2
        self.cfg.training.epochs = 6
        self.cfg.training.batch_size = 2
        self.cfg.training.output_dir = str(Path(self.directory.name) / "checkpoints")
        self.cfg.training.checkpoint_every_epochs = 5
        self.cfg.training.validation_every_epochs = 5
        self.cfg.training.validation_max_queries = 2
        validation = [dict(row, query_id=f"q{i}") for i, row in enumerate(self.rows)]
        corpus = SimpleNamespace(
            train=self.rows,
            val=validation,
            all_documents=self.rows,
            relevant_ids={row["query_id"]: row["document_id"] for row in validation},
        )
        with patch("train.Corpus", return_value=corpus), \
                patch("train.mean_recall", side_effect=[0.1, 0.4, 0.4]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                train_main.__wrapped__(self.cfg)
        checkpoint_dir = Path(self.cfg.training.output_dir)
        for epoch in (1, 5, 6):
            for name in ("query_encoder.pt", "document_encoder.pt"):
                self.assertTrue((checkpoint_dir / f"epoch_{epoch:03d}" / name).is_file())
            self.assertIn(f"Epoch {epoch}: validation recall@2=", output.getvalue())
        self.assertTrue((checkpoint_dir / "query_encoder.pt").is_file())
        self.assertTrue((checkpoint_dir / "document_encoder.pt").is_file())
        self.assertTrue((checkpoint_dir / "config.yaml").is_file())
        with (checkpoint_dir / "best" / "metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
        self.assertEqual(metadata["epoch"], 5)
        self.assertEqual(metadata["validation_recall"], 0.4)
        for name in ("query_encoder.pt", "document_encoder.pt"):
            best = torch.load(checkpoint_dir / "best" / name, weights_only=True)
            epoch = torch.load(checkpoint_dir / "epoch_005" / name, weights_only=True)
            self.assertTrue(all(torch.equal(best[key], epoch[key]) for key in best))
        for name in ("loss.png", "training_scores.png", "validation_recall.png"):
            self.assertEqual((checkpoint_dir / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with (checkpoint_dir / "training_history.csv").open(newline="", encoding="utf-8") as file:
            history = list(csv.DictReader(file))
        self.assertEqual([int(row["epoch"]) for row in history], list(range(1, 7)))
        self.assertTrue(all(row["training_loss"] and row["validation_loss"] for row in history))
        self.assertEqual([int(row["epoch"]) for row in history if row["validation_recall"]], [1, 5, 6])
        logged = re.findall(r"Epoch (\d+)/6: training loss .*?, positive ([-\d.]+), negative ([-\d.]+)",
                            output.getvalue())
        self.assertEqual(len(logged), 6)
        for row, (epoch, positive, negative) in zip(history, logged):
            self.assertEqual(int(row["epoch"]), int(epoch))
            self.assertAlmostEqual(float(row["positive_score"]), float(positive), delta=0.000051)
            self.assertAlmostEqual(float(row["negative_score"]), float(negative), delta=0.000051)
            self.assertAlmostEqual(float(row["score_gap"]),
                                   float(row["positive_score"]) - float(row["negative_score"]))

    def test_epoch_checkpoint_survives_validation_failure(self):
        self.cfg.retriever.method = "dpr"
        self.cfg.dataset = {"name": "fiqa"}
        self.cfg.base_path = self.directory.name
        self.cfg.top_k = 2
        self.cfg.training.epochs = 2
        self.cfg.training.batch_size = 2
        self.cfg.training.output_dir = str(Path(self.directory.name) / "checkpoints")
        self.cfg.training.checkpoint_every_epochs = 1
        self.cfg.training.validation_every_epochs = 1
        self.cfg.training.validation_max_queries = 2
        validation = [dict(row, query_id=f"q{i}") for i, row in enumerate(self.rows)]
        corpus = SimpleNamespace(
            train=self.rows, val=validation, all_documents=self.rows,
            relevant_ids={row["query_id"]: row["document_id"] for row in validation},
        )
        with patch("train.Corpus", return_value=corpus):
            with patch("train.retrieveQueryAndGetRelevancies", side_effect=RuntimeError("evaluation failed")):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                        train_main.__wrapped__(self.cfg)
        checkpoint = Path(self.cfg.training.output_dir) / "epoch_001"
        self.assertTrue((checkpoint / "query_encoder.pt").is_file())
        self.assertTrue((checkpoint / "document_encoder.pt").is_file())


if __name__ == "__main__":
    unittest.main()
