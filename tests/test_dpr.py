import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import BertTokenizer

from src.data_utils.query import Query
from src.evaluation.metrics import Evaluate
from src.retrievers.dpr import DPR
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.retrievers.vector_database import encode_token_ids


def make_config():
    return OmegaConf.create({
        "model": {
            "vocab_size": 50,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "feed_forward_dim": 32,
            "max_len": 8,
            "pooling_type": "mean",
        }
    })


class DPRModelLoadingTests(unittest.TestCase):
    def test_selecting_dpr_includes_encoder_config(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            default_cfg = compose(config_name="fiqa_experiment")
            dpr_cfg = compose(config_name="fiqa_experiment", overrides=["retriever=dpr"])
            tfidf_cfg = compose(config_name="fiqa_experiment", overrides=["retriever=tf_idf"])
        if default_cfg.retriever.method == "dpr":
            self.assertEqual(default_cfg.model.tokenizer_name, "bert-base-uncased")
        self.assertEqual(dpr_cfg.model.tokenizer_name, "bert-base-uncased")
        self.assertEqual(dpr_cfg.retriever.method, "dpr")
        self.assertNotIn("model", tfidf_cfg)

    def test_initializes_independent_encoders(self):
        dpr = DPR(make_config())
        self.assertIsNot(dpr.query_encoder, dpr.document_encoder)
        self.assertIsNot(
            dpr.query_encoder.token_embedding.embedding.weight,
            dpr.document_encoder.token_embedding.embedding.weight,
        )
        tokens = torch.tensor([[1, 2, 0]])
        self.assertEqual(dpr.query_encoder(tokens).shape, (1, 16))
        self.assertEqual(dpr.document_encoder(tokens).shape, (1, 16))

    def test_loads_each_checkpoint_independently(self):
        source = DPR(make_config())
        with torch.no_grad():
            source.query_encoder.token_embedding.embedding.weight.fill_(1)
            source.document_encoder.token_embedding.embedding.weight.fill_(2)

        with tempfile.TemporaryDirectory() as directory:
            query_path = Path(directory) / "query.pt"
            document_path = Path(directory) / "document.pt"
            torch.save(source.query_encoder.state_dict(), query_path)
            torch.save(source.document_encoder.state_dict(), document_path)

            for name, path, expected in (
                ("query_encoder", query_path, source.query_encoder),
                ("document_encoder", document_path, source.document_encoder),
            ):
                with self.subTest(encoder=name):
                    cfg = make_config()
                    cfg.model[f"{name}_weights_path"] = str(path)
                    loaded = getattr(DPR(cfg), name)
                    for key, value in expected.state_dict().items():
                        self.assertTrue(torch.equal(loaded.state_dict()[key], value), key)

    def test_missing_checkpoint_raises(self):
        cfg = make_config()
        cfg.model.document_encoder_weights_path = "missing-document-checkpoint.pt"
        with self.assertRaises(FileNotFoundError):
            DPR(cfg)


class DPRRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        vocab = Path(self.directory.name) / "vocab.txt"
        vocab.write_text(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nfinance\nbank\nloan\nwater\nriver\n",
            encoding="utf-8",
        )
        BertTokenizer(str(vocab)).save_pretrained(self.directory.name)
        self.cfg = make_config()
        self.cfg.model.tokenizer_name = self.directory.name
        self.cfg.model.vocab_size = None
        self.cfg.model.pooling_type = "cls"
        self.cfg.retriever = {"batch_size": 2}
        self.documents = [
            {"text": ["finance bank", "bank loan"], "document_id": ["d1", "d2"]},
            {"text": ["water river"], "document_id": ["d3"]},
        ]

    def test_index_search_lookup_and_evaluate(self):
        with self.assertWarnsRegex(UserWarning, "randomly initialized"):
            dpr = DPR(self.cfg, self.documents)
        self.assertEqual(len(dpr.retriever.db), 3)
        token_ids = dpr.tokenize(["finance bank", "water"])
        self.assertTrue(torch.all(token_ids[:, 0] == dpr.tokenizer.cls_token_id))
        self.assertEqual(token_ids[1, -1].item(), dpr.tokenizer.pad_token_id)

        queries = ["finance", "river", "bank"]
        ids, scores = dpr.Search(queries, k=5)
        self.assertEqual(scores.shape, (3, 3))
        self.assertTrue((scores[:, :-1] >= scores[:, 1:]).all())
        self.assertTrue(all(set(row) == {"d1", "d2", "d3"} for row in ids))
        query_vectors = encode_token_ids(dpr.query_encoder, dpr.tokenize(queries)).numpy()
        all_scores = query_vectors @ dpr.retriever.db.vectors.T
        positions = {doc_id: i for i, doc_id in enumerate(dpr.retriever.db.ids)}
        for i, row in enumerate(ids):
            np.testing.assert_allclose(scores[i], [all_scores[i, positions[doc_id]] for doc_id in row], atol=1e-5)
        docs = dpr.Lookup(queries, k=5)
        self.assertEqual([[doc.GetId() for doc in row] for row in docs], ids)
        self.assertEqual(
            {doc.GetId(): doc.GetText() for doc in docs[0]},
            {"d1": "finance bank", "d2": "bank loan", "d3": "water river"},
        )

        labeled = [Query("finance", "q1", ["d1"]), Query("river", "q2", ["d3"])]
        relevancies = retrieveQueryAndGetRelevancies(dpr, labeled, k=3)
        self.assertEqual(Evaluate(relevancies, labeled)["recall"], 1.0)
        self.assertEqual(dpr.Search([], 3)[1].shape, (0, 3))

    def test_invalid_inputs_are_explicit(self):
        dpr = DPR(self.cfg)
        with self.assertRaisesRegex(RuntimeError, "Build a DPR index"):
            dpr.Search(["finance"], 1)
        with self.assertRaisesRegex(ValueError, "without documents"):
            dpr.build_index([])
        with self.assertRaisesRegex(ValueError, "unique"):
            dpr.build_index([{"text": ["bank", "loan"], "document_id": ["d1", "d1"]}])
        with self.assertWarns(UserWarning):
            dpr.build_index(self.documents)
        with self.assertRaisesRegex(ValueError, "k must be positive"):
            dpr.Lookup(["finance"], 0)
        self.cfg.model.vocab_size = 100
        with self.assertRaisesRegex(ValueError, "vocab_size must match"):
            DPR(self.cfg)
        self.cfg.model.vocab_size = None
        self.cfg.model.padding_idx = 1
        with self.assertRaisesRegex(ValueError, "padding_idx must match"):
            DPR(self.cfg)

    def test_checkpoints_reproduce_retrieval(self):
        with self.assertWarns(UserWarning):
            original = DPR(self.cfg, self.documents)
        query_path = Path(self.directory.name) / "query.pt"
        document_path = Path(self.directory.name) / "document.pt"
        torch.save(original.query_encoder.state_dict(), query_path)
        torch.save(original.document_encoder.state_dict(), document_path)
        self.cfg.model.query_encoder_weights_path = str(query_path)
        self.cfg.model.document_encoder_weights_path = str(document_path)

        restored = DPR(self.cfg, self.documents)
        original_ids, original_scores = original.Search(["finance", "river"], 3)
        restored_ids, restored_scores = restored.Search(["finance", "river"], 3)
        self.assertEqual(restored_ids, original_ids)
        np.testing.assert_allclose(restored_scores, original_scores)


if __name__ == "__main__":
    unittest.main()
