import unittest

import numpy as np
import torch

from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.evaluation.metrics import retrieval_metrics
from src.retrievers.bm25 import BM25
from src.retrievers.models.encoder import ScaledDotProductAttention, TransformerEncoder
from src.retrievers.tf_idf import TFIDF
from src.retrievers.vector_database import VectorDatabase, encode_token_ids
from src.training.loss import DPRLoss
from src.training.trainer import _positive_mask, _prepare_examples


class RetrievalContractTests(unittest.TestCase):
    def test_fiqa_qrels_and_shared_passage(self):
        corpus = [{"_id": "d", "title": "Title", "text": "passage"},
                  {"_id": "e", "title": "", "text": ""}]
        queries = [{"_id": "q1", "text": "one"}, {"_id": "q2", "text": "two"}]
        qrels = {"train": [{"query-id": "q1", "corpus-id": "d", "score": 1},
                           {"query-id": "q2", "corpus-id": "d", "score": 1}],
                 "validation": [{"query-id": "q2", "corpus-id": "e", "score": 1}]}
        data = Corpus._fiqa((corpus, queries, qrels))
        self.assertEqual(len(data["corpus"]), 2)
        self.assertEqual(data["train"][0]["text"], ["Title\npassage"])
        self.assertEqual(data["train"][1]["document_id"], ["d"])
        with self.assertRaisesRegex(ValueError, "Dangling"):
            Corpus._fiqa((corpus, queries, {"train": [
                {"query-id": "q1", "corpus-id": "missing", "score": 1}]}))

    def test_squad_contexts_are_shared_across_splits(self):
        data = Corpus._squad({"train": [{"id": "1", "question": "q1", "context": "shared"}],
                              "validation": [{"id": "2", "question": "q2", "context": "shared"}]})
        self.assertEqual(len(data["corpus"]), 1)
        self.assertEqual(data["train"][0]["document_id"], data["validation"][0]["document_id"])

    def test_mask_uses_query_ids_and_all_known_ids(self):
        rows = [{"query_id": "q1", "question": "same", "text": ["shared", "other"],
                 "document_id": ["d", "e"]},
                {"query_id": "q2", "question": "same", "text": ["shared"],
                 "document_id": ["d"]}]
        examples, positives = _prepare_examples(rows, "training")
        mask = _positive_mask([example[0] for example in examples],
                              [("shared", "d"), ("other", "e")], positives, "cpu")
        self.assertEqual(mask.tolist(), [[True, True], [True, False]])

    def test_baselines_handle_empty_unseen_and_rank(self):
        docs = [{"text": ["bank loan"], "document_id": ["a"]},
                {"text": ["water river"], "document_id": ["b"]},
                {"text": [""], "document_id": ["c"]}]
        for cls in (TFIDF, BM25):
            model = cls(docs)
            self.assertEqual(model.Lookup(["BANK loan"], 1)[0][0].GetId(), "a")
            np.testing.assert_array_equal(model.CalculateScores(["unseen"]), [[0, 0, 0]])
            self.assertEqual(model.Lookup([], 1), [])

    def test_metrics_use_standard_denominators(self):
        query = Query("q", "q", ["a", "b", "c", "d"])
        values = retrieval_metrics([[False, True]], [query], 5)
        self.assertEqual(values, {"recall@5": 0.25, "precision@5": 0.2, "mrr@5": 0.5})

    def test_masked_attention_and_optional_maps(self):
        q = torch.ones(1, 2, 2, requires_grad=True)
        output, weights = ScaledDotProductAttention()(q, q, q, torch.zeros(1, 1, 2, dtype=torch.bool))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))
        output.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        encoder = TransformerEncoder(10, 4, 2, 2, 8, 8, dropout=0)
        pooled, attention = encoder(torch.tensor([[1, 2, 0]]), return_attention=True)
        self.assertEqual(pooled.shape, (1, 4))
        self.assertEqual(len(attention), 2)
        self.assertEqual(attention[0].shape, (1, 2, 3, 3))

    def test_loss_contract_and_vector_restoration(self):
        with self.assertRaisesRegex(ValueError, "same width"):
            DPRLoss()(torch.ones(2, 3), torch.ones(2, 4))
        with self.assertRaisesRegex(ValueError, "finite"):
            DPRLoss()(torch.tensor([[float("nan")]]), torch.ones(1, 1))
        encoder = torch.nn.Linear(2, 2)
        encoder.train()
        with self.assertRaises(RuntimeError):
            encode_token_ids(encoder, torch.ones(1, 3))
        self.assertTrue(encoder.training)
        database = VectorDatabase(2)
        with self.assertRaisesRegex(ValueError, "finite"):
            database.insert(["a"], [[float("inf"), 0]])
        database.insert(["a"], [[1, 0]])
        with self.assertRaisesRegex(ValueError, "positive"):
            database.query([[1, 0]], 0)


if __name__ == "__main__":
    unittest.main()
