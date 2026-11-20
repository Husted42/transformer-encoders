"""NOTE : Debugging was build with AI """
"""Offline diagnostic: overfit one small, fixed batch of real FIQA pairs."""

import math
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from huggingface_hub import try_to_load_from_cache
from hydra import compose, initialize_config_dir

from src.retrievers.dpr import DPR
from src.retrievers.vector_database import encode_token_ids
from src.training.trainer import Trainer


ROOT = Path(__file__).resolve().parents[2]
FIQA_PATH = ROOT / "retrieval_data" / "fiqa-retrieval-v1"
BATCH_SIZE = 8
EPOCHS = 40


def fixed_pairs(train):
    indices = list(range(len(train)))
    random.Random(42).shuffle(indices)
    rows = []
    known_passages = []
    for index in indices:
        row = train[index]
        valid = [(text, doc_id) for text, doc_id in zip(row["text"], row["document_id"])
                 if isinstance(text, str) and text.strip()]
        if not valid or any(row["question"] == previous["question"] for previous in rows):
            continue
        text, doc_id = valid[0]
        all_positives = {passage for passage, _ in valid}
        if any(text in positives or previous["text"][0] in all_positives
               for previous, positives in zip(rows, known_passages)):
            continue
        rows.append({"query_id": row["query_id"], "question": row["question"],
                     "text": [text], "document_id": [doc_id]})
        known_passages.append(all_positives)
        if len(rows) == BATCH_SIZE:
            return rows
    raise ValueError(f"Could not select {BATCH_SIZE} distinct FIQA query/passage pairs.")


def evaluate_batch(dpr, rows):
    queries = dpr.tokenize([row["question"] for row in rows])
    passages = dpr.tokenize([row["text"][0] for row in rows])
    query_vectors = encode_token_ids(dpr.query_encoder, queries)
    document_vectors = encode_token_ids(dpr.document_encoder, passages)
    scores = query_vectors @ document_vectors.T
    positives = scores.diag()
    negatives = scores[~torch.eye(len(rows), dtype=torch.bool, device=scores.device)]
    return {
        "loss": F.cross_entropy(scores, torch.arange(len(rows), device=scores.device)).item(),
        "positive": positives.mean().item(),
        "negative": negatives.mean().item(),
        "top1": (scores.argmax(dim=1) == torch.arange(len(rows), device=scores.device)).float().mean().item(),
    }


class RealFIQAOverfitTest(unittest.TestCase):
    def test_score_matrix_gradients_and_fixed_batch_overfit(self):
        if not FIQA_PATH.is_dir():
            self.skipTest("Cached FIQA data not found at retrieval_data/fiqa.")
        tokenizer_config = try_to_load_from_cache("bert-base-uncased", "tokenizer_config.json")
        if not isinstance(tokenizer_config, str):
            self.skipTest("bert-base-uncased tokenizer is not available in the Hugging Face cache.")

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
                    cfg = compose(config_name="fiqa_experiment", overrides=[
                        "retriever=dpr",
                        "model.pooling_type=mean",
                        "model.query_encoder_weights_path=null",
                        "model.document_encoder_weights_path=null",
                        f"training.batch_size={BATCH_SIZE}",
                        f"training.epochs={EPOCHS}",
                        "training.learning_rate=0.0001",
                    ])
                cfg.device = "cpu"
                cfg.model.tokenizer_name = str(Path(tokenizer_config).parent)
                rows = fixed_pairs(load_from_disk(str(FIQA_PATH))["train"])
                dpr = DPR(cfg)
                trainer = Trainer(dpr, cfg.training)
                initial = evaluate_batch(dpr, rows)

                original_loss = trainer.criterion.forward
                original_step = trainer.optimizer.step
                gradients = []
                batches = []

                def check_loss(queries, documents, positive_mask):
                    scores = queries @ documents.T
                    self.assertEqual(tuple(scores.shape), (BATCH_SIZE, BATCH_SIZE))
                    self.assertTrue(torch.isfinite(scores).all())
                    self.assertTrue(torch.equal(positive_mask, torch.eye(BATCH_SIZE, dtype=torch.bool)))
                    for i in range(BATCH_SIZE):
                        for j in range(BATCH_SIZE):
                            self.assertTrue(torch.allclose(scores[i, j], torch.dot(queries[i], documents[j]), atol=1e-5))
                    loss = original_loss(queries, documents, positive_mask)
                    expected = F.cross_entropy(scores, torch.arange(BATCH_SIZE, device=scores.device))
                    self.assertTrue(torch.allclose(loss, expected))
                    detached = scores.detach()
                    negatives = detached[~torch.eye(BATCH_SIZE, dtype=torch.bool, device=scores.device)]
                    batches.append((loss.detach().item(), detached.diag().mean().item(), negatives.mean().item()))
                    return loss

                def check_step():
                    norms = []
                    for encoder in (dpr.query_encoder, dpr.document_encoder):
                        parameters = list(encoder.parameters())
                        self.assertTrue(all(parameter.grad is not None for parameter in parameters))
                        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in parameters))
                        norm = math.sqrt(sum(parameter.grad.detach().square().sum().item()
                                             for parameter in parameters))
                        self.assertGreater(norm, 0.0)
                        norms.append(norm)
                    gradients.append(norms)
                    if len(gradients) == 1 or len(gradients) % 10 == 0:
                        loss, positive, negative = batches[-1]
                        print(f"  batch {len(gradients)}: loss={loss:.4f}, "
                              f"positive={positive:.4f}, negative={negative:.4f}, "
                              f"query_grad={norms[0]:.4f}, document_grad={norms[1]:.4f}")
                    return original_step()

                print(f"\nFixed FIQA batch before training: {initial}")
                with patch.object(trainer.criterion, "forward", side_effect=check_loss):
                    with patch.object(trainer.optimizer, "step", side_effect=check_step):
                        losses = trainer.train(rows)
                final = evaluate_batch(dpr, rows)
                print(f"Fixed FIQA batch after training:  {final}")

                self.assertEqual(len(gradients), EPOCHS)
                self.assertEqual(len(losses), EPOCHS)
                self.assertLess(final["loss"], initial["loss"] * 0.5)
                self.assertGreater(final["positive"] - final["negative"],
                                   initial["positive"] - initial["negative"])
                self.assertGreaterEqual(final["top1"], 0.75)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
