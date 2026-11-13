import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.retrievers.dpr import DPR


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


if __name__ == "__main__":
    unittest.main()
