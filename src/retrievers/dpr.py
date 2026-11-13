import torch

from src.retrievers.models.encoder import TransformerEncoder


class DPR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.__load_models()

    def __load_models(self):
        model = self.cfg.model
        encoder_args = {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "n_layers": model.n_layers,
            "feed_forward_dim": model.feed_forward_dim,
            "max_len": model.max_len,
            "padding_idx": model.get("padding_idx", 0),
            "dropout": model.get("dropout", 0.1),
            "pooling_type": model.pooling_type,
        }
        self.query_encoder = TransformerEncoder(**encoder_args)
        self.document_encoder = TransformerEncoder(**encoder_args)

        query_weights_path = model.get("query_encoder_weights_path")
        if query_weights_path:
            self.query_encoder.load_state_dict(
                torch.load(query_weights_path, map_location="cpu", weights_only=True)
            )

        document_weights_path = model.get("document_encoder_weights_path")
        if document_weights_path:
            self.document_encoder.load_state_dict(
                torch.load(document_weights_path, map_location="cpu", weights_only=True)
            )
