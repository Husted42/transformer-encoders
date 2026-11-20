import warnings

import numpy as np
import torch
from transformers import AutoTokenizer

from src.data_utils.dataset import DocumentDataset
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.vector_database import DenseRetriever, VectorDatabase, encode_token_ids


class DPR:
    def __init__(self, cfg, documents=None):
        self.cfg = cfg
        self.batch_size = cfg.get("retriever", {}).get("batch_size", 32)
        if self.batch_size < 1:
            raise ValueError("DPR batch_size must be positive.")
        device = cfg.get("device", "cpu")
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        tokenizer_name = cfg.model.get("tokenizer_name")
        self.tokenizer = (AutoTokenizer.from_pretrained(
            tokenizer_name, revision=cfg.model.get("tokenizer_revision")
        ) if tokenizer_name else None)
        self.__load_models()
        self.retriever = None
        if documents is not None:
            self.build_index(documents)

    def __load_models(self):
        model = self.cfg.model
        vocab_size = model.get("vocab_size")
        padding_idx = model.get("padding_idx")
        if self.tokenizer is not None:
            if self.tokenizer.pad_token_id is None:
                raise ValueError("The DPR tokenizer must define a padding token.")
            if model.pooling_type == "cls" and self.tokenizer.cls_token_id is None:
                raise ValueError("CLS pooling requires a tokenizer with a CLS token.")
            if vocab_size is not None and vocab_size != len(self.tokenizer):
                raise ValueError("vocab_size must match the tokenizer vocabulary size.")
            if padding_idx is not None and padding_idx != self.tokenizer.pad_token_id:
                raise ValueError("padding_idx must match the tokenizer padding token ID.")
            vocab_size = len(self.tokenizer)
            padding_idx = self.tokenizer.pad_token_id
        if vocab_size is None:
            raise ValueError("Set vocab_size or configure a tokenizer.")
        if padding_idx is None:
            padding_idx = 0
        encoder_args = {
            "vocab_size": vocab_size,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "n_layers": model.n_layers,
            "feed_forward_dim": model.feed_forward_dim,
            "max_len": model.max_len,
            "padding_idx": padding_idx,
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
        self.encoders_trained = bool(query_weights_path and document_weights_path)
        self.query_encoder.to(self.device)
        self.document_encoder.to(self.device)

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("Configure model.tokenizer_name before indexing or querying.")
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.model.max_len,
            return_tensors="pt",
        )["input_ids"]
        if self.cfg.model.pooling_type == "cls" and not (
            tokens[:, 0] == self.tokenizer.cls_token_id
        ).all():
            raise ValueError("CLS pooling requires a CLS token at the start of every sequence.")
        return tokens.to(self.device)

    def build_index(self, documents):
        """Encode corpus entries and prepare the dense retriever."""
        passages = DocumentDataset(documents).GetDocuments()
        if not passages:
            raise ValueError("Cannot build a DPR index without documents.")
        ids = [document.GetId() for document in passages]
        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique.")
        if not self.encoders_trained:
            warnings.warn(
                "DPR is using at least one randomly initialized encoder; "
                "retrieval rankings will not be meaningful until the encoders are trained.",
                UserWarning,
                stacklevel=2,
            )

        vectors = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            token_ids = self.tokenize([document.GetText() for document in batch])
            vectors.append(encode_token_ids(self.document_encoder, token_ids, self.batch_size).cpu())

        db = VectorDatabase(self.cfg.model.d_model)
        db.insert(ids, torch.cat(vectors))
        self.retriever = DenseRetriever(
            db, self.query_encoder, self.tokenize,
            {document.GetId(): document for document in passages},
        )

    def Search(self, queries: list[str], k: int):
        if self.retriever is None:
            raise RuntimeError("Build a DPR index before querying.")
        if k < 1:
            raise ValueError("k must be positive.")
        if not queries:
            return [], np.empty((0, min(k, len(self.retriever.db))), dtype=np.float32)

        ids, scores = [], []
        for start in range(0, len(queries), self.batch_size):
            batch_ids, batch_scores = self.retriever.Search(queries[start:start + self.batch_size], k)
            ids.extend(batch_ids)
            scores.append(batch_scores)
        return ids, np.concatenate(scores, axis=0)

    def Lookup(self, queries: list[str], k: int) -> list:
        ids, _ = self.Search(queries, k)
        return [[self.retriever.documents[document_id] for document_id in row] for row in ids]
