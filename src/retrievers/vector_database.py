
import json
import os
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from src.data_utils.document import Document

# NOTE: I uesd the implementaion from https://www.pinecone.io/learn/vector-database/

class FlatIndex:
    """
    To find similar passages, we need to compare their embeddings.
    For simplicity, we use a flat index that compares the query with
    every stored vector.
    """

    def search(self, stored: np.ndarray, queries: np.ndarray, k: int):
        # Compare each query embedding with all the embeddings in our database.
        # We use the inner product as our similarity measure, as in DPR.
        scores = queries @ stored.T

        # We only want the k most similar passages, sorted by their scores.
        k = min(k, stored.shape[0])
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)

        # This will return the indices of the k highest-scoring passages for each query.
        return (
            np.take_along_axis(part, order, axis=1),
            np.take_along_axis(part_scores, order, axis=1)
        )


class VectorDatabase:
    """
    For each passage, we store its ID and the corresponding embedding.
    The vectors are saved in vectors.npy, while ids.json keeps track
    of which passage each vector belongs to. We also save the
    configuration so we can load the database again later.
    """

    # __init__ takes the dimensionality of the vectors and optional encoder information.
    def __init__(self, dim: int, encoder_info: Optional[dict] = None) -> None:
        self.dim = dim
        self.index = FlatIndex()
        self.encoder_info = encoder_info or {}
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.ids: list = []

    def __len__(self) -> int:
        return len(self.ids)

    def _check(self, vectors) -> np.ndarray:
        # All embeddings must have the same number of dimensions.
        # We also convert PyTorch tensors to NumPy before storing them.
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of shape (N, {self.dim}), got {vectors.shape}.")

        return vectors

    def insert(self, ids: Sequence, vectors) -> None:
        """
        We want to be able to add new passages or update existing ones.
        If an ID already exists, we remove the old embedding before
        inserting the new one.
        """
        vectors = self._check(vectors)

        if len(ids) != len(vectors) or len(set(ids)) != len(ids):
            raise ValueError("ids must be unique and match the number of vectors.")

        self.delete([i for i in ids if i in set(self.ids)])

        self.vectors = np.concatenate([self.vectors, vectors])
        self.ids.extend(ids)

    def delete(self, ids: Sequence) -> None:
        # Remove both the IDs and their embeddings so they stay aligned.
        drop = set(ids)
        keep = [n for n, i in enumerate(self.ids) if i not in drop]

        self.vectors = self.vectors[keep]
        self.ids = [self.ids[n] for n in keep]

    def query(self, vectors, k: int) -> tuple[list[list], np.ndarray]:
        """
        Search for the k nearest embeddings and return their passage IDs
        together with their similarity scores.
        """
        if len(self) == 0:
            raise RuntimeError("The vector database is empty.")

        idx, scores = self.index.search(self.vectors, self._check(vectors), k)

        return [[self.ids[j] for j in row] for row in idx], scores

    def save(self, directory: str) -> None:
        # Save the embeddings and IDs separately, along with the settings
        # needed to reconstruct the database.
        os.makedirs(directory, exist_ok=True)

        np.save(os.path.join(directory, "vectors.npy"), self.vectors)

        with open(os.path.join(directory, "ids.json"), "w", encoding="utf-8") as f:
            json.dump(self.ids, f)

        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "dim": self.dim,
                "index": "flat",
                "metric": "inner_product",
                "encoder": self.encoder_info
            }, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "VectorDatabase":
        # Load the saved vectors, IDs and configuration so we can
        # continue searching without having to encode everything again.
        with open(os.path.join(directory, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)

        db = cls(cfg["dim"], cfg.get("encoder"))
        db.vectors = np.load(os.path.join(directory, "vectors.npy"))

        with open(os.path.join(directory, "ids.json"), encoding="utf-8") as f:
            db.ids = json.load(f)

        return db


@torch.no_grad()
def encode_token_ids(
    encoder: torch.nn.Module,
    token_ids: torch.Tensor,
    batch_size: int = 64
) -> torch.Tensor:
    """
    Before we can store or search passages, we need to represent them
    as embeddings. We use our Transformer encoder to do this in batches.
    """
    was_training = encoder.training
    encoder.eval()

    out = [
        encoder(token_ids[i:i + batch_size])
        for i in range(0, len(token_ids), batch_size)
    ]

    encoder.train(was_training)
    return torch.cat(out)


def build_database(
    encoder: torch.nn.Module,
    token_ids: torch.Tensor,
    ids: Sequence,
    batch_size: int = 64,
    encoder_info: Optional[dict] = None
) -> VectorDatabase:
    """
    First we encode all our passages into vectors.
    We then create a database and insert each embedding together
    with the ID of the original passage.
    """
    vectors = encode_token_ids(encoder, token_ids, batch_size)

    db = VectorDatabase(vectors.shape[1], encoder_info)
    db.insert(list(ids), vectors)

    return db


class DenseRetriever:
    """
    To search the database, we first encode the query into the same
    vector space as our passages. We can then compare the embeddings
    and retrieve the passages with the highest similarity.

    We also keep the original documents so we can return the actual
    passages instead of just their IDs.
    """

    def __init__(
        self,
        db: VectorDatabase,
        query_encoder: torch.nn.Module,
        tokenize: Callable[[list[str]], torch.Tensor],
        documents: Optional[dict[str, Document]] = None
    ) -> None:
        self.db = db
        self.query_encoder = query_encoder
        self.tokenize = tokenize
        self.documents = documents or {}

    def encode_queries(self, queries: list[str]) -> torch.Tensor:
        # Convert the queries into embeddings that we can compare
        # with the vectors already stored in our database.
        return encode_token_ids(self.query_encoder, self.tokenize(queries))

    def Search(self, queries: list[str], k: int):
        # Find the k most similar passages for each query.
        return self.db.query(self.encode_queries(queries), k)

    def Lookup(self, queries: list[str], k: int) -> list:
        # Once we have the nearest IDs, we use them to retrieve
        # the original passages rather than just returning vectors.
        ids, _ = self.Search(queries, k)

        return [
            [self.documents.get(i, Document(text="", _id=i)) for i in row]
            for row in ids
        ]
