import json
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    """Persistent exact inner-product index with document metadata."""

    def __init__(self, dimension, metadata):
        if dimension < 1:
            raise ValueError("Embedding dimension must be positive.")
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []
        self.metadata = metadata

    def build(self, documents, embeddings):
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if self.index.ntotal:
            raise ValueError("The index has already been built.")
        if vectors.ndim != 2 or vectors.shape != (len(documents), self.index.d):
            raise ValueError("Embedding shape must match the documents and index dimension.")
        if not len(documents) or not np.isfinite(vectors).all():
            raise ValueError("Documents and finite embeddings are required.")
        ids = [doc["id"] for doc in documents]
        if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("Document IDs must be unique nonempty strings.")
        if any(not isinstance(doc["text"], str) for doc in documents):
            raise ValueError("Document text must be a string.")
        self.documents = [{"id": doc["id"], "text": doc["text"]} for doc in documents]
        self.index.add(vectors)

    def save(self, directory):
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Index directory is not empty: {directory}")
        if not self.index.ntotal:
            raise ValueError("Cannot save an empty index.")
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with (directory / "documents.json").open("w", encoding="utf-8") as file:
            json.dump(self.documents, file, ensure_ascii=False)
        with (directory / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump({"dimension": self.index.d, "metric": "inner_product", **self.metadata}, file, indent=2)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with (directory / "metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
        with (directory / "documents.json").open(encoding="utf-8") as file:
            documents = json.load(file)
        index = faiss.read_index(str(directory / "index.faiss"))
        if (metadata["metric"] != "inner_product" or index.metric_type != faiss.METRIC_INNER_PRODUCT
                or index.d != metadata["dimension"] or index.ntotal != len(documents)):
            raise ValueError("Index and metadata do not agree.")
        ids = [doc["id"] for doc in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("Stored document IDs are not unique.")
        store = cls(index.d, metadata)
        store.index = index
        store.documents = documents
        return store

    def search(self, embeddings, k):
        if k < 1 or not self.index.ntotal:
            raise ValueError("Search requires a positive k and a nonempty index.")
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.index.d or not np.isfinite(vectors).all():
            raise ValueError("Query embeddings must be finite and match the index dimension.")
        scores, positions = self.index.search(vectors, min(k, self.index.ntotal))
        return [[{**self.documents[int(position)], "score": float(score)}
                 for position, score in zip(row_positions, row_scores)]
                for row_positions, row_scores in zip(positions, scores)]
