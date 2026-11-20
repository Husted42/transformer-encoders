import hashlib
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from src.data_utils.download_retrieval_datasets import get_fiqa_dataset, get_squad_dataset


class Corpus:
    """A shared passage collection and split-specific query relevance judgments."""

    def __init__(self, dataset_name: str, base_dir: str):
        if dataset_name not in {"fiqa", "squad"}:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.dataset_name = dataset_name
        self.version = f"{dataset_name}-retrieval-v1"
        path = Path(base_dir) / self.version
        if path.exists():
            self.ds = load_from_disk(str(path))
        else:
            data = (self._fiqa(get_fiqa_dataset(str(Path(base_dir) / "downloads")))
                    if dataset_name == "fiqa"
                    else self._squad(get_squad_dataset(str(Path(base_dir) / "downloads"))))
            self.ds = DatasetDict(data)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.ds.save_to_disk(str(path))
        self._validate()

    @staticmethod
    def _fiqa(data):
        corpus, queries, qrels = data
        documents = {}
        for row in corpus:
            doc_id = str(row["_id"])
            if doc_id in documents:
                raise ValueError(f"Duplicate corpus ID {doc_id}.")
            text = "\n".join(part.strip() for part in (row["title"], row["text"]) if part and part.strip())
            documents[doc_id] = text
        query_texts = {}
        for row in queries:
            query_id = str(row["_id"])
            if query_id in query_texts:
                raise ValueError(f"Duplicate query ID {query_id}.")
            query_texts[query_id] = row["text"]
        result = {"corpus": Dataset.from_dict({
            "text": [[text] for text in documents.values()],
            "document_id": [[doc_id] for doc_id in documents],
        })}
        for split, labels in qrels.items():
            positives = {}
            for label in labels:
                query_id, doc_id = str(label["query-id"]), str(label["corpus-id"])
                if query_id not in query_texts or doc_id not in documents:
                    raise ValueError(f"Dangling qrel: query {query_id}, document {doc_id}.")
                if label["score"] > 0:
                    positives.setdefault(query_id, set()).add(doc_id)
            rows = []
            for query_id, ids in positives.items():
                ids = sorted(ids)
                rows.append({"query_id": query_id, "question": query_texts[query_id],
                             "document_id": ids, "text": [documents[doc_id] for doc_id in ids]})
            result[split] = Dataset.from_list(rows)
        return result

    @staticmethod
    def _squad(data):
        documents = {}
        result = {}
        for split, examples in data.items():
            rows = []
            for row in examples:
                text = row["context"]
                doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if doc_id in documents and documents[doc_id] != text:
                    raise ValueError("SQuAD context hash collision.")
                documents[doc_id] = text
                rows.append({"query_id": f"{split}:{row['id']}", "question": row["question"],
                             "document_id": [doc_id], "text": [text]})
            result[split] = Dataset.from_list(rows)
        result["corpus"] = Dataset.from_dict({
            "text": [[text] for text in documents.values()],
            "document_id": [[doc_id] for doc_id in documents],
        })
        return result

    def _validate(self):
        if "corpus" not in self.ds or "train" not in self.ds or "validation" not in self.ds:
            raise ValueError(f"Invalid retrieval cache {self.version}: missing corpus or splits.")
        documents = {}
        for row in self.ds["corpus"]:
            if len(row["document_id"]) != 1 or len(row["text"]) != 1:
                raise ValueError("Corpus entries must have one ID and one text.")
            doc_id, text = row["document_id"][0], row["text"][0]
            if not doc_id or not isinstance(text, str) or doc_id in documents:
                raise ValueError(f"Invalid or duplicate document ID: {doc_id}.")
            documents[doc_id] = text
        self.relevant_ids = {}
        for split in ("train", "validation", "test"):
            if split not in self.ds:
                continue
            for row in self.ds[split]:
                query_id, ids = row["query_id"], row["document_id"]
                if query_id in self.relevant_ids or not row["question"].strip() or not ids:
                    raise ValueError(f"Invalid or duplicate query ID: {query_id}.")
                if len(ids) != len(set(ids)) or len(ids) != len(row["text"]):
                    raise ValueError(f"Invalid qrels for query {query_id}.")
                if any(documents.get(doc_id) != text for doc_id, text in zip(ids, row["text"])):
                    raise ValueError(f"Dangling or inconsistent qrel for query {query_id}.")
                self.relevant_ids[query_id] = ids

    @property
    def train(self):
        return self.ds.get("train")

    @property
    def val(self):
        return self.ds.get("validation")

    @property
    def test(self):
        return self.ds.get("test")

    @property
    def all_documents(self):
        return self.ds["corpus"]
