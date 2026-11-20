import argparse

import torch

from src.data_utils.corpus import Corpus
from src.data_utils.dataset import DocumentDataset
from src.retrievers.checkpoint import load_checkpoint
from src.retrievers.vector_database import encode_token_ids
from src.retrievers.vector_store import FaissVectorStore


def build_index(checkpoint, output, device="auto"):
    dpr, checkpoint_metadata, hashes = load_checkpoint(checkpoint, device)
    cfg = dpr.cfg
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f"{cfg.base_path}/retrieval_data")
    if checkpoint_metadata.get("corpus_version") != getattr(corpus, "version", None):
        raise ValueError("Checkpoint corpus version does not match the index corpus.")
    passages = DocumentDataset(corpus.all_documents).GetDocuments()
    if not passages:
        raise ValueError("Cannot index an empty document collection.")
    vectors = []
    for start in range(0, len(passages), dpr.batch_size):
        batch = passages[start:start + dpr.batch_size]
        tokens = dpr.tokenize([doc.GetText() for doc in batch])
        vectors.append(encode_token_ids(dpr.document_encoder, tokens, dpr.batch_size).cpu())
        print(f"Encoded {min(start + len(batch), len(passages))}/{len(passages)} passages", flush=True)
    documents = [{"id": doc.GetId(), "text": doc.GetText()} for doc in passages]
    store = FaissVectorStore(cfg.model.d_model, {
        "checkpoint_hashes": hashes, "checkpoint_epoch": checkpoint_metadata["epoch"],
        "pooling_type": cfg.model.pooling_type, "tokenizer": cfg.model.tokenizer_name,
        "tokenizer_revision": cfg.model.get("tokenizer_revision"),
        "max_len": cfg.model.max_len, "dataset": cfg.dataset.name,
        "corpus_version": getattr(corpus, "version", None),
    })
    store.build(documents, torch.cat(vectors).numpy())
    store.save(output)
    print(f"Saved {len(documents)} passages to {output}")


def main():
    parser = argparse.ArgumentParser(description="Build a persistent DPR FAISS index.")
    parser.add_argument("--checkpoint", default="outputs/dpr/checkpoints/best")
    parser.add_argument("--output", default="outputs/dpr/vector_store")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    build_index(args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
