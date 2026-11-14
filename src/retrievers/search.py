import argparse

from src.retrievers.checkpoint import load_checkpoint
from src.retrievers.vector_database import encode_token_ids
from src.retrievers.vector_store import FaissVectorStore


def search(query, checkpoint, index, k=10, device="auto"):
    if not query.strip():
        raise ValueError("Query must not be empty.")
    store = FaissVectorStore.load(index)
    dpr, _, hashes = load_checkpoint(checkpoint, device)
    if hashes != store.metadata["checkpoint_hashes"]:
        raise ValueError("Checkpoint does not match the indexed encoders; rebuild the index.")
    if (dpr.cfg.model.tokenizer_name != store.metadata["tokenizer"]
            or dpr.cfg.model.max_len != store.metadata["max_len"]
            or dpr.cfg.model.pooling_type != store.metadata["pooling_type"]):
        raise ValueError("Checkpoint tokenizer or encoder settings do not match the index.")
    embeddings = encode_token_ids(dpr.query_encoder, dpr.tokenize([query])).cpu().numpy()
    return store.search(embeddings, k)[0]


def main():
    parser = argparse.ArgumentParser(description="Search a persistent DPR FAISS index.")
    parser.add_argument("query")
    parser.add_argument("--checkpoint", default="outputs/dpr/checkpoints/best")
    parser.add_argument("--index", default="outputs/dpr/vector_store")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    for result in search(args.query, args.checkpoint, args.index, args.k, args.device):
        print(f'{result["id"]}  score={result["score"]:.4f}\n{result["text"]}\n')


if __name__ == "__main__":
    main()
