import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.retrievers.dpr import DPR


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(directory, device="auto"):
    directory = Path(directory)
    with (directory / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    cfg = OmegaConf.create(metadata["config"])
    cfg.device = device
    cfg.model.query_encoder_weights_path = str(directory / "query_encoder.pt")
    cfg.model.document_encoder_weights_path = str(directory / "document_encoder.pt")
    hashes = {
        "query": checkpoint_hash(cfg.model.query_encoder_weights_path),
        "document": checkpoint_hash(cfg.model.document_encoder_weights_path),
    }
    return DPR(cfg), metadata, hashes
