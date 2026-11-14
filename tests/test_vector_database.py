"""Tests that also explain the vector database. Run with `python -m tests.test_vector_database`.

Mental model:
    VectorDatabase = a matrix `vectors` (N x d, one row per passage) + a list `ids` (row i <-> ids[i]).
    query(q, k)    = scores = q @ vectors.T  (inner product with every row) -> top-k rows -> their ids.
    save(folder)   = vectors.npy (the matrix) + ids.json (the row labels) + config.json (settings).
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data_utils.document import Document
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.vector_database import DenseRetriever, VectorDatabase, build_database

np.set_printoptions(precision=2, suppress=True)
OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")


def show(title):
    print(f"\n--- {title} ---")


def test_insert_stores_rows_and_ids():
    show("insert: each id gets one row in the matrix")
    db = VectorDatabase(dim=3)
    db.insert(["a", "b"], np.array([[1, 0, 0], [0, 1, 0]]))
    db.insert(["c"], np.array([[0, 0, 1]]))
    print("ids    :", db.ids)
    print("vectors:\n", db.vectors)
    assert db.ids == ["a", "b", "c"]
    assert db.vectors.shape == (3, 3) and len(db) == 3
    assert np.array_equal(db.vectors[2], [0, 0, 1])  # row 2 belongs to ids[2] == "c"

    # inserting an existing id replaces its vector instead of adding a duplicate
    db.insert(["a"], np.array([[5, 5, 5]]))
    assert len(db) == 3 and np.array_equal(db.vectors[db.ids.index("a")], [5, 5, 5])

    # bad input is rejected
    for bad_ids, bad_vecs in [(["x"], np.zeros((1, 4))), (["x", "y"], np.zeros((1, 3))), (["x", "x"], np.zeros((2, 3)))]:
        try:
            db.insert(bad_ids, bad_vecs)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_query_is_inner_product_top_k():
    show("query: score = inner product, return the k best ids, best first")
    db = VectorDatabase(dim=2)
    db.insert(["east", "north", "north-east", "west"],
              np.array([[1, 0], [0, 1], [1, 1], [-1, 0]]))
    q = np.array([[2.0, 1.0]])
    ids, scores = db.query(q, k=3)
    print("query vector:", q[0])
    print("all scores  :", dict(zip(db.ids, (q @ db.vectors.T)[0])))
    print("top-3       :", list(zip(ids[0], scores[0])))
    # 2*1+1*0=2 (east), 2*1+1*1=3 (north-east), 2*0+1*1=1 (north), -2 (west)
    assert ids == [["north-east", "east", "north"]]
    assert np.allclose(scores, [[3, 2, 1]])

    # several queries at once: one row of results per query
    ids, _ = db.query(np.array([[1.0, 0.0], [0.0, 1.0]]), k=1)
    assert ids == [["east"], ["north"]]
    # asking for more than exist returns everything, and an empty db is an error
    assert len(db.query(q, k=100)[0][0]) == 4
    try:
        VectorDatabase(2).query(q, 1)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_delete_removes_rows():
    show("delete: removes the id and its row")
    db = VectorDatabase(dim=2)
    db.insert(["a", "b", "c"], np.eye(3, 2))
    db.delete(["b"])
    print("ids after delete:", db.ids, "| matrix shape:", db.vectors.shape)
    assert db.ids == ["a", "c"] and db.vectors.shape == (2, 2)
    db.delete(["missing"])  # unknown ids are ignored
    assert len(db) == 2


def test_save_writes_three_files_and_load_restores():
    show("save/load: what is stored on disk")
    db = VectorDatabase(dim=4, encoder_info={"d_model": 4, "pooling": "mean"})
    rng = np.random.default_rng(0)
    db.insert(["p0", "p1", "p2"], rng.normal(size=(3, 4)))
    # Saved to a visible folder so you can open the files and inspect them after the run.
    folder = os.path.join(OUTPUT_DIR, "my_db")
    shutil.rmtree(folder, ignore_errors=True)
    db.save(folder)

    print("folder contents:", sorted(os.listdir(folder)))
    vectors = np.load(os.path.join(folder, "vectors.npy"))
    with open(os.path.join(folder, "ids.json")) as f:
        ids = json.load(f)
    with open(os.path.join(folder, "config.json")) as f:
        config = json.load(f)
    print("vectors.npy shape:", vectors.shape, "(N passages x d)")
    print("ids.json         :", ids, "(row i of vectors.npy <-> ids[i])")
    print("config.json      :", config)
    assert sorted(os.listdir(folder)) == ["config.json", "ids.json", "vectors.npy"]
    assert vectors.shape == (3, 4) and ids == ["p0", "p1", "p2"]
    assert config == {"dim": 4, "index": "flat", "metric": "inner_product",
                      "encoder": {"d_model": 4, "pooling": "mean"}}

    loaded = VectorDatabase.load(folder)
    assert loaded.ids == db.ids and np.array_equal(loaded.vectors, db.vectors)
    assert loaded.encoder_info == db.encoder_info
    q = rng.normal(size=(1, 4))
    assert loaded.query(q, 2)[0] == db.query(q, 2)[0]  # identical answers after reload


def test_build_database_encodes_passages_with_the_encoder():
    show("build_database: token ids -> TransformerEncoder -> one vector per passage")
    torch.manual_seed(0)
    enc = TransformerEncoder(vocab_size=50, d_model=16, n_heads=2, n_layers=1, feed_forward_dim=32, max_len=8)
    enc.train()  # build_database must switch to eval (no dropout) and restore this afterwards
    passages = torch.tensor([[1, 5, 6, 0], [1, 7, 8, 9], [1, 20, 21, 0]])  # 0 = padding
    db = build_database(enc, passages, ["p0", "p1", "p2"], encoder_info={"d_model": 16})
    print("passage token ids:\n", passages.numpy())
    print("stored vectors shape:", db.vectors.shape, "-> 3 passages x 16 dims")
    assert db.vectors.shape == (3, 16) and db.ids == ["p0", "p1", "p2"]
    assert enc.training  # training mode restored

    enc.eval()
    with torch.no_grad():
        expected = enc(passages).numpy()
    assert np.allclose(db.vectors, expected, atol=1e-6)  # stored vectors == the encoder's pooled output


def test_dense_retriever_uses_query_encoder_and_returns_documents():
    show("DenseRetriever: query text -> tokens -> query encoder -> search -> Documents")
    torch.manual_seed(0)
    make = lambda: TransformerEncoder(vocab_size=50, d_model=16, n_heads=2, n_layers=1,
                                      feed_forward_dim=32, max_len=8)
    doc_enc, query_enc = make(), make()  # two different random encoders, as in DPR
    passages = torch.tensor([[1, 5, 6, 0], [1, 7, 8, 9], [1, 20, 21, 0]])
    ids = ["p0", "p1", "p2"]
    db = build_database(doc_enc, passages, ids)

    seen = []

    def tokenize(texts):  # stand-in tokenizer: the "text" is the index of a passage's token row
        seen.extend(texts)
        return passages[[int(t) for t in texts]]

    documents = {i: Document(text=f"text of {i}", _id=i) for i in ids}
    retriever = DenseRetriever(db, query_enc, tokenize, documents)

    qvec = retriever.encode_queries(["1"])
    print("query vector shape:", tuple(qvec.shape))
    found_ids, scores = retriever.Search(["0", "1", "2"], k=2)
    print("Search ids   :", found_ids)
    print("Search scores:\n", scores)
    assert seen[0] == "1" and qvec.shape == (1, 16)
    assert len(found_ids) == 3 and all(len(r) == 2 for r in found_ids)
    assert np.all(scores[:, :-1] >= scores[:, 1:])  # best first

    # the query side must go through the query encoder, not the document encoder
    with torch.no_grad():
        query_enc.eval()
        assert np.allclose(qvec.numpy(), query_enc(passages[[1]]).numpy(), atol=1e-6)
        assert not np.allclose(qvec.numpy(), doc_enc.eval()(passages[[1]]).numpy(), atol=1e-3)

    docs = retriever.Lookup(["0"], k=2)[0]
    print("Lookup       :", [(d.GetId(), d.GetText()) for d in docs])
    assert docs[0].GetText() == f"text of {docs[0].GetId()}"

    # without a documents mapping (e.g. db loaded from disk) Lookup still returns ids, with empty text
    bare = DenseRetriever(db, query_enc, tokenize)
    assert bare.Lookup(["0"], k=1)[0][0].GetText() == ""


def test_same_encoder_passage_finds_itself():
    show("sanity: with one shared encoder, a passage used as a query ranks itself first")
    torch.manual_seed(0)
    enc = TransformerEncoder(vocab_size=50, d_model=16, n_heads=2, n_layers=1, feed_forward_dim=32, max_len=8)
    passages = torch.tensor([[1, 5, 6, 0], [1, 7, 8, 9], [1, 20, 21, 0]])
    db = build_database(enc, passages,     ["p0", "p1", "p2"])
    retriever = DenseRetriever(db, enc, lambda t: passages[[int(x) for x in t]])
    top = retriever.Search(["0", "1", "2"], k=1)[0]
    print("top hit per query:", top)
    assert top == [["p0"], ["p1"], ["p2"]]


if __name__ == "__main__":
    test_insert_stores_rows_and_ids()
    test_query_is_inner_product_top_k()
    test_delete_removes_rows()
    test_save_writes_three_files_and_load_restores()
    test_build_database_encodes_passages_with_the_encoder()
    test_dense_retriever_uses_query_encoder_and_returns_documents()
    test_same_encoder_passage_finds_itself()
    print("\nvector database tests passed")
