"""Toy DPR to test the vector database: `python -m tests.demo_dpr`.

NOTE : I build this using claude to test if the Database works, 

Synthetic corpus: each topic owns a set of words. A passage mixes words of one topic with
shared filler words; a query is a few words of a topic. Two TransformerEncoders (query and
document) are trained with the in-batch softmax loss, passages are stored in a VectorDatabase
(saved to disk and reloaded), and retrieval is evaluated on held-out passages.
"""

import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.data_utils.document import Document
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.vector_database import DenseRetriever, VectorDatabase, build_database

random.seed(0)
torch.manual_seed(0)

PAD, CLS = 0, 1
N_TOPICS, WORDS_PER_TOPIC, N_FILLER = 20, 15, 10
PASSAGE_LEN, QUERY_LEN, MAX_LEN = 8, 3, 12

topics = [[f"t{t}w{i}" for i in range(WORDS_PER_TOPIC)] for t in range(N_TOPICS)]
filler = [f"filler{i}" for i in range(N_FILLER)]
vocab = {w: i + 2 for i, w in enumerate([w for ws in topics for w in ws] + filler)}


def tokenize(texts):
    batch = torch.full((len(texts), MAX_LEN), PAD)
    for r, text in enumerate(texts):
        ids = [CLS] + [vocab[w] for w in text.split()]
        batch[r, :len(ids)] = torch.tensor(ids)
    return batch


def make_passage(t):
    words = random.sample(topics[t], PASSAGE_LEN - 2) + random.sample(filler, 2)
    random.shuffle(words)
    return " ".join(words)


def make_query(t):
    return " ".join(random.sample(topics[t], QUERY_LEN))


def make_encoder():
    return TransformerEncoder(vocab_size=len(vocab) + 2, d_model=64, n_heads=4, n_layers=2,
                              feed_forward_dim=128, max_len=MAX_LEN, pooling_type="mean")


# ---- 1. train the two encoders with in-batch contrastive loss ----
query_enc, doc_enc = make_encoder(), make_encoder()
opt = torch.optim.Adam(list(query_enc.parameters()) + list(doc_enc.parameters()), lr=1e-3)
print("=== 1. Training (in-batch softmax loss, one positive per topic per batch) ===")
for step in range(301):
    # one (query, passage) pair per topic, so every other passage in the batch is a true negative
    q = tokenize([make_query(t) for t in range(N_TOPICS)])
    d = tokenize([make_passage(t) for t in range(N_TOPICS)])
    scores = query_enc(q) @ doc_enc(d).T                      # (B, B) inner products
    loss = F.cross_entropy(scores, torch.arange(N_TOPICS))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

# ---- 2. build the vector database from fresh (held-out) passages ----
print("\n=== 2. Build, save and reload the vector database ===")
PER_TOPIC = 10
passage_topic = [t for t in range(N_TOPICS) for _ in range(PER_TOPIC)]
passage_text = [make_passage(t) for t in passage_topic]
passage_ids = [f"p{i}" for i in range(len(passage_text))]
db = build_database(doc_enc, tokenize(passage_text), passage_ids,
                    encoder_info={"d_model": 64, "pooling": "mean"})
folder = os.path.join(tempfile.mkdtemp(), "toy_db")
db.save(folder)
print(f"saved {len(db)} vectors of dim {db.dim} to {folder}: {sorted(os.listdir(folder))}")
db = VectorDatabase.load(folder)
print(f"reloaded {len(db)} vectors")

documents = {i: Document(text=t, _id=i) for i, t in zip(passage_ids, passage_text)}
retriever = DenseRetriever(db, query_enc, tokenize, documents)

# ---- 3. evaluate: a retrieved passage is relevant if it has the query's topic ----
print("\n=== 3. Evaluation on 200 held-out queries ===")
K = 10
eval_topics = [random.randrange(N_TOPICS) for _ in range(200)]
ids_per_query, _ = retriever.Search([make_query(t) for t in eval_topics], K)
topic_of = dict(zip(passage_ids, passage_topic))
rel = [[topic_of[i] == t for i in row] for row, t in zip(ids_per_query, eval_topics)]
precision = sum(sum(r) / K for r in rel) / len(rel)
recall = sum(sum(r) / min(K, PER_TOPIC) for r in rel) / len(rel)
mrr = sum(next((1 / (n + 1) for n, x in enumerate(r) if x), 0) for r in rel) / len(rel)
print(f"Precision@{K} {precision:.3f}   Recall@{K} {recall:.3f}   MRR {mrr:.3f}")
print(f"(random guessing would give Precision@{K} ~ {1 / N_TOPICS:.3f})")

# ---- 4. interactive-style example ----
print("\n=== 4. Example query ===")
t = 7
query = make_query(t)
print("query:", query, f"(topic {t})")
for doc in retriever.Lookup([query], 3)[0]:
    print(f"  {doc.GetId():>5} (topic {topic_of[doc.GetId()]}): {doc.GetText()}")
