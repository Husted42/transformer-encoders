"""Demo of the transformer encoder: run with `python tests/demo_encoder.py`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.retrievers.models.encoder import (
    MultiHeadSelfAttention,
    PositionalEncoding,
    ScaledDotProductAttention,
    TokenEmbedding,
    TransformerEncoder,
)

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)

PAD = 0


def header(title):
    print(f"\n=== {title} ===")


header("1. Scaled dot-product attention on a tiny example")
q = k = torch.tensor([[[1., 0.], [0., 1.]]])
v = torch.tensor([[[2., 0.], [0., 6.]]])
out, weights = ScaledDotProductAttention()(q, k, v)
print("weights (each row sums to 1):\n", weights[0])
print("output (weighted mix of values):\n", out[0])

header("2. Sinusoidal positional encoding makes order visible")
emb = TokenEmbedding(50, 8, padding_idx=PAD)
pe = PositionalEncoding(8, max_len=16)
same_token = emb(torch.full((1, 4), 7))
print("Same token at 4 positions, identical before encoding:",
      bool(torch.allclose(same_token[0, 0], same_token[0, 3])))
print("...and distinct after encoding:",
      not torch.allclose(pe(same_token)[0, 0], pe(same_token)[0, 3]))

header("3. Padding mask: padded positions get zero attention")
ids = torch.tensor([[5, 9, 3, PAD, PAD]])
mask = (ids != PAD).unsqueeze(1)
x = pe(emb(ids))
_, w = MultiHeadSelfAttention(8, 2)(x, mask)
print("attention from token 0 (head 0) over positions:", w[0, 0, 0])
print("total weight on padding:", w[0, :, :, 3:].sum().item())

header("4. Full encoder: shapes and pooling modes")
batch = torch.tensor([[1, 12, 7, 20, 4, 2],
                      [1, 33, 8, PAD, PAD, PAD]])  # token 1 acts as CLS
for pooling in ("mean", "max", "cls"):
    enc = TransformerEncoder(vocab_size=50, d_model=16, n_heads=4, n_layers=2,
                             feed_forward_dim=32, max_len=16, pooling_type=pooling).eval()
    tokens = enc(batch, return_token_embeddings=True)
    pooled = enc(batch)
    print(f"{pooling:>4}: tokens {tuple(tokens.shape)} -> pooled {tuple(pooled.shape)}")

header("5. Padding does not change the sequence vector")
enc = TransformerEncoder(vocab_size=50, d_model=16, n_heads=4, n_layers=2,
                         feed_forward_dim=32, max_len=16).eval()
padded = enc(batch)[1]
trimmed = enc(batch[1:2, :3])[0]
print("padded == trimmed:", bool(torch.allclose(padded, trimmed, atol=1e-5)))

header("6. Semantic-style similarity (untrained, so only illustrative)")
a = enc(torch.tensor([[1, 12, 7, 20]]))
b = enc(torch.tensor([[1, 12, 7, 20, PAD, PAD]]))
c = enc(torch.tensor([[1, 40, 41, 42]]))
cos = torch.nn.functional.cosine_similarity
print("same text, with/without padding:", cos(a, b).item())
print("different text:", cos(a, c).item())

header("7. Gradients flow to every parameter")
enc.train()
enc(batch).pow(2).mean().backward()
print("all params have grads:", all(p.grad is not None for p in enc.parameters()))
