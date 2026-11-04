"""Run small examples to inspect the encoder and its components."""

import torch

from src.retrievers.models.encoder import (
    EncoderBlock,
    FeedForwardNetwork,
    MultiHeadSelfAttention,
    PositionalEncoding,
    ScaledDotProductAttention,
    TokenEmbedding,
    TransformerEncoder,
)


def test_encoder():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 6
    d_model = 12
    n_heads = 3
    vocab_size = 100
    n_layers = 2
    feed_forward_dim = 24

    # Start with real tokens, then pad the last two positions of one sequence.
    token_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    token_ids[0, -2:] = 0
    mask = (token_ids != 0).unsqueeze(1)
    print("Token IDs:")
    print(token_ids)

    print("\n1. Token embeddings")
    embedding = TokenEmbedding(vocab_size, d_model, padding_idx=0)
    x = embedding(token_ids)
    print("Embedding shape:", x.shape)
    print("Padding vector:", x[0, -1])
    assert x.shape == (batch_size, seq_len, d_model)
    assert torch.all(x[0, -1] == 0)

    print("\n2. Positional encoding")
    positional_encoding = PositionalEncoding(d_model, max_len=20)
    x_pe = positional_encoding(x)
    print(f"Before: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
    print(f"After:  mean={x_pe.mean().item():.4f}, std={x_pe.std().item():.4f}")
    print("First position vector:", positional_encoding.pe[0, 0])
    assert x_pe.shape == x.shape

    print("\n3. Scaled dot-product attention")
    attention = ScaledDotProductAttention()
    q = torch.tensor([[[1., 0.], [0., 1.]]])
    k = q.clone()
    v = torch.tensor([[[2., 0.], [0., 6.]]])
    output, weights = attention(q, k, v)
    print("Attention weights:", weights)
    print("Output:", output)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 2))
    expected = torch.tensor([[[1.3395, 1.9814], [0.6605, 4.0186]]])
    assert torch.allclose(output, expected, atol=1e-4)

    print("\n4. Multi-head self-attention")
    mha = MultiHeadSelfAttention(d_model, n_heads)
    attention_output, weights = mha(x_pe, mask)
    print("Output shape:", attention_output.shape)
    print("Per-head weights shape:", weights.shape)
    print("Weights assigned to padding:", weights[0, :, :, -2:])
    assert attention_output.shape == x_pe.shape
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
    assert torch.all(weights[0, :, :, -2:] == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, n_heads, seq_len))

    print("\n5. Feed-forward network")
    feed_forward = FeedForwardNetwork(d_model, feed_forward_dim)
    ffn_output = feed_forward(x_pe)
    print("Input shape:", x_pe.shape)
    print("Output shape:", ffn_output.shape)
    assert ffn_output.shape == x_pe.shape

    print("\n6. Encoder block")
    block = EncoderBlock(d_model, n_heads, feed_forward_dim, dropout=0.1)
    block.eval()  # Disable dropout so repeated runs can be compared.
    block_output = block(x_pe, mask)
    print("Output shape:", block_output.shape)
    print("Output differs from input:", not torch.allclose(x_pe, block_output))
    assert block_output.shape == x_pe.shape
    assert torch.isfinite(block_output).all()

    print("\n7. Full encoder")
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        feed_forward_dim=feed_forward_dim,
        max_len=20,
        pooling_type="mean",
    )
    encoder.eval()
    tokens = encoder(token_ids, return_token_embeddings=True)
    pooled = encoder(token_ids)
    print("Contextual token shape:", tokens.shape)
    print("Pooled sequence shape:", pooled.shape)
    assert tokens.shape == (batch_size, seq_len, d_model)
    assert pooled.shape == (batch_size, d_model)
    assert torch.isfinite(pooled).all()

    # Removing trailing padding should not change the sequence vector.
    unpadded = encoder(token_ids[:1, :-2])
    padding_ignored = torch.allclose(unpadded[0], pooled[0], atol=1e-5)
    print("Padding leaves sequence vector unchanged:", padding_ignored)
    assert padding_ignored

    print("\n8. Gradient check")
    encoder.train()
    output = encoder(token_ids)
    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    print("Dummy loss:", loss.item())
    has_gradients = all(parameter.grad is not None for parameter in encoder.parameters())
    print("All learned parameters received gradients:", has_gradients)
    assert has_gradients

    print("\n9. Parameter count")
    print("Total parameters:", sum(p.numel() for p in encoder.parameters()))
    print("Embedding parameters:", encoder.token_embedding.embedding.weight.numel())
    print("\nBasic encoder implementation check passed.")


if __name__ == "__main__":
    test_encoder()
