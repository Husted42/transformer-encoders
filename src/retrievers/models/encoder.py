import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class ScaledDotProductAttention(nn.Module):
    """
    Combine value vectors using scaled query-key attention.
    """
    def __init__(self):
        super().__init__()

    def forward(self,q,k,v,mask=None):
        """Compute attention weights and their weighted combination of values.

        Args:
            q : Queries of shape (B, L_q, d_k).
            k : Keys of shape (B, L_k, d_k).
            v : Values of shape (B, L_k, d_v).
            mask : Boolean tensor broadcastable to (B, L_q, L_k).
                True allows attention; False blocks it.
                A key padding mask has shape (B, 1, L_k). Defaults to None.

        Returns:
            (B, L_q, d_v) and attention weights of shape (B, L_q, L_k), in that order.

        Note:
            B is batch size, 
            L_q and L_k are query and key sequence lengths,
            d_k and d_v are query/key and value feature widths.
        """
        scores = q @ k.transpose(-2,-1)
        scaled_scores = scores/sqrt(q.shape[-1])
        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(~mask,float("-inf"))
        attention_weights = F.softmax(scaled_scores,dim=-1)
        output = attention_weights @ v
        return (output,attention_weights)


class MultiHeadSelfAttention(nn.Module):
    """Let each token gather information through several attention heads.

    Args:
        d_model: Number of features in each input and output token.
        n_heads: Number of heads. Must divide d_model evenly.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        if n_heads <= 0  or d_model <= 0:
            raise ValueError("d_model and n_heads needs to be positive")

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model,d_model)
        self.k_proj = nn.Linear(d_model,d_model)
        self.v_proj = nn.Linear(d_model,d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, x, mask=None):
        """Project tokens, attend within each head, and combine the results.

        Args:
            x : Token vectors with shape (B, L, d_model).
            mask : Boolean mask with shape (B, 1, L).
                True allows attention to a key; False blocks it. Each sequence
                must have at least one allowed key. Defaults to None.

        Returns:
            Output with shape (B, L, d_model)
            and per-head attention weights with shape (B, n_heads, L, L).

        Note:
            B is batch size and L is sequence length. 
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        batch_size, seq_len, _ = x.shape
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        head_outputs, attention_weights = self.attention(q, k, v, mask)
        combined = head_outputs.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(combined)
        return output, attention_weights

class FeedForwardNetwork(nn.Module):
    """Transform each token's features using the same two-layer network.

    Args:
        d_model (int): Number of features in each input and output token.
        feed_forward_dim (int): Number of features in the intermediate layer.
    """

    def __init__(self,d_model:int,feed_forward_dim:int):
        super().__init__()
        self.linear1 = nn.Linear(d_model,feed_forward_dim)
        self.linear2 = nn.Linear(feed_forward_dim,d_model)

    def forward(self,x):
        """Apply the feed-forward network at every token position.

        Args:
            x : shape (B, L, d_model), where is batch size and L is sequence length.

        Returns:
            torch.Tensor: Transformed vectors with shape (B, L, d_model).
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class TokenEmbedding(nn.Module):
    """Look up a learned vector for each token ID.

    Args:
        vocab_size : Number of entries in the token vocabulary.
        d_model : Number of features in each embedding vector.
        padding_idx : Padding token ID. Its row starts at zero and receives
            no gradients from embedding lookups. Defaults to 0.
    """

    def __init__(self,vocab_size:int,d_model:int,padding_idx:int=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model,padding_idx)

    def forward(self,token_ids):
        """Retrieve the embedding for each token in the batch.

        Args:
            token_ids : Integer IDs with shape (B, L), where B is
                batch size and L is sequence length. IDs must be valid vocabulary
                indices; torch.long is the usual input dtype.

        Returns:
            torch.Tensor: Token vectors with shape (B, L, d_model).
        """
        return self.embedding(token_ids)

class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal position information to token vectors.

    Sine and cosine features use different frequencies to represent positions.
    The table is stored as a buffer.

    Args:
        d_model : Number of features in each token and position vector.
        max_len : Number of positions to precompute. Inputs can be shorter
            than this capacity, but cannot exceed it.
    """
    def __init__(self,d_model:int,max_len:int):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        even_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
        frequencies = 10000.0 ** (-even_indices / d_model)
        angles = position * frequencies
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, : d_model // 2])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """Add the required position vectors to the input embeddings.

        Args:
            x (torch.Tensor): Floating-point token vectors with shape
                (B, L, d_model), where B is batch size and L is sequence length.
                The input and this module must be on the same device.

        Returns:
            torch.Tensor: Vectors with position information, with the same
            shape and dtype as x.
        """
        seq_len = x.shape[1]

        if seq_len > self.pe.shape[1]:
            raise ValueError("Input sequence exceeds positional encoding capacity.")

        return x + self.pe[:, :seq_len].to(dtype=x.dtype)


class EncoderBlock(nn.Module):
    """Apply self-attention and a feed-forward network with post-norm residuals.

    Each stage applies dropout to its output, adds the stage's input, and then
    applies layer normalization across token features.

    Args:
        d_model : Number of features in each token vector.
        n_heads : Number of attention heads. Must divide d_model evenly.
        feed_forward_dim : Intermediate width of the feed-forward network.
        dropout : Probability of dropping a branch activation during
            training.
    """

    def __init__(self,d_model:int,n_heads:int,feed_forward_dim:int,dropout:float=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model,n_heads)
        self.feed_forward = FeedForwardNetwork(d_model,feed_forward_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        """Gather context, then transform each token's features.

        Args:
            x : Token vectors with shape (B, L, d_model), where
                B is batch size and L is sequence length.
            mask (: Boolean key mask with shape (B, 1, L).
                True allows attention; False blocks it. Each sequence must have
                at least one allowed key. Defaults to None.

        Returns:
            torch.Tensor: Contextual token vectors with shape (B, L, d_model).
        """
        attention_output , _ = self.attention(x,mask)
        x = self.norm1(x+self.dropout1(attention_output))
        ffn_output = self.feed_forward(x)
        x = self.norm2(x+self.dropout2(ffn_output))
        return x

class TransformerEncoder(nn.Module):
    """Combine token embeddings, positions, and independently learned encoder blocks.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Number of features in each token vector.
        n_heads: Number of attention heads in each block.
        n_layers: Number of encoder blocks to create.
        feed_forward_dim: Intermediate width of each feed-forward network.
        max_len: Maximum sequence length supported by positional encoding.
        padding_idx: Padding token ID. Defaults to 0.
        dropout: Dropout probability in each block. Defaults to 0.1.
        pooling_type: Mean, max, or cls pooling. Defaults to mean. For cls,
            the caller must put a CLS token at the first position.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        feed_forward_dim: int,
        max_len: int,
        padding_idx: int = 0,
        dropout: float = 0.1,
        pooling_type: str = "mean",
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        if pooling_type not in {"mean", "max", "cls"}:
            raise ValueError("pooling_type must be mean, max, or cls.")
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model 
        self.padding_idx = padding_idx
        self.pooling_type = pooling_type
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=padding_idx,
        )
        self.position_encoding = PositionalEncoding(d_model,max_len)
        self.feed_forward_dim = feed_forward_dim
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, feed_forward_dim, dropout)
            for _ in range(n_layers)
        ])

    def pool(self, x, mask=None):
        """Combine contextual token vectors into one vector per sequence.

        Args:
            x: Token vectors with shape (B, L, d_model).
            mask: Optional boolean mask with shape (B, 1, L). True marks valid
                tokens. If omitted, every position participates in pooling.

        Returns:
            torch.Tensor: Sequence vectors with shape (B, d_model).

        """
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("Pooling expects a nonempty (B, L, d_model) tensor.")
        if mask is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            if mask.dtype != torch.bool or mask.shape != (x.shape[0], 1, x.shape[1]):
                raise ValueError("mask must be boolean with shape (B, 1, L).")
            valid = mask.squeeze(1)
        if not valid.any(dim=1).all():
            raise ValueError("Each sequence must contain at least one valid token.")

        if self.pooling_type == "cls":
            if not valid[:, 0].all():
                raise ValueError("CLS pooling requires a valid token at position 0.")
            return x[:, 0]
        if self.pooling_type == "max":
            return x.masked_fill(~valid.unsqueeze(-1), float("-inf")).amax(dim=1)

        # Padded query outputs need not be zero, so explicitly exclude them.
        total = x.masked_fill(~valid.unsqueeze(-1), 0.0).sum(dim=1)
        count = valid.sum(dim=1, keepdim=True).to(dtype=x.dtype)
        return total / count

    def forward(self, token_ids, return_token_embeddings: bool = False):
        """Embed tokens, add positions, run the blocks, and pool the result.

        Args:
            token_ids: Integer token IDs with shape (B, L). Padding is identified
                by padding_idx; every sequence must have a nonpadding token.
            return_token_embeddings: Return contextual vectors for each token
                instead of pooling. Defaults to False.

        Returns:
            torch.Tensor: Shape (B, d_model), or (B, L, d_model) when
            return_token_embeddings is True. Padded token outputs are not zeroed.

        """
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have nonempty shape (B, L).")
        mask = (token_ids != self.padding_idx).unsqueeze(1)
        if not mask.any(dim=-1).all():
            raise ValueError("Each sequence must contain at least one nonpadding token.")

        x = self.token_embedding(token_ids)
        x = self.position_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        if return_token_embeddings:
            return x
        return self.pool(x, mask)
