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


class MultiHeadAttention(nn.Module):
    """each token gather information through several attention heads.

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
    def __init__(self,d_model:int,feed_forward_dim:int):
        super().__init__()
        self.linear1 = nn.Linear(d_model,feed_forward_dim)
        self.linear2 = nn.Linear(feed_forward_dim,d_model)

    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.model = None
        raise NotImplementedError()

    def pool(self, x):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.cfg.model.pooling_type == 'cls':
            raise NotImplementedError()
        
        elif self.cfg.model.pooling_type == 'max':
            raise NotImplementedError()
        
        elif self.cfg.model.pooling_type == 'mean':
            raise NotImplementedError()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        raise NotImplementedError()
