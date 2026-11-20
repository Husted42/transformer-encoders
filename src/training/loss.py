import torch
import torch.nn as nn
import torch.nn.functional as F

class DPRLoss(nn.Module):
    """In-batch contrastive loss for Dense Passage Retrieval.

    Let q_i be query i's embedding and d_j document j's embedding. The score
    matrix is s_ij = q_i^T d_j. For the set P_i of positive documents for
    query i, the per-query loss is

        ell_i = -log(
            sum_{j in P_i} exp(s_ij) / sum_k exp(s_ik)
        )

    The goal is to learn the parameters of f and g so that the inner product between paired queries and
    documents is high and low otherwise
    """

    def forward(self, queries, documents, positive_mask=None):
        if queries.ndim != 2 or documents.ndim != 2 or queries.shape[1] != documents.shape[1]:
            raise ValueError("Query and document embeddings must be matrices with the same width.")
        if not len(queries) or not len(documents):
            raise ValueError("Query and document embeddings must be nonempty.")
        if queries.device != documents.device:
            raise ValueError("Query and document embeddings must be on the same device.")
        if not torch.isfinite(queries).all() or not torch.isfinite(documents).all():
            raise ValueError("Query and document embeddings must be finite.")
        # Get similarity scores s_ij for all query-document pairs
        scores = queries @ documents.T
        if not torch.isfinite(scores).all():
            raise ValueError("Similarity scores must be finite.")
        if positive_mask is None:
            if len(queries) != len(documents):
                raise ValueError("Diagonal positives require equal query and document counts.")
            return F.cross_entropy(scores, torch.arange(len(queries), device=scores.device))

        if positive_mask.dtype != torch.bool:
            raise ValueError("positive_mask must be boolean.")
        if positive_mask.shape != scores.shape or positive_mask.device != scores.device:
            raise ValueError("positive_mask must match the score matrix shape and device.")
        if not positive_mask.any(dim=1).all():
            raise ValueError("Every query must have at least one positive.")
        # calculated ell_i for all queries
        log_probabilities = F.log_softmax(scores, dim=1)
        return -torch.logsumexp(
            log_probabilities.masked_fill(~positive_mask, float("-inf")), dim=1
        ).mean()