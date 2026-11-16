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
        # Get similarity scores s_ij for all query-document pairs
        scores = queries @ documents.T
        if positive_mask is None:
            return F.cross_entropy(scores, torch.arange(len(queries), device=scores.device))

        # calculated ell_i for all queries
        log_probabilities = F.log_softmax(scores, dim=1)
        return -torch.logsumexp(
            log_probabilities.masked_fill(~positive_mask, float("-inf")), dim=1
        ).mean()