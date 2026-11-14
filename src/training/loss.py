import torch
import torch.nn as nn
import torch.nn.functional as F


class DPRLoss(nn.Module):
    def forward(self, queries, documents, positive_mask=None):
        if queries.ndim != 2 or documents.ndim != 2 or queries.shape[1] != documents.shape[1]:
            raise ValueError("Queries and documents must be 2D with matching embedding dimensions.")
        if queries.shape[0] == 0 or documents.shape[0] == 0:
            raise ValueError("A DPR batch must contain queries and documents.")
        scores = queries @ documents.T
        if positive_mask is None:
            if queries.shape[0] != documents.shape[0]:
                raise ValueError("Paired batches must have the same number of queries and documents.")
            return F.cross_entropy(scores, torch.arange(len(queries), device=scores.device))
        if positive_mask.dtype != torch.bool or positive_mask.shape != scores.shape:
            raise ValueError("positive_mask must be boolean and match the score matrix shape.")
        if positive_mask.device != scores.device:
            raise ValueError("positive_mask must be on the same device as the embeddings.")
        if not positive_mask.any(dim=1).all():
            raise ValueError("Each query must have at least one positive document.")
        log_probabilities = F.log_softmax(scores, dim=1)
        return -torch.logsumexp(log_probabilities.masked_fill(~positive_mask, float("-inf")), dim=1).mean()