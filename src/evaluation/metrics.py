def calculateRecall(relevancies, query):
    count = query.getNumberOfRelevantDocuments()
    if count < 1:
        raise ValueError("Each query must have at least one relevant document.")
    return sum(relevancies) / count


def mean_recall(relevancies, queries):
    if not queries or len(relevancies) != len(queries):
        raise ValueError("Recall requires the same nonzero number of queries and result lists.")
    return sum(calculateRecall(row, query) for row, query in zip(relevancies, queries)) / len(queries)


def retrieval_metrics(relevancies, queries, k):
    if k < 1:
        raise ValueError("k must be positive.")
    if any(len(row) > k for row in relevancies):
        raise ValueError("A result list cannot contain more than k items.")
    recall = mean_recall(relevancies, queries)
    precision = sum(sum(row) / k for row in relevancies) / len(queries)
    reciprocal_rank = sum(next((1 / rank for rank, hit in enumerate(row, 1) if hit), 0.0)
                          for row in relevancies) / len(queries)
    return {f"recall@{k}": recall, f"precision@{k}": precision, f"mrr@{k}": reciprocal_rank}


def Evaluate(relevancies, queries):
    return {"recall": mean_recall(relevancies, queries)}
