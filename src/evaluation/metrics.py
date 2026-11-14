from src.data_utils.query import Query

def calculateRecall(relevancies: list[bool], query: Query):
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())

def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {'recall': None}
    n_queries = len(queries)
    total_recall = 0
    
    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
    
    scores['recall'] = total_recall / n_queries
    return scores


def mean_recall(relevancies: list[list[bool]], queries: list[Query]) -> float:
    """Mean fraction of known relevant documents retrieved for each query."""
    if not queries or len(relevancies) != len(queries):
        raise ValueError("Recall requires the same nonzero number of queries and result lists.")
    counts = [query.getNumberOfRelevantDocuments() for query in queries]
    if any(count < 1 for count in counts):
        raise ValueError("Each query must have at least one relevant document.")
    return sum(sum(row) / count for row, count in zip(relevancies, counts)) / len(queries)
