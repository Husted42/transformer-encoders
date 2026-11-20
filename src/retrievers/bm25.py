import numpy as np

from src.retrievers.tf_idf import TFIDF


class BM25(TFIDF):
    def __init__(self, documents=None, index_path=None, k1=1.5, b=0.75):
        if not np.isfinite(k1) or k1 <= 0 or not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires positive k1 and b between 0 and 1.")
        super().__init__(documents, index_path)
        self.k1, self.b = k1, b
        self.idf_bm25 = self.GetInverseDocumentFrequenciesBM25()
        self.bm25_matrix = self.GetDocumentsBM25Vectors()

    def GetDocumentLengths(self):
        lengths = [sum(counts.values()) for counts in self.counts]
        return dict(zip(self.index.GetDocuments(), lengths)), np.mean(lengths) if lengths else 0.0

    def GetInverseDocumentFrequenciesBM25(self):
        n_docs = len(self.counts)
        return {term: np.log(1 + (n_docs - count + 0.5) / (count + 0.5))
                for term, count in self.GetDocumentFrequencies().items()}

    def GetDocumentsBM25Vectors(self):
        lengths = [sum(counts.values()) for counts in self.counts]
        average = np.mean(lengths) if lengths else 0.0
        if not average:
            return self._matrix(lambda row, term, count, length: 0.0)
        return self._matrix(lambda row, term, count, length:
                            self.idf_bm25[term] * count * (self.k1 + 1) /
                            (count + self.k1 * (1 - self.b + self.b * length / average)))

    def CalculateScores(self, queries):
        return self.QueryToVector(queries).dot(self.bm25_matrix.T).toarray()
