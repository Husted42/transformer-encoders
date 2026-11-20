import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from src.retrievers.base_retriever import Retriever


class TFIDF(Retriever):
    def __init__(self, documents=None, index_path=None):
        super().__init__(documents, index_path)
        self.counts = [Counter(self.PreprocessText(doc.GetText())) for doc in self.index.GetDocuments()]
        self.corpus_vocabulary = set().union(*(set(counts) for counts in self.counts))
        self.term_to_index = {term: index for index, term in enumerate(sorted(self.corpus_vocabulary))}
        frequencies = self.GetDocumentFrequencies()
        n_docs = len(self.counts)
        self.idf = {term: np.log((n_docs + 1) / (count + 1)) + 1
                    for term, count in frequencies.items()}
        self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text):
        return re.findall(r"\w+", text.lower())

    def GetQueryVocabulary(self, query):
        return list(set(self.PreprocessText(query)))

    def GetDocumentVocabulary(self, document):
        return self.GetQueryVocabulary(document.GetText())

    def GetCorpusVocabulary(self):
        return self.corpus_vocabulary

    def GetDocumentFrequencies(self):
        return dict(Counter(term for counts in self.counts for term in counts))

    def GetInverseDocumentFrequencies(self):
        return self.idf

    def GetDocumentTermCounts(self, document):
        return dict(Counter(self.PreprocessText(document.GetText())))

    def _matrix(self, values):
        rows, columns, data = [], [], []
        for row, counts in enumerate(self.counts):
            length = sum(counts.values())
            for term, count in counts.items():
                rows.append(row)
                columns.append(self.term_to_index[term])
                data.append(values(row, term, count, length))
        return csr_matrix((data, (rows, columns)),
                          shape=(len(self.counts), len(self.term_to_index)), dtype=np.float32)

    def GetDocumentsTFIDFVectors(self):
        return self._matrix(lambda row, term, count, length: count / length * self.idf[term])

    def QueryToVector(self, queries):
        rows, columns, values = [], [], []
        for row, query in enumerate(queries):
            for term, count in Counter(self.PreprocessText(query)).items():
                if term in self.term_to_index:
                    rows.append(row)
                    columns.append(self.term_to_index[term])
                    values.append(count)
        return csr_matrix((values, (rows, columns)),
                          shape=(len(queries), len(self.term_to_index)), dtype=np.float32)

    def CalculateScores(self, queries):
        return self.QueryToVector(queries).dot(self.tfidf_matrix.T).toarray()
