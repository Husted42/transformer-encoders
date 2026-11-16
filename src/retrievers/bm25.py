from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.idf_bm25 = self.GetInverseDocumentFrequenciesBM25()
    
    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())

    def GetInverseDocumentFrequenciesBM25(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        n_docs = len(self.index.GetDocuments())
        doc_freq=self.GetDocumentFrequencies()
        bm25_inverse_doc_freq={}
        for i in doc_freq:
            bm25_inverse_doc_freq[i]=np.log((n_docs+1)/(doc_freq[i]+0.5))
        return bm25_inverse_doc_freq

    def GetDocumentsBM25Vectors(self):
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}
        
        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        bm25_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        docs=self.index.GetDocuments()
        doc_length=self.GetDocumentLengths()[0]
        avg_doc_length=self.GetDocumentLengths()[1]
        for term in self.corpus_vocabulary:
            term_idx=self.term_to_index[term]
            term_idf=self.idf_bm25[term]
            for doc_idx,doc in enumerate(docs):
                doc_term_count=self.GetDocumentTermCounts(doc)
                tf=(doc_term_count[term]*(self.k1+1))/(doc_term_count[term]+self.k1*(1-self.b+self.b*(doc_length[doc]/avg_doc_length)))
                bm25_matrix[doc_idx,term_idx]=tf*term_idf

        return bm25_matrix.tocsr()   # Convert to CSR format for efficient row slicing

    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores