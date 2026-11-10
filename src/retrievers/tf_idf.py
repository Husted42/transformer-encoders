from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
import re
from scipy.sparse import lil_matrix


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """

        for i in text:
            if not i.isalnum() and i!=' ':
                text=re.sub(str([i]),'',text)
        text=text.lower().split()
        return text
        
    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        text=self.PreprocessText(query)
        query_unique_words=list(set(text))
        return query_unique_words

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        
        doc_text=self.PreprocessText(document.GetText())
        doc_unique_words=list(set(doc_text))
        return doc_unique_words
    
    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        doc_frequency={}
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        all_docs_text=list(*processed_texts)
        for term in self.corpus_vocabulary:
            count=0
            for doc in all_docs_text:
                if term in doc:
                    count+=1
            doc_frequency[term]=count
        return doc_frequency
    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        n_docs = len(self.index.GetDocuments())
        doc_freq=self.GetDocumentFrequencies()
        inverse_doc_freq={}
        for i in doc_freq:
            inverse_doc_freq[i]=np.log(n_docs/doc_freq[i])
        return inverse_doc_freq
    
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        doc_text=self.PreprocessText(document.GetText())
        doc_term_count={}

        for term in doc_text:
            doc_term_count[term]=doc_term_count.get(term,0)+1
        return doc_term_count



    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        
        docs=self.index.GetDocuments()
        for term in self.corpus_vocabulary:
            term_idx=self.term_to_index[term]
            term_idf=self.idf[term]
            for doc_idx,doc in enumerate(docs):
                doc_term_count=self.GetDocumentTermCounts(doc)
                tf=doc_term_count[term]/sum(doc_term_count.values())
                tfidf_matrix[doc_idx,term_idx]=tf*term_idf
        
        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        

        for query_idx, query in enumerate(queries):
            clean_text=self.PreprocessText(query)

            term_count={}
            for term in clean_text:
                term_count[term]=term_count.get(term,0)+1
            
            for term,count in term_count.items():
                if term in self.term_to_index():
                    term_idx=self.term_to_index[term]
                    query_vector[query_idx,term_idx]=count

        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    