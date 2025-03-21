from sdk.time_logger import TimeLogger
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingRetriever:
    """Generates embeddings and retrieves the most relevant document chunks using FAISS."""
    def __init__(self, document_chunks):
        start_time = time.time()
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.chunks = document_chunks
        self.embeddings = np.array(self.model.encode(self.chunks))  # Store embeddings
        self.index = self.create_faiss_index()
        TimeLogger.log("Embedding generation and FAISS indexing completed", start_time)
    
    def create_faiss_index(self):
        """Creates a FAISS index for efficient search."""
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.embeddings)
        return index
    
    def retrieve_context(self, query, top_k=2):
        """Finds the most relevant chunks using FAISS."""
        start_time = time.time()
        query_embedding = np.array(self.model.encode([query])).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        TimeLogger.log("Context retrieval completed", start_time)
        if len(indices[0]) == 0:
            return [], []  # Edge case: No relevant matches found
        return [self.chunks[i] for i in indices[0]], indices[0].tolist()
