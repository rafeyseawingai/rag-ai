from sdk.time_logger import TimeLogger  # For tracking execution time
import time  # For measuring time intervals
import numpy as np  # For numerical operations and array handling
from sentence_transformers import SentenceTransformer  # For generating text embeddings
import faiss  # Facebook AI Similarity Search for efficient vector search

class EmbeddingRetriever:
    """Generates embeddings and retrieves the most relevant document chunks using FAISS."""
    def __init__(self, document_chunks):
        # Start timing the initialization process
        start_time = time.time()
        
        # Initialize the sentence transformer model with a pre-trained model
        # all-MiniLM-L6-v2 is a lightweight model good for generating text embeddings
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Store the input document chunks for later reference
        self.chunks = document_chunks
        
        # Convert all document chunks to embeddings using the model
        # Store as numpy array for efficient computation
        self.embeddings = np.array(self.model.encode(self.chunks))
        
        # Create and store FAISS index for similarity search
        self.index = self.create_faiss_index()
        
        # Log the completion time of initialization
        TimeLogger.log("Embedding generation and FAISS indexing completed", start_time)
    
    def create_faiss_index(self):
        """Creates a FAISS index for efficient search."""
        # Get the dimensionality of the embeddings
        dimension = self.embeddings.shape[1]
        
        # Create a FAISS index that uses L2 (Euclidean) distance
        index = faiss.IndexFlatL2(dimension)
        
        # Add the document embeddings to the FAISS index
        index.add(self.embeddings)
        
        # Return the populated index
        return index
    
    def retrieve_context(self, query, top_k=2):
        """Finds the most relevant chunks using FAISS."""
        # Start timing the retrieval process
        start_time = time.time()
        
        # Convert query to embedding and ensure correct data type
        query_embedding = np.array(self.model.encode([query])).astype('float32')
        
        # Search the FAISS index for top_k most similar vectors
        # Returns distances and indices of nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Log completion time of retrieval
        TimeLogger.log("Context retrieval completed", start_time)
        
        # Handle edge case where no matches are found
        if len(indices[0]) == 0:
            return [], []
        
        # Return the matching chunks and their indices
        # indices[0] is used because FAISS returns results in a 2D array
        return [self.chunks[i] for i in indices[0]], indices[0].tolist()
