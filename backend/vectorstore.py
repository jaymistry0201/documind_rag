import faiss
import numpy as np

class FaissVectorStore:
    """Simple FAISS-based vector store for adding and querying embeddings."""

    def __init__(self, dim):
        """Initialize a FAISS L2 index with given embedding dimension."""
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metas = []

    def add(self, embeddings, metas):
        """Add embeddings and their metadata to the index."""
        embs_np = np.array(embeddings).astype("float32")
        self.index.add(embs_np)
        self.metas.extend(metas)

    def query(self, query_emb, top_k=5):
        """Return top-k nearest embeddings and their metadata for a query vector."""
        query_np = np.array([query_emb]).astype("float32")
        D, I = self.index.search(query_np, top_k)
        results = []
        for i, score in zip(I[0], D[0]):
            if i < len(self.metas):
                results.append((self.metas[i], score))
        return results
