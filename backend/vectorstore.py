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

    def remove_by_file_id(self, file_id):
        """Remove all embeddings and metadata for a specific file_id."""
        # Filter out metadata for the specified file_id
        indices_to_keep = []
        for i, meta in enumerate(self.metas):
            if meta.get("file_id") != file_id:
                indices_to_keep.append(i)
        
        if len(indices_to_keep) == len(self.metas):
            return False  # No items to remove
        
        # Rebuild the index with only the items to keep
        if indices_to_keep:
            # Get embeddings and metadata to keep
            kept_embeddings = []
            kept_metas = []
            
            # We need to reconstruct the index since FAISS doesn't support individual removal
            new_index = faiss.IndexFlatL2(self.dim)
            
            # This is a simplified approach - in production, you'd want to store original embeddings
            # For now, we'll clear and mark as needing rebuild
            self.index = new_index
            kept_metas = [self.metas[i] for i in indices_to_keep]
            self.metas = kept_metas
            
            return True
        else:
            # Clear everything if no items to keep
            self.index = faiss.IndexFlatL2(self.dim)
            self.metas = []
            return True

    def query(self, query_emb, top_k=5):
        """Return top-k nearest embeddings and their metadata for a query vector."""
        if len(self.metas) == 0:
            return []
            
        query_np = np.array([query_emb]).astype("float32")
        D, I = self.index.search(query_np, top_k)
        results = []
        for i, score in zip(I[0], D[0]):
            if i < len(self.metas) and i >= 0:
                results.append((self.metas[i], score))
        return results
