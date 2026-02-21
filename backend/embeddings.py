from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    """Manages text embeddings for RAG using SentenceTransformer."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """Return embeddings vectors for a list of text chunks."""
        return self.model.encode(texts, show_progress_bar=True).tolist()
