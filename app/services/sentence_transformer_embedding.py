"""
Sentence Transformers embedding function for ChromaDB.
"""

from typing import List
from chromadb.api.types import EmbeddingFunction, Documents
from sentence_transformers import SentenceTransformer

from app.config import settings


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using Sentence Transformers.
    Runs locally without requiring external API calls.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize Sentence Transformer embedding function.

        Args:
            model_name: Model name (defaults to settings.sentence_transformer_model)
        """
        self.model_name = model_name or settings.sentence_transformer_model

        # Load the model (will download on first use, then cache locally)
        self.model = SentenceTransformer(self.model_name)

    def __call__(self, input: Documents) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        This method is called by ChromaDB.

        Args:
            input: List of documents (strings) from ChromaDB

        Returns:
            List of embedding vectors
        """
        # Generate embeddings using the model
        # convert_to_numpy=False returns a list instead of numpy array
        embeddings = self.model.encode(
            input,
            convert_to_tensor=False,
            show_progress_bar=False
        )

        # Convert to list of lists (ChromaDB expects this format)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
