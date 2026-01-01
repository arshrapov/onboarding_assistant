"""
Gemini embedding function for ChromaDB.
"""

from typing import List
from chromadb.api.types import EmbeddingFunction, Documents
import google.generativeai as genai

from app.config import settings


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using Google Gemini API.
    """

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize Gemini embedding function.

        Args:
            api_key: Google API key (defaults to settings.gemini_api_key)
            model_name: Embedding model name (defaults to settings.embedding_model)
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.embedding_model

        # Configure Gemini API
        genai.configure(api_key=self.api_key)

    def __call__(self, input: Documents) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        This method is called by ChromaDB.

        Args:
            input: List of documents (strings) from ChromaDB

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches (Gemini supports batch embedding)
        batch_size = 100
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]

            result = genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type="retrieval_document"
            )

            embeddings.extend(result["embedding"])

        return embeddings
