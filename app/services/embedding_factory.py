"""
Factory for creating embedding functions based on configuration.
"""

from chromadb.api.types import EmbeddingFunction

from app.config import settings
from app.services.gemini_embedding import GeminiEmbeddingFunction
from app.services.sentence_transformer_embedding import SentenceTransformerEmbeddingFunction


def create_embedding_function(provider: str = None) -> EmbeddingFunction:
    """
    Create an embedding function based on the configured provider.

    Args:
        provider: Embedding provider name (defaults to settings.embedding_provider)
                 Options: "gemini", "sentence-transformers"

    Returns:
        EmbeddingFunction instance

    Raises:
        ValueError: If provider is not supported
    """
    provider = provider or settings.embedding_provider

    if provider == "gemini":
        return GeminiEmbeddingFunction()
    elif provider == "sentence-transformers":
        return SentenceTransformerEmbeddingFunction()
    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Options: 'gemini', 'sentence-transformers'"
        )
