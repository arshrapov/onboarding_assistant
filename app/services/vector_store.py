"""
ChromaDB implementation of the VectorStore interface.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import EmbeddingFunction
from typing import List, Optional, Dict, Any
import hashlib

from app.core.models import VectorStore, CodeChunk, SearchResult
from app.config import settings
from app.services.gemini_embedding import GeminiEmbeddingFunction


class ChromaDBStore(VectorStore):
    """ChromaDB implementation of vector storage."""

    def __init__(self, embedding_function: EmbeddingFunction = None):
        """Initialize ChromaDB client based on configuration.

        Args:
            embedding_function: ChromaDB embedding function (defaults to GeminiEmbeddingFunction)
        """
        self.embedding_function = embedding_function or GeminiEmbeddingFunction()
        self.mode = settings.chroma_mode

        if self.mode == "server":
            # Server mode: connect to remote ChromaDB instance
            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port
            )
        else:
            # Embedded mode: use local persistent storage
            self.client = chromadb.PersistentClient(
                path=settings.indexes_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

    def _get_full_collection_name(self, collection_name: str) -> str:
        """
        Get full collection name with prefix.

        Args:
            collection_name: Base collection name

        Returns:
            Full collection name with prefix
        """
        return f"{settings.chroma_collection_prefix}{collection_name}"

    def create_collection(self, collection_name: str) -> None:
        """
        Create a new collection for storing embeddings.

        Args:
            collection_name: Name of the collection
        """
        full_name = self._get_full_collection_name(collection_name)

        # Delete if exists (for fresh indexing)
        try:
            self.client.delete_collection(name=full_name)
        except Exception:
            pass  # Collection doesn't exist, that's fine

        # Create with cosine similarity metric
        # Use embedding_function for ChromaDB
        self.client.create_collection(
            name=full_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection and all its data.

        Args:
            collection_name: Name of the collection to delete
        """
        full_name = self._get_full_collection_name(collection_name)
        self.client.delete_collection(name=full_name)

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists, False otherwise
        """
        full_name = self._get_full_collection_name(collection_name)
        try:
            self.client.get_collection(name=full_name, embedding_function=self.embedding_function)
            return True
        except Exception:
            return False

    def add_documents(
        self,
        collection_name: str,
        chunks: List[CodeChunk]
    ) -> None:
        """
        Add documents to a collection. Embeddings are generated automatically
        by the embedding_function.

        Args:
            collection_name: Name of the collection
            chunks: List of code chunks to add
        """
        full_name = self._get_full_collection_name(collection_name)
        collection = self.client.get_collection(name=full_name, embedding_function=self.embedding_function)

        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "chunk_type": chunk.chunk_type,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language or "unknown",
                "signature": chunk.signature or "",
                **chunk.metadata
            }
            for chunk in chunks
        ]

        # Add to collection in batches
        # ChromaDB will automatically call embedding_function to generate embeddings
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents using vector similarity.
        Query embedding is generated automatically by the embedding_function.

        Args:
            collection_name: Name of the collection to search
            query_text: Query text (embedding will be generated automatically)
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results sorted by similarity
        """
        full_name = self._get_full_collection_name(collection_name)
        collection = self.client.get_collection(name=full_name, embedding_function=self.embedding_function)

        # Perform search - ChromaDB will automatically generate query embedding
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=filter_metadata
        )

        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                document = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                # Convert distance to similarity score (cosine: lower is better)
                # ChromaDB returns cosine distance [0, 2], convert to similarity [0, 1]
                score = 1 - (distance / 2)

                # Reconstruct CodeChunk
                chunk = CodeChunk(
                    id=chunk_id,
                    content=document,
                    chunk_type=metadata.get("chunk_type", "file"),
                    file_path=metadata.get("file_path", ""),
                    start_line=metadata.get("start_line", 0),
                    end_line=metadata.get("end_line", 0),
                    language=metadata.get("language"),
                    signature=metadata.get("signature"),
                    metadata={
                        k: v for k, v in metadata.items()
                        if k not in ["chunk_type", "file_path", "start_line",
                                    "end_line", "language", "signature"]
                    }
                )

                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=score,
                        distance=distance
                    )
                )

        return sorted(search_results, key=lambda x: x.score, reverse=True)

    def get_collection_count(self, collection_name: str) -> int:
        """
        Get the number of documents in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of documents in the collection
        """
        full_name = self._get_full_collection_name(collection_name)
        collection = self.client.get_collection(name=full_name, embedding_function=self.embedding_function)
        return collection.count()


def create_collection_name(repo_url: str) -> str:
    """
    Create a unique collection name from repository URL.

    Args:
        repo_url: Git repository URL

    Returns:
        Sanitized collection name
    """
    # Create hash of URL for uniqueness
    url_hash = hashlib.md5(repo_url.encode()).hexdigest()[:8]

    # Extract repo name from URL
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_name = repo_name.replace(".git", "")

    # Sanitize: ChromaDB collection names have restrictions
    repo_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in repo_name)

    return f"{repo_name}_{url_hash}"
