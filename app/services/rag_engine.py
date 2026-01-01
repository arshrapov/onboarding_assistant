"""
RAG (Retrieval-Augmented Generation) Engine for code question answering.
"""

from typing import List, Optional, Dict, Any
from chromadb.api.types import EmbeddingFunction
import google.generativeai as genai

from app.core.models import VectorStore, CodeChunk, SearchResult
from app.services.vector_store import ChromaDBStore, create_collection_name
from app.services.embedding_factory import create_embedding_function
from app.config import settings


class RAGEngine:
    """
    Orchestrates the RAG workflow:
    1. Indexing: Store code chunks with embeddings
    2. Retrieval: Find relevant code for queries
    3. Generation: Answer questions using retrieved context
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_function: Optional[EmbeddingFunction] = None
    ):
        """
        Initialize RAG engine.

        Args:
            vector_store: Vector store implementation (defaults to ChromaDBStore)
            embedding_function: ChromaDB embedding function (defaults to factory-created based on settings)
        """
        # Create embedding function first
        self.embedding_function = embedding_function or create_embedding_function()

        # Initialize vector store with embedding function
        self.vector_store = vector_store or ChromaDBStore(
            embedding_function=self.embedding_function
        )

        # Configure Gemini API for LLM generation
        genai.configure(api_key=settings.gemini_api_key)
        self.llm_model = genai.GenerativeModel(settings.gemini_model)

    def create_index(self, repo_url: str, repo_name: str) -> str:
        """
        Create a new vector index for a repository.

        Args:
            repo_url: Repository URL
            repo_name: Repository name

        Returns:
            Collection name created
        """
        collection_name = create_collection_name(repo_url)
        self.vector_store.create_collection(collection_name)
        return collection_name

    def delete_index(self, collection_name: str) -> None:
        """
        Delete a repository index.

        Args:
            collection_name: Name of the collection to delete
        """
        self.vector_store.delete_collection(collection_name)

    def index_exists(self, collection_name: str) -> bool:
        """
        Check if an index exists.

        Args:
            collection_name: Collection name

        Returns:
            True if index exists
        """
        return self.vector_store.collection_exists(collection_name)

    def add_chunks(
        self,
        collection_name: str,
        chunks: List[CodeChunk]
    ) -> None:
        """
        Add code chunks to the index. Embeddings are generated automatically
        by the vector store's embedding function.

        Args:
            collection_name: Collection to add to
            chunks: List of code chunks to index
        """
        if not chunks:
            return

        # Add to vector store (embeddings generated automatically)
        self.vector_store.add_documents(
            collection_name=collection_name,
            chunks=chunks
        )

    def search_code(
        self,
        collection_name: str,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for relevant code chunks. Query embedding is generated automatically
        by the vector store's embedding function.

        Args:
            collection_name: Collection to search
            query: Natural language query
            top_k: Number of results (defaults to settings.top_k_results)
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        if top_k is None:
            top_k = settings.top_k_results

        # Search vector store (query embedding generated automatically)
        return self.vector_store.search(
            collection_name=collection_name,
            query_text=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

    def answer_question(
        self,
        collection_name: str,
        question: str,
        top_k: Optional[int] = None
    ) -> str:
        """
        Answer a question using RAG.

        Args:
            collection_name: Repository collection to search
            question: User's question
            top_k: Number of context chunks to retrieve

        Returns:
            Generated answer with code references
        """
        # Retrieve relevant code
        search_results = self.search_code(
            collection_name=collection_name,
            query=question,
            top_k=top_k
        )

        if not search_results:
            return "I couldn't find relevant code to answer your question. Please try rephrasing or asking about a different topic."

        # Build context from search results
        context = self._build_context(search_results)

        # Generate answer
        prompt = self._build_qa_prompt(question, context, search_results)

        response = self.llm_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=settings.max_tokens,
                temperature=settings.temperature
            )
        )

        return response.text

    def _build_context(self, search_results: List[SearchResult]) -> str:
        """
        Build context string from search results.

        Args:
            search_results: List of search results

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(search_results, 1):
            chunk = result.chunk
            context_parts.append(
                f"--- Code Reference {i} ---\n"
                f"File: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}\n"
                f"Type: {chunk.chunk_type}\n"
                f"Language: {chunk.language or 'unknown'}\n"
                f"Relevance Score: {result.score:.2f}\n\n"
                f"```{chunk.language or ''}\n"
                f"{chunk.content}\n"
                f"```\n"
            )

        return "\n".join(context_parts)

    def _build_qa_prompt(
        self,
        question: str,
        context: str,
        search_results: List[SearchResult]
    ) -> str:
        """
        Build prompt for question answering.

        Args:
            question: User's question
            context: Context from search results
            search_results: Original search results for references

        Returns:
            Formatted prompt
        """
        # Build file references
        file_refs = []
        for result in search_results:
            chunk = result.chunk
            file_refs.append(
                f"- {chunk.file_path}:{chunk.start_line}"
            )

        references_list = "\n".join(file_refs)

        prompt = f"""You are a helpful code assistant analyzing a repository.

Answer the following question based on the provided code context.

QUESTION:
{question}

CODE CONTEXT:
{context}

INSTRUCTIONS:
1. Answer the question accurately based on the code context provided
2. Include specific file references using the format: file_path:line_number
3. If the code context doesn't fully answer the question, say so
4. Be concise but thorough
5. Use code snippets when helpful
6. Reference these files when relevant:
{references_list}

ANSWER:"""

        return prompt

    def get_collection_stats(self, collection_name: str) -> Dict[str, int]:
        """
        Get statistics about a collection.

        Args:
            collection_name: Collection name

        Returns:
            Dictionary with statistics
        """
        count = self.vector_store.get_collection_count(collection_name)
        return {
            "total_chunks": count,
            "collection_name": collection_name
        }
