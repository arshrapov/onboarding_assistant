"""
RAG (Retrieval-Augmented Generation) Engine using LlamaIndex.
Handles repository loading, indexing, retrieval, and generation.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import pickle
import hashlib

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.readers.github import GithubRepositoryReader, GithubClient
import chromadb

# CodeChunk and SearchResult removed - working directly with LlamaIndex Documents
from app.utils.github_utils import parse_github_url, get_repo_identifier, normalize_github_url
from app.core.exceptions import RepositoryDownloadError
from app.config import settings


def create_collection_name(repo_url: str) -> str:
    """
    Create a unique collection name from repository URL.

    Args:
        repo_url: Git repository URL

    Returns:
        Sanitized collection name for ChromaDB
    """
    # Create hash of URL for uniqueness
    url_hash = hashlib.md5(repo_url.encode()).hexdigest()[:8]

    # Extract repo name from URL
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_name = repo_name.replace(".git", "")

    # Sanitize: ChromaDB collection names have restrictions
    repo_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in repo_name)

    # Combine with hash for uniqueness and add prefix
    collection_name = f"{settings.chroma_collection_prefix}{repo_name}_{url_hash}"

    return collection_name.lower()  # ChromaDB requires lowercase


class RAGEngine:
    """
    LlamaIndex-based RAG Engine that handles:
    1. Repository downloading from GitHub
    2. Document indexing with vector embeddings
    3. Semantic search
    4. Question answering with LLM
    """

    def __init__(self):
        """Initialize RAG engine with LlamaIndex components."""
        # Configure LlamaIndex global settings
        Settings.llm = Gemini(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        Settings.embed_model = GeminiEmbedding(
            api_key=settings.gemini_api_key,
            model_name=settings.embedding_model
        )

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.indexes_dir
        )

        # Store loaded indexes in memory
        self.indexes: Dict[str, VectorStoreIndex] = {}

        # Ensure directories exist
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        Path(settings.repos_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.indexes_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)

    # ==================== Repository Loading ====================

    def load_repository(
        self,
        repo_url: str,
        force: bool = False,
        github_token: Optional[str] = None
    ) -> tuple[str, List[Document]]:
        """
        Load a GitHub repository using LlamaIndex GithubRepositoryReader.

        Args:
            repo_url: GitHub repository URL
            force: If True, remove existing data and re-download
            github_token: Optional GitHub personal access token

        Returns:
            Tuple of (local_path, documents)

        Raises:
            RepositoryDownloadError: If loading fails
        """
        # Normalize URL
        try:
            normalized_url = normalize_github_url(repo_url)
        except Exception as e:
            raise RepositoryDownloadError(f"Invalid repository URL: {str(e)}")

        # Get storage path
        repo_identifier = get_repo_identifier(normalized_url)
        repo_path = Path(settings.repos_dir) / repo_identifier

        try:
            # Check if repository already exists
            if repo_path.exists():
                if force:
                    shutil.rmtree(repo_path)
                else:
                    # Try to load existing documents
                    documents = self._load_documents_from_disk(repo_path)
                    if documents:
                        return str(repo_path), documents
                    # If no documents found, re-download
                    shutil.rmtree(repo_path)

            # Parse GitHub URL
            owner, repo_name = parse_github_url(normalized_url)

            # Initialize GitHub client
            github_client = GithubClient(
                github_token=github_token or settings.github_token
            )

            # Create GithubRepositoryReader
            reader = GithubRepositoryReader(
                github_client=github_client,
                owner=owner,
                repo=repo_name,
                use_parser=False,  # Get raw content
                verbose=True,
                filter_directories=(
                    ["node_modules", ".git", "__pycache__", "venv", ".venv",
                     "dist", "build", ".idea", ".vscode", "coverage"],
                    GithubRepositoryReader.FilterType.EXCLUDE
                ),
                filter_file_extensions=(
                    [ext for ext in settings.supported_extensions],
                    GithubRepositoryReader.FilterType.INCLUDE
                )
            )

            # Load documents from GitHub
            print(f"Loading repository {owner}/{repo_name} from GitHub...")
            try:
                documents = reader.load_data(branch="main")
            except Exception as branch_error:
                print(f"Failed to load from 'main' branch: {branch_error}")
                print("Trying 'master' branch...")
                try:
                    documents = reader.load_data(branch="master")
                except Exception as master_error:
                    raise RepositoryDownloadError(
                        f"Failed to load from both 'main' and 'master' branches. "
                        f"Main error: {branch_error}. Master error: {master_error}"
                    )

            if not documents:
                raise RepositoryDownloadError(
                    f"No documents loaded from repository {owner}/{repo_name}"
                )

            # Save documents to disk
            self._save_documents_to_disk(documents, repo_path)

            print(f"Successfully loaded {len(documents)} documents from {owner}/{repo_name}")

            return str(repo_path), documents

        except RepositoryDownloadError:
            raise
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Full error traceback:\n{error_details}")
            raise RepositoryDownloadError(f"Failed to load repository: {str(e)}")

    def _save_documents_to_disk(self, documents: List[Document], save_path: Path) -> None:
        """Save documents to disk."""
        save_path.mkdir(parents=True, exist_ok=True)

        documents_file = save_path / "documents.pkl"
        with open(documents_file, 'wb') as f:
            pickle.dump(documents, f)

        # Save metadata
        metadata_file = save_path / "metadata.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"Total documents: {len(documents)}\n")
            f.write("=" * 80 + "\n\n")
            for i, doc in enumerate(documents):
                f.write(f"Document {i + 1}:\n")
                f.write(f"  File: {doc.metadata.get('file_path', 'Unknown')}\n")
                f.write(f"  Size: {len(doc.text)} characters\n")
                f.write("-" * 80 + "\n")

    def _load_documents_from_disk(self, repo_path: Path) -> Optional[List[Document]]:
        """Load documents from disk."""
        documents_file = repo_path / "documents.pkl"
        if not documents_file.exists():
            return None

        try:
            with open(documents_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load documents from {documents_file}: {e}")
            return None

    # ==================== Index Management ====================

    def create_index(
        self,
        repo_url: str,
        collection_name: str,
        documents: Optional[List[Document]] = None
    ) -> VectorStoreIndex:
        """
        Create a new vector index from documents.

        Args:
            repo_url: Repository URL
            collection_name: ChromaDB collection name
            documents: Optional documents to index (if None, loads from disk)

        Returns:
            Created VectorStoreIndex
        """
        # Load documents if not provided
        if documents is None:
            repo_identifier = get_repo_identifier(repo_url)
            repo_path = Path(settings.repos_dir) / repo_identifier
            documents = self._load_documents_from_disk(repo_path)
            if not documents:
                raise ValueError(f"No documents found for {repo_url}")

        # Create ChromaDB collection
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build index from documents
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        # Cache index
        self.indexes[collection_name] = index

        return index

    def load_index(self, collection_name: str) -> VectorStoreIndex:
        """
        Load an existing index.

        Args:
            collection_name: Collection name

        Returns:
            Loaded VectorStoreIndex
        """
        # Check cache first
        if collection_name in self.indexes:
            return self.indexes[collection_name]

        # Load from ChromaDB
        chroma_collection = self.chroma_client.get_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        index = VectorStoreIndex.from_vector_store(vector_store)

        # Cache index
        self.indexes[collection_name] = index

        return index

    def index_exists(self, collection_name: str) -> bool:
        """Check if index exists."""
        try:
            self.chroma_client.get_collection(name=collection_name)
            return True
        except:
            return False

    def delete_index(self, collection_name: str) -> None:
        """Delete an index."""
        try:
            self.chroma_client.delete_collection(name=collection_name)
            if collection_name in self.indexes:
                del self.indexes[collection_name]
        except Exception as e:
            print(f"Warning: Could not delete collection {collection_name}: {e}")

    # ==================== Document Management ====================

    def add_documents(self, collection_name: str, documents: List[Document]) -> None:
        """
        Add documents to an existing index.

        Args:
            collection_name: Collection name
            documents: List of LlamaIndex Documents to add
        """
        if not documents:
            return

        # Get or create index
        if self.index_exists(collection_name):
            index = self.load_index(collection_name)
        else:
            # Create new index
            chroma_collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                [],
                storage_context=storage_context
            )
            self.indexes[collection_name] = index

        # Insert documents into index
        for doc in documents:
            index.insert(doc)

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        Search for relevant documents using semantic search.

        Args:
            collection_name: Collection name
            query: Search query
            top_k: Number of results

        Returns:
            List of NodeWithScore objects from LlamaIndex
        """
        if top_k is None:
            top_k = settings.top_k_results

        # Load index
        index = self.load_index(collection_name)

        # Query index
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        return nodes

    # ==================== Question Answering ====================

    def answer_question(
        self,
        collection_name: str,
        question: str,
        top_k: Optional[int] = None
    ) -> str:
        """
        Answer a question using RAG with LlamaIndex.

        Args:
            collection_name: Collection to search
            question: User's question
            top_k: Number of context chunks

        Returns:
            Generated answer
        """
        if top_k is None:
            top_k = settings.top_k_results

        # Load index
        index = self.load_index(collection_name)

        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact"
        )

        # Query
        response = query_engine.query(question)

        return str(response)

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection statistics.

        Args:
            collection_name: Collection name

        Returns:
            Statistics dictionary
        """
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            return {
                "total_chunks": collection.count(),
                "collection_name": collection_name
            }
        except:
            return {
                "total_chunks": 0,
                "collection_name": collection_name
            }
