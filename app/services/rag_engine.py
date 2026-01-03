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
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.storage.docstore import SimpleDocumentStore
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

        # Configure text splitter for chunking
        Settings.text_splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
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

        # IMPORTANT: Prepend file path to document text to make files searchable by name
        # This allows both vector search and BM25 to find files by their path/name
        for doc in documents:
            file_path = doc.metadata.get('file_path', '')
            file_name = doc.metadata.get('file_name', '')
            if file_path and not doc.text.startswith(f"FILE: {file_path}"):
                # Add file path header to the beginning of the document
                doc.text = f"FILE: {file_path}\nFILENAME: {file_name}\n\n{doc.text}"

        docstore = SimpleDocumentStore()
        docstore.add_documents(documents)

        # Create ChromaDB collection
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)

        # Build index from documents
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        # Save docstore to disk for later use with BM25Retriever
        docstore_path = Path(settings.indexes_dir) / f"{collection_name}_docstore.pkl"
        try:
            with open(docstore_path, 'wb') as f:
                pickle.dump(docstore, f)
        except Exception as e:
            print(f"Warning: Could not save docstore: {e}")

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

        # Load docstore if it exists
        docstore_path = Path(settings.indexes_dir) / f"{collection_name}_docstore.pkl"
        docstore = None
        if docstore_path.exists():
            try:
                with open(docstore_path, 'rb') as f:
                    docstore = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load docstore: {e}")
                docstore = None

        # Create index from vector store
        if docstore is not None:
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=docstore
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context
            )
            # Manually attach docstore since from_vector_store doesn't preserve it
            index._docstore = docstore
        else:
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

            # Delete docstore file if it exists
            docstore_path = Path(settings.indexes_dir) / f"{collection_name}_docstore.pkl"
            if docstore_path.exists():
                docstore_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete collection {collection_name}: {e}")

    # ==================== Overview Metadata Collection ====================

    def collect_overview_metadata(
        self,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Collect metadata for overview generation from documents.

        Args:
            documents: List of LlamaIndex Document objects

        Returns:
            Dictionary with collected metadata for overview generation
        """
        import json
        from collections import Counter

        overview_data = {
            "stats": {
                "total_files": 0,
                "total_size_bytes": 0,
                "avg_file_size": 0.0,
                "languages": {},
            },
            "special_files": {},
            "patterns": {
                "frameworks": set(),
                "entry_points": [],
                "has_tests": False,
                "has_docker": False,
                "has_ci_cd": False,
                "project_type": "unknown",
            },
            "package_info": {
                "name": None,
                "version": None,
                "description": None,
                "license": None,
                "dependencies_count": 0,
                "top_dependencies": [],
            }
        }

        # Special files to collect (case-insensitive)
        special_files_set = {
            # Documentation
            'readme.md', 'readme.rst', 'readme.txt', 'readme',
            'contributing.md', 'changelog.md', 'history.md',
            # Package/dependency files
            'requirements.txt', 'pyproject.toml', 'setup.py', 'pipfile',
            'package.json', 'package-lock.json', 'yarn.lock',
            'cargo.toml', 'go.mod', 'go.sum',
            'pom.xml', 'build.gradle', 'build.gradle.kts',
            'gemfile', 'gemfile.lock',
            # Configuration files
            'dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
            '.dockerignore',
            'makefile', 'cmakelists.txt',
            '.env.example', 'config.yaml', 'config.yml', 'config.json',
            # CI/CD
            '.travis.yml', '.gitlab-ci.yml', 'jenkinsfile',
            'license', 'license.txt', 'license.md',
        }

        language_counter = Counter()
        total_size = 0

        for doc in documents:
            file_path = doc.metadata.get('file_path', '')
            file_name = Path(file_path).name.lower()
            language = doc.metadata.get('language', 'Unknown')
            content = doc.text
            content_size = len(content)

            # Update statistics
            overview_data["stats"]["total_files"] += 1
            total_size += content_size
            language_counter[language] += 1

            # Collect special files (limit depth and size)
            parts = Path(file_path).parts
            depth = len(parts)

            # Only collect files at root or first level, max 5000 chars
            if depth <= 2 and file_name in special_files_set and content_size <= 500_000:
                overview_data["special_files"][file_path] = content[:5000]

            # Detect patterns
            # Entry points
            if file_name in ['main.py', '__main__.py', 'app.py', 'index.js',
                           'main.go', 'main.rs', 'main.java', 'main.cpp']:
                overview_data["patterns"]["entry_points"].append(file_path)

            # Tests
            if 'test' in file_path.lower() or file_name.startswith('test_'):
                overview_data["patterns"]["has_tests"] = True

            # Docker
            if file_name in ['dockerfile', 'docker-compose.yml', 'docker-compose.yaml']:
                overview_data["patterns"]["has_docker"] = True

            # CI/CD
            if '.github/workflows' in file_path or file_name in ['.travis.yml', '.gitlab-ci.yml', 'jenkinsfile']:
                overview_data["patterns"]["has_ci_cd"] = True

            # Framework detection (from content)
            if language == 'Python':
                if 'from fastapi import' in content or 'import fastapi' in content:
                    overview_data["patterns"]["frameworks"].add('FastAPI')
                if 'from flask import' in content or 'import flask' in content:
                    overview_data["patterns"]["frameworks"].add('Flask')
                if 'from django' in content or 'import django' in content:
                    overview_data["patterns"]["frameworks"].add('Django')
            elif language == 'JavaScript' or language == 'TypeScript':
                if 'from react' in content or "from 'react'" in content or 'import React' in content:
                    overview_data["patterns"]["frameworks"].add('React')
                if 'from vue' in content or "from 'vue'" in content:
                    overview_data["patterns"]["frameworks"].add('Vue')
                if 'express()' in content or "require('express')" in content:
                    overview_data["patterns"]["frameworks"].add('Express')

            # Parse package.json
            if file_name == 'package.json':
                try:
                    pkg = json.loads(content)
                    overview_data["package_info"]["name"] = pkg.get("name")
                    overview_data["package_info"]["version"] = pkg.get("version")
                    overview_data["package_info"]["description"] = pkg.get("description")
                    overview_data["package_info"]["license"] = pkg.get("license")

                    deps = list(pkg.get("dependencies", {}).keys())
                    dev_deps = list(pkg.get("devDependencies", {}).keys())
                    all_deps = deps + dev_deps
                    overview_data["package_info"]["dependencies_count"] = len(all_deps)
                    overview_data["package_info"]["top_dependencies"] = all_deps[:15]
                except Exception as e:
                    print(f"Warning: Could not parse package.json: {e}")

            # Parse requirements.txt
            elif file_name == 'requirements.txt':
                try:
                    deps = [
                        line.strip().split('==')[0].split('>=')[0].split('<=')[0].split('[')[0]
                        for line in content.split('\n')
                        if line.strip() and not line.strip().startswith('#')
                    ]
                    overview_data["package_info"]["dependencies_count"] = len(deps)
                    overview_data["package_info"]["top_dependencies"] = deps[:15]
                except Exception as e:
                    print(f"Warning: Could not parse requirements.txt: {e}")

            # Parse pyproject.toml
            elif file_name == 'pyproject.toml':
                try:
                    # Simple regex-based parsing for common fields
                    if 'name = ' in content:
                        for line in content.split('\n'):
                            if line.strip().startswith('name ='):
                                overview_data["package_info"]["name"] = line.split('=')[1].strip().strip('"\'')
                            elif line.strip().startswith('version ='):
                                overview_data["package_info"]["version"] = line.split('=')[1].strip().strip('"\'')
                            elif line.strip().startswith('description ='):
                                overview_data["package_info"]["description"] = line.split('=')[1].strip().strip('"\'')
                except Exception as e:
                    print(f"Warning: Could not parse pyproject.toml: {e}")

        # Finalize statistics
        overview_data["stats"]["total_size_bytes"] = total_size
        if overview_data["stats"]["total_files"] > 0:
            overview_data["stats"]["avg_file_size"] = total_size / overview_data["stats"]["total_files"]

        # Convert language counter to dict
        overview_data["stats"]["languages"] = {
            lang: count for lang, count in language_counter.most_common()
        }

        # Convert frameworks set to list
        overview_data["patterns"]["frameworks"] = list(overview_data["patterns"]["frameworks"])

        # Detect project type
        has_setup_py = any('setup.py' in f for f in overview_data["special_files"].keys())
        has_init_py = any('__init__.py' in doc.metadata.get('file_path', '') for doc in documents)
        has_api_routes = any(
            'route' in doc.text.lower() or 'endpoint' in doc.text.lower() or '@app.' in doc.text
            for doc in documents[:100]  # Check first 100 docs for performance
        )
        has_main = len(overview_data["patterns"]["entry_points"]) > 0

        if has_api_routes or 'FastAPI' in overview_data["patterns"]["frameworks"] or 'Flask' in overview_data["patterns"]["frameworks"]:
            overview_data["patterns"]["project_type"] = "web_service"
        elif (has_setup_py or has_init_py) and not has_main:
            overview_data["patterns"]["project_type"] = "library"
        elif has_main:
            overview_data["patterns"]["project_type"] = "application"
        else:
            overview_data["patterns"]["project_type"] = "mixed"


        return overview_data

    # ==================== Document Management ====================

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
        project_overview: Optional[str] = None,
        top_k: Optional[int] = None,
        chat_history: Optional[List[tuple[str, str]]] = None
    ) -> str:
        """
        Answer a question using RAG with a multi-stage approach:
        1. Generate search query from question + project overview + chat history
        2. Retrieve relevant nodes from vector store
        3. Generate final answer using question + overview + chat history + retrieved context

        Args:
            collection_name: Collection to search
            question: User's question
            project_overview: Optional project overview for context
            top_k: Number of context chunks to retrieve
            chat_history: Optional list of (question, answer) tuples for conversation context

        Returns:
            Generated answer
        """
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever

        if top_k is None:
            top_k = settings.top_k_results

        # ==================== Stage 1: Generate search query ====================
        # Build chat history context if available
        chat_context = ""
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-3:]  # Use last 3 turns for context
            history_parts = []
            for q, a in recent_history:
                history_parts.append(f"Q: {q}\nA: {a[:200]}...")  # Truncate long answers
            chat_context = f"\n\nRECENT CONVERSATION:\n" + "\n".join(history_parts)

        # Create a prompt to generate an optimized search query using question + overview + chat history
        search_query_prompt = f"""
You are helping to search a codebase. Given the user's question, generate an optimized search query.

USER QUESTION:
{question}

PROJECT OVERVIEW:
{project_overview if project_overview else "No overview available"}{chat_context}

If the user is asking for a specific file (e.g., "show system_objects.py"), include the COMPLETE filename in your search query.
If asking about code concepts, focus on technical terms, function names, class names that would appear in relevant code.

Return ONLY the search query text, nothing else. Make it 1-2 sentences maximum.
"""

        response = Settings.llm.complete(search_query_prompt)
        search_query = response.text.strip()

        # Remove markdown code formatting if present
        if search_query.startswith('`') and search_query.endswith('`'):
            search_query = search_query[1:-1].strip()

        # ==================== Stage 2: Retrieve relevant nodes ====================
        # Load index
        index = self.load_index(collection_name)

        # Build list of retrievers - use BM25 only if docstore is available and has documents
        retrievers = [index.as_retriever(similarity_top_k=top_k)]

        # Check if docstore is available and has documents
        if hasattr(index, 'docstore') and index.docstore is not None:
            try:
                # Try to get documents from docstore to verify it's populated
                doc_ids = index.docstore.get_all_document_hashes()
                if doc_ids and len(doc_ids) > 0:
                    retrievers.append(
                        BM25Retriever.from_defaults(
                            docstore=index.docstore, similarity_top_k=top_k
                        )
                    )
            except Exception as e:
                print(f"Warning: Could not create BM25 retriever: {e}")

        # Create retriever (fusion if we have multiple, otherwise single)
        if len(retrievers) > 1:
            retriever = QueryFusionRetriever(
                retrievers,
                num_queries=1,
                use_async=True,
            )
        else:
            retriever = retrievers[0]

        # Retrieve nodes using the generated search query
        retrieved_nodes = retriever.retrieve(search_query)

        # ==================== Stage 3: Generate final answer ====================
        # Build context from retrieved nodes
        context_parts = []
        for idx, node in enumerate(retrieved_nodes, 1):
            node_text = node.text if hasattr(node, 'text') else str(node)
            file_path = node.metadata.get('file_path', 'Unknown') if hasattr(node, 'metadata') else 'Unknown'
            score = node.score if hasattr(node, 'score') else 'N/A'

            context_parts.append(f"""
--- Context {idx} (Score: {score}) ---
File: {file_path}
Content:
{node_text}
""")

        context_str = "\n".join(context_parts)

        # Build conversation history for final prompt
        conversation_context = ""
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-5:]  # Use last 5 turns for final answer
            history_parts = []
            for i, (q, a) in enumerate(recent_history, 1):
                history_parts.append(f"Turn {i}:\nUser: {q}\nAssistant: {a[:300]}...")  # Truncate long answers
            conversation_context = f"\n\nCONVERSATION HISTORY:\n" + "\n\n".join(history_parts) + "\n"

        # Create final prompt with all information
        final_prompt = f"""
You are an expert assistant helping developers understand a codebase.

PROJECT OVERVIEW:
{project_overview if project_overview else "No overview available"}{conversation_context}

USER QUESTION:
{question}

RELEVANT CODE CONTEXT:
{context_str}

Based on the project overview, conversation history (if any), and the relevant code context above, provide a detailed and accurate answer to the user's question.
If this is a follow-up question, consider the previous conversation to understand context and references (like "it", "that", "the previous one", etc.).
If the context doesn't contain enough information to fully answer the question, acknowledge this and provide the best answer you can with the available information.
Include specific file paths and code references when relevant.
"""

        final_response = Settings.llm.complete(final_prompt)

        return str(final_response.text)

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
