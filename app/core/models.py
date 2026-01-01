"""
Data models and abstractions for the Onboarding Assistant.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ChunkType(str, Enum):
    """Type of code chunk."""
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    SEGMENT = "segment"


class CodeChunk(BaseModel):
    """Represents a chunk of code with metadata."""

    id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="The actual code content")
    chunk_type: ChunkType = Field(..., description="Type of chunk")
    file_path: str = Field(..., description="Relative path to the file")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    language: Optional[str] = Field(None, description="Programming language")
    signature: Optional[str] = Field(None, description="Function/class signature")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True


class SearchResult(BaseModel):
    """Result from vector similarity search."""

    chunk: CodeChunk = Field(..., description="The code chunk")
    score: float = Field(..., description="Similarity score (0-1)")
    distance: float = Field(..., description="Distance metric from query")

    def __lt__(self, other):
        """Enable sorting by score (descending)."""
        return self.score > other.score


class RepositoryIndex(BaseModel):
    """Metadata for an indexed repository."""

    repo_url: str = Field(..., description="Git repository URL")
    repo_name: str = Field(..., description="Repository name")
    collection_name: str = Field(..., description="ChromaDB collection name")
    indexed_at: datetime = Field(default_factory=datetime.now, description="Indexing timestamp")
    total_chunks: int = Field(0, description="Number of chunks indexed")
    file_count: int = Field(0, description="Number of files indexed")
    languages: List[str] = Field(default_factory=list, description="Programming languages found")
    status: str = Field("active", description="Index status (active, archived, error)")


# Onboarding Job Models

class OnboardingState(str, Enum):
    """States in the repository onboarding workflow."""
    CREATED = "created"
    CLONING = "cloning"
    PARSING = "parsing"
    GENERATING_OVERVIEW = "generating_overview"
    COMPLETED = "completed"
    FAILED = "failed"


class StateTransition(BaseModel):
    """Record of a state transition."""
    from_state: OnboardingState
    to_state: OnboardingState
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    triggered_by: str = Field(default="system", description="What triggered this transition")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class StateMetrics(BaseModel):
    """Metrics for a specific state execution."""
    state: OnboardingState
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    attempts: int = 0
    data: Dict[str, Any] = Field(default_factory=dict, description="State-specific metrics")

    class Config:
        use_enum_values = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "state": self.state,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "attempts": self.attempts,
            "data": self.data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateMetrics":
        """Create from dictionary."""
        return cls(
            state=data["state"],
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            duration_seconds=data.get("duration_seconds"),
            attempts=data.get("attempts", 0),
            data=data.get("data", {})
        )


class OnboardingJob(BaseModel):
    """Repository onboarding job with state machine."""

    job_id: str = Field(..., description="Unique job identifier")
    repo_url: str = Field(..., description="Repository URL")

    # State machine
    current_state: OnboardingState = Field(default=OnboardingState.CREATED)
    state_history: List[StateTransition] = Field(default_factory=list)
    state_metrics: Dict[str, StateMetrics] = Field(default_factory=dict)

    # Retry logic
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    last_error_state: Optional[OnboardingState] = None

    # Results
    collection_name: Optional[str] = None
    clone_path: Optional[str] = None

    # Aggregate metrics
    total_files: int = 0
    total_chunks: int = 0
    languages_detected: List[str] = Field(default_factory=list)
    failed_files: List[str] = Field(default_factory=list)

    # Overview generation
    metadata_for_overview: Dict[str, str] = Field(default_factory=dict, description="File contents collected during parsing for overview generation")
    project_overview: Optional[str] = Field(None, description="Generated project overview")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Error tracking
    error: Optional[str] = None

    class Config:
        use_enum_values = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "repo_url": self.repo_url,
            "current_state": self.current_state,
            "state_history": [
                {
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "timestamp": t.timestamp.isoformat(),
                    "triggered_by": t.triggered_by,
                    "metadata": t.metadata
                }
                for t in self.state_history
            ],
            "state_metrics": {
                state: metrics.to_dict()
                for state, metrics in self.state_metrics.items()
            },
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_error_state": self.last_error_state,
            "collection_name": self.collection_name,
            "clone_path": self.clone_path,
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "languages_detected": self.languages_detected,
            "failed_files": self.failed_files,
            "metadata_for_overview": self.metadata_for_overview,
            "project_overview": self.project_overview,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OnboardingJob":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            repo_url=data["repo_url"],
            current_state=OnboardingState(data["current_state"]),
            state_history=[
                StateTransition(
                    from_state=OnboardingState(t["from_state"]),
                    to_state=OnboardingState(t["to_state"]),
                    timestamp=datetime.fromisoformat(t["timestamp"]),
                    triggered_by=t.get("triggered_by", "system"),
                    metadata=t.get("metadata", {})
                )
                for t in data.get("state_history", [])
            ],
            state_metrics={
                state: StateMetrics.from_dict(metrics)
                for state, metrics in data.get("state_metrics", {}).items()
            },
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            last_error_state=OnboardingState(data["last_error_state"]) if data.get("last_error_state") else None,
            collection_name=data.get("collection_name"),
            clone_path=data.get("clone_path"),
            total_files=data.get("total_files", 0),
            total_chunks=data.get("total_chunks", 0),
            languages_detected=data.get("languages_detected", []),
            failed_files=data.get("failed_files", []),
            metadata_for_overview=data.get("metadata_for_overview", {}),
            project_overview=data.get("project_overview"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error=data.get("error")
        )

    def calculate_progress_percent(self) -> int:
        """Calculate overall progress percentage (0-100)."""
        state_weights = {
            OnboardingState.CREATED: 0,
            OnboardingState.CLONING: 10,
            OnboardingState.PARSING: 20,
            OnboardingState.GENERATING_OVERVIEW: 90,
            OnboardingState.COMPLETED: 100,
            OnboardingState.FAILED: 0
        }

        base = state_weights.get(self.current_state, 0)

        # Add sub-progress for parsing state (parsing + indexing combined)
        if self.current_state == OnboardingState.PARSING and self.total_files > 0:
            metrics = self.state_metrics.get(OnboardingState.PARSING.value)
            if metrics and metrics.data.get("processed_files"):
                processed = metrics.data["processed_files"]
                file_progress = processed / self.total_files
                base += file_progress * 70  # 70% allocated to parsing/indexing

        return min(int(base), 100)


class VectorStore(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """
        Create a new collection for storing embeddings.

        Args:
            collection_name: Name of the collection
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection and all its data.

        Args:
            collection_name: Name of the collection to delete
        """
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists, False otherwise
        """
        pass

    @abstractmethod
    def add_documents(
        self,
        collection_name: str,
        chunks: List[CodeChunk]
    ) -> None:
        """
        Add documents to a collection.
        Embeddings are generated automatically by the vector store's embedding provider.

        Args:
            collection_name: Name of the collection
            chunks: List of code chunks to add
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents using vector similarity.
        Query embedding is generated automatically by the vector store's embedding provider.

        Args:
            collection_name: Name of the collection to search
            query_text: Query text (embedding will be generated automatically)
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results sorted by similarity
        """
        pass

    @abstractmethod
    def get_collection_count(self, collection_name: str) -> int:
        """
        Get the number of documents in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of documents in the collection
        """
        pass
