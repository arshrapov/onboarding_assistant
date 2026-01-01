"""
Pydantic schemas for onboarding API request/response models.
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class OnboardingRequest(BaseModel):
    """Request model for starting repository onboarding."""

    repo_url: HttpUrl = Field(
        ...,
        description="GitHub repository URL (e.g., https://github.com/owner/repo)",
        examples=["https://github.com/facebook/react"]
    )

    force_reclone: bool = Field(
        default=False,
        description="If True, remove existing clone and re-clone the repository"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "repo_url": "https://github.com/facebook/react",
                "force_reclone": False
            }
        }


class OnboardingResponse(BaseModel):
    """Response model for onboarding job initiation."""

    job_id: str = Field(..., description="Unique job identifier for tracking onboarding progress")
    status: str = Field(..., description="Current job status (created, cloning, parsing, completed, failed)")
    repo_url: str = Field(..., description="Repository URL being onboarded")
    collection_name: str = Field(..., description="ChromaDB collection name for this repository")
    message: str = Field(..., description="Human-readable status message")
    progress_percent: int = Field(..., description="Overall progress percentage (0-100)", ge=0, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "created",
                "repo_url": "https://github.com/facebook/react",
                "collection_name": "onboarding_facebook_react",
                "message": "Repository onboarding started in background",
                "progress_percent": 0
            }
        }


class OnboardingStatusResponse(BaseModel):
    """Response model for detailed onboarding job status."""

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    repo_url: str = Field(..., description="Repository URL")
    collection_name: str = Field(..., description="ChromaDB collection name")
    progress_percent: int = Field(..., description="Overall progress percentage (0-100)", ge=0, le=100)
    current_state: str = Field(..., description="Current workflow state")

    # Paths and counts
    clone_path: Optional[str] = Field(None, description="Local path where repository is cloned")
    total_files: int = Field(default=0, description="Total number of files processed")
    total_chunks: int = Field(default=0, description="Total number of code chunks indexed")

    # Languages and errors
    languages_detected: List[str] = Field(default_factory=list, description="Programming languages detected")
    failed_files: List[str] = Field(default_factory=list, description="List of files that failed to parse")

    # Retry info
    retry_count: int = Field(default=0, description="Number of retry attempts made")
    max_retries: int = Field(default=3, description="Maximum retry attempts allowed")

    # Error tracking
    error: Optional[str] = Field(None, description="Error message if job failed")

    # Project overview
    project_overview: Optional[str] = Field(None, description="AI-generated project overview (available after completion)")

    # Timestamps
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")

    # State metrics
    state_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed metrics for each state"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "parsing",
                "repo_url": "https://github.com/facebook/react",
                "collection_name": "onboarding_facebook_react",
                "progress_percent": 65,
                "current_state": "parsing",
                "clone_path": "/app/data/repos/facebook-react",
                "total_files": 1234,
                "total_chunks": 5678,
                "languages_detected": ["javascript", "typescript", "markdown"],
                "failed_files": [],
                "retry_count": 0,
                "max_retries": 3,
                "error": None,
                "project_overview": None,
                "created_at": "2026-01-01T12:00:00Z",
                "updated_at": "2026-01-01T12:05:30Z",
                "completed_at": None,
                "state_metrics": {}
            }
        }


class OnboardingJobSummary(BaseModel):
    """Summary model for a single onboarding job."""

    job_id: str
    repo_url: str
    status: str
    collection_name: str
    progress_percent: int
    total_chunks: int
    languages_detected: List[str]
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None


class OnboardingListResponse(BaseModel):
    """Response model for listing all onboarding jobs."""

    total: int = Field(..., description="Total number of jobs")
    jobs: List[Dict[str, Any]] = Field(..., description="List of job summaries")

    class Config:
        json_schema_extra = {
            "example": {
                "total": 2,
                "jobs": [
                    {
                        "job_id": "550e8400-e29b-41d4-a716-446655440000",
                        "repo_url": "https://github.com/facebook/react",
                        "status": "completed",
                        "collection_name": "onboarding_facebook_react",
                        "progress_percent": 100,
                        "total_chunks": 5678,
                        "languages_detected": ["javascript", "typescript"],
                        "created_at": "2026-01-01T12:00:00Z",
                        "updated_at": "2026-01-01T12:10:00Z",
                        "error": None
                    },
                    {
                        "job_id": "660e8400-e29b-41d4-a716-446655440001",
                        "repo_url": "https://github.com/vuejs/vue",
                        "status": "parsing",
                        "collection_name": "onboarding_vuejs_vue",
                        "progress_percent": 45,
                        "total_chunks": 2345,
                        "languages_detected": ["javascript", "typescript"],
                        "created_at": "2026-01-01T13:00:00Z",
                        "updated_at": "2026-01-01T13:02:30Z",
                        "error": None
                    }
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""

    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid GitHub repository URL",
                "error_code": "INVALID_URL"
            }
        }