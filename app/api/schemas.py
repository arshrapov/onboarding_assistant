"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Literal
from datetime import datetime


class RepositoryCloneRequest(BaseModel):
    """Request model for cloning a repository."""

    repo_url: HttpUrl = Field(
        ...,
        description="GitHub repository URL (e.g., https://github.com/owner/repo)",
        examples=["https://github.com/facebook/react"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "repo_url": "https://github.com/facebook/react"
            }
        }


class RepositoryCloneResponse(BaseModel):
    """Response model for repository clone initiation."""

    job_id: str = Field(..., description="Unique job identifier for tracking clone progress")
    status: Literal["pending", "cloning"] = Field(..., description="Current job status")
    repo_url: str = Field(..., description="Repository URL being cloned")
    message: str = Field(..., description="Human-readable status message")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "repo_url": "https://github.com/facebook/react",
                "message": "Repository cloning started in background"
            }
        }


class RepositoryStatusResponse(BaseModel):
    """Response model for repository clone status."""

    job_id: str = Field(..., description="Job identifier")
    status: Literal["pending", "cloning", "completed", "failed"] = Field(
        ...,
        description="Current job status"
    )
    repo_url: str = Field(..., description="Repository URL")
    local_path: Optional[str] = Field(None, description="Local path where repository is cloned")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "repo_url": "https://github.com/facebook/react",
                "local_path": "/app/data/repos/facebook-react",
                "error": None,
                "created_at": "2026-01-01T12:00:00Z",
                "updated_at": "2026-01-01T12:05:30Z"
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