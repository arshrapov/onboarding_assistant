"""
FastAPI routes for repository management.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from typing import List

from app.api.schemas import (
    RepositoryCloneRequest,
    RepositoryCloneResponse,
    RepositoryStatusResponse,
    ErrorResponse
)
from app.services.repo_downloader import repo_downloader, RepositoryDownloadError
from app.utils.github_utils import validate_github_url
import uuid


router = APIRouter(prefix="/api/repositories", tags=["repositories"])


def clone_repository_background(job_id: str, repo_url: str):
    """
    Background task to clone repository.

    Args:
        job_id: Job identifier
        repo_url: Repository URL to clone
    """
    try:
        repo_downloader.clone_repository(repo_url, job_id=job_id)
    except RepositoryDownloadError as e:
        # Error is already stored in the job
        pass


@router.post(
    "/clone",
    response_model=RepositoryCloneResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Clone a GitHub repository",
    description="Initiates cloning of a GitHub repository in the background. Returns a job ID for tracking progress.",
    responses={
        202: {"description": "Clone job initiated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid repository URL"},
    }
)
async def clone_repository(
    request: RepositoryCloneRequest,
    background_tasks: BackgroundTasks
):
    """
    Clone a GitHub repository.

    The repository will be cloned in the background. Use the returned job_id
    to check the status of the cloning operation.
    """
    repo_url = str(request.repo_url)

    # Validate GitHub URL
    if not validate_github_url(repo_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid GitHub repository URL. Expected format: https://github.com/owner/repo"
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Add background task
    background_tasks.add_task(clone_repository_background, job_id, repo_url)

    # Create initial job entry
    from app.services.repo_downloader import CloneJob
    job = CloneJob(job_id, repo_url)
    repo_downloader.jobs[job_id] = job

    return RepositoryCloneResponse(
        job_id=job_id,
        status="pending",
        repo_url=repo_url,
        message="Repository cloning started in background"
    )


@router.get(
    "/{job_id}/status",
    response_model=RepositoryStatusResponse,
    summary="Get clone job status",
    description="Retrieve the current status of a repository clone job.",
    responses={
        200: {"description": "Job status retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    }
)
async def get_clone_status(job_id: str):
    """
    Get the status of a repository clone job.

    Returns detailed information about the clone operation, including
    status, local path (if completed), and any error messages.
    """
    job = repo_downloader.get_job_status(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found"
        )

    return RepositoryStatusResponse(**job.to_dict())


@router.get(
    "/",
    response_model=List[str],
    summary="List cloned repositories",
    description="Get a list of all repositories that have been cloned to the local system.",
    responses={
        200: {"description": "List of cloned repositories"},
    }
)
async def list_repositories():
    """
    List all cloned repositories.

    Returns a list of repository directory names that have been cloned
    to the local file system.
    """
    return repo_downloader.list_cloned_repositories()


@router.delete(
    "/cleanup",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cleanup a cloned repository",
    description="Remove a cloned repository from the local file system.",
    responses={
        204: {"description": "Repository removed successfully"},
        404: {"model": ErrorResponse, "description": "Repository not found"},
    }
)
async def cleanup_repository(repo_url: str):
    """
    Remove a cloned repository from disk.

    Args:
        repo_url: The GitHub repository URL to remove
    """
    if not validate_github_url(repo_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid GitHub repository URL"
        )

    success = repo_downloader.cleanup_repository(repo_url)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found or already removed"
        )

    return None