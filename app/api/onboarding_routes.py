"""
FastAPI routes for repository onboarding (full RAG workflow).
"""

from fastapi import APIRouter, HTTPException, status
from typing import List

from app.api.onboarding_schemas import (
    OnboardingRequest,
    OnboardingResponse,
    OnboardingStatusResponse,
    OnboardingListResponse,
    ErrorResponse
)
from app.services.onboarding_service import RepositoryOnboardingService
from app.core.exceptions import OnboardingError
from app.utils.github_utils import validate_github_url


router = APIRouter(prefix="/api/v1/onboarding", tags=["onboarding"])

# Service instance (singleton)
onboarding_service = RepositoryOnboardingService()


@router.post(
    "",
    response_model=OnboardingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start repository onboarding",
    description="Initiates full repository onboarding workflow: clone → parse → index → generate AI overview. Returns job ID for tracking.",
    responses={
        202: {"description": "Onboarding job initiated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid repository URL"},
    }
)
async def start_onboarding(request: OnboardingRequest):
    """
    Start full repository onboarding workflow.

    This endpoint:
    1. Clones the repository
    2. Filters and parses code files
    3. Generates embeddings
    4. Indexes to ChromaDB vector store
    5. Generates AI-powered project overview using LLM

    The process runs asynchronously in the background.
    Use the returned job_id to track progress.
    """
    repo_url = str(request.repo_url)

    # Validate GitHub URL
    if not validate_github_url(repo_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid GitHub repository URL. Expected format: https://github.com/owner/repo"
        )

    try:
        # Create job
        job = onboarding_service.create_job(
            repo_url=repo_url,
            force_reclone=request.force_reclone
        )

        # Start processing
        onboarding_service.start_job(job.job_id)

        return OnboardingResponse(
            job_id=job.job_id,
            status=job.current_state,
            repo_url=job.repo_url,
            collection_name=job.collection_name,
            message="Repository onboarding started in background",
            progress_percent=job.calculate_progress_percent()
        )

    except OnboardingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/{job_id}",
    response_model=OnboardingStatusResponse,
    summary="Get onboarding job status",
    description="Retrieve detailed status of an onboarding job including progress, metrics, and any errors.",
    responses={
        200: {"description": "Job status retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    }
)
async def get_onboarding_status(job_id: str):
    """
    Get the status of an onboarding job.

    Returns comprehensive information including:
    - Current state (created, cloning, parsing, completed, failed)
    - Progress percentage
    - File and chunk counts
    - Languages detected
    - Error details (if failed)
    - State transition history
    """
    job = onboarding_service.get_job_status(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found"
        )

    return OnboardingStatusResponse(
        job_id=job.job_id,
        status=job.current_state,
        repo_url=job.repo_url,
        collection_name=job.collection_name,
        progress_percent=job.calculate_progress_percent(),
        current_state=job.current_state,
        clone_path=job.clone_path,
        total_files=job.total_files,
        total_chunks=job.total_chunks,
        languages_detected=job.languages_detected,
        failed_files=job.failed_files,
        retry_count=job.retry_count,
        max_retries=job.max_retries,
        error=job.error,
        project_overview=job.project_overview,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        state_metrics=job.state_metrics
    )


@router.get(
    "/{job_id}/overview",
    response_model=dict,
    summary="Get project overview",
    description="Retrieve the AI-generated project overview for a completed onboarding job.",
    responses={
        200: {"description": "Project overview retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        400: {"model": ErrorResponse, "description": "Overview not yet available"},
    }
)
async def get_project_overview(job_id: str):
    """
    Get the AI-generated project overview for a repository.

    The overview is only available after the onboarding job has completed
    the GENERATING_OVERVIEW state. It includes:
    - Project purpose and description
    - Technology stack
    - Key modules and components
    - Entry points
    - Architecture overview
    """
    job = onboarding_service.get_job_status(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found"
        )

    if not job.project_overview:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Project overview not yet available. Current state: {job.current_state}"
        )

    return {
        "job_id": job.job_id,
        "repo_url": job.repo_url,
        "overview": job.project_overview,
        "generated_at": job.updated_at
    }


@router.get(
    "",
    response_model=OnboardingListResponse,
    summary="List all onboarding jobs",
    description="Get a list of all onboarding jobs with their current status.",
    responses={
        200: {"description": "List of onboarding jobs"},
    }
)
async def list_onboarding_jobs():
    """
    List all onboarding jobs.

    Returns a summary of all jobs including their current state,
    progress, and basic metadata.
    """
    jobs = onboarding_service.list_jobs()

    job_summaries = [
        {
            "job_id": job.job_id,
            "repo_url": job.repo_url,
            "status": job.current_state,
            "collection_name": job.collection_name,
            "progress_percent": job.calculate_progress_percent(),
            "total_chunks": job.total_chunks,
            "languages_detected": job.languages_detected,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "error": job.error
        }
        for job in jobs
    ]

    return OnboardingListResponse(
        total=len(job_summaries),
        jobs=job_summaries
    )
