"""
Service for downloading and managing Git repositories.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import git
from git.exc import GitCommandError

from app.config import settings
from app.utils.github_utils import parse_github_url, get_repo_identifier, normalize_github_url


class RepositoryDownloadError(Exception):
    """Custom exception for repository download errors."""
    pass


class CloneJob:
    """Represents a repository clone job."""

    def __init__(self, job_id: str, repo_url: str):
        self.job_id = job_id
        self.repo_url = repo_url
        self.status = "pending"
        self.local_path: Optional[str] = None
        self.error: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "repo_url": self.repo_url,
            "status": self.status,
            "local_path": self.local_path,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CloneJob":
        """Create CloneJob from dictionary."""
        job = cls(data["job_id"], data["repo_url"])
        job.status = data["status"]
        job.local_path = data.get("local_path")
        job.error = data.get("error")
        job.created_at = datetime.fromisoformat(data["created_at"])
        job.updated_at = datetime.fromisoformat(data["updated_at"])
        return job


class RepositoryDownloader:
    """Service for downloading Git repositories."""

    def __init__(self):
        """Initialize repository downloader."""
        self.repos_dir = Path(settings.repos_dir)
        self.cache_dir = Path(settings.cache_dir)
        self.jobs_file = self.cache_dir / "clone_jobs.json"
        self.jobs: Dict[str, CloneJob] = {}
        self._ensure_dirs()
        self._load_jobs()

    def _ensure_dirs(self) -> None:
        """Ensure the required directories exist."""
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_jobs(self) -> None:
        """Load jobs from JSON file."""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.jobs = {
                        job_id: CloneJob.from_dict(job_data)
                        for job_id, job_data in data.items()
                    }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # If file is corrupted, start fresh
                print(f"Warning: Could not load jobs file: {e}. Starting with empty jobs.")
                self.jobs = {}

    def _save_jobs(self) -> None:
        """Save jobs to JSON file."""
        try:
            data = {
                job_id: job.to_dict()
                for job_id, job in self.jobs.items()
            }
            with open(self.jobs_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving jobs to file: {e}")

    def _update_job(self, job: CloneJob) -> None:
        """Update job in memory and persist to disk."""
        job.updated_at = datetime.utcnow()
        self.jobs[job.job_id] = job
        self._save_jobs()

    def get_clone_path(self, repo_url: str) -> Path:
        """
        Get the local path where repository should be cloned.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Path object for the clone destination

        Raises:
            RepositoryDownloadError: If URL is invalid
        """
        try:
            repo_identifier = get_repo_identifier(repo_url)
            return self.repos_dir / repo_identifier
        except Exception as e:
            raise RepositoryDownloadError(f"Failed to determine clone path: {str(e)}")

    def clone_repository(
        self,
        repo_url: str,
        job_id: Optional[str] = None,
        force: bool = False
    ) -> CloneJob:
        """
        Clone a Git repository.

        Args:
            repo_url: GitHub repository URL
            job_id: Optional job ID (will be generated if not provided)
            force: If True, remove existing directory and re-clone

        Returns:
            CloneJob object with job details

        Raises:
            RepositoryDownloadError: If cloning fails
        """
        # Generate job ID if not provided
        if job_id is None:
            job_id = str(uuid.uuid4())

        # Normalize URL
        try:
            normalized_url = normalize_github_url(repo_url)
        except Exception as e:
            raise RepositoryDownloadError(f"Invalid repository URL: {str(e)}")

        # Create job
        job = CloneJob(job_id, normalized_url)
        self._update_job(job)

        # Get clone path
        clone_path = self.get_clone_path(normalized_url)

        try:
            # Check if repository already exists
            if clone_path.exists():
                if force:
                    shutil.rmtree(clone_path)
                else:
                    # Repository already exists, mark as completed
                    job.status = "completed"
                    job.local_path = str(clone_path)
                    self._update_job(job)
                    return job

            # Update job status
            job.status = "cloning"
            self._update_job(job)

            # Clone repository
            git.Repo.clone_from(
                normalized_url,
                clone_path,
                depth=1,  # Shallow clone for faster download
                single_branch=True  # Only clone default branch
            )

            # Update job status
            job.status = "completed"
            job.local_path = str(clone_path)
            self._update_job(job)

        except GitCommandError as e:
            job.status = "failed"
            job.error = f"Git clone failed: {str(e)}"
            self._update_job(job)
            raise RepositoryDownloadError(job.error)

        except Exception as e:
            job.status = "failed"
            job.error = f"Unexpected error during cloning: {str(e)}"
            self._update_job(job)
            raise RepositoryDownloadError(job.error)

        return job

    def get_job_status(self, job_id: str) -> Optional[CloneJob]:
        """
        Get the status of a clone job.

        Args:
            job_id: Job identifier

        Returns:
            CloneJob object or None if not found
        """
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> Dict[str, CloneJob]:
        """
        Get all clone jobs.

        Returns:
            Dictionary of all jobs
        """
        return self.jobs.copy()

    def cleanup_repository(self, repo_url: str) -> bool:
        """
        Remove a cloned repository from disk.

        Args:
            repo_url: GitHub repository URL

        Returns:
            True if removed successfully, False otherwise
        """
        try:
            clone_path = self.get_clone_path(repo_url)
            if clone_path.exists():
                shutil.rmtree(clone_path)
                return True
            return False
        except Exception:
            return False

    def list_cloned_repositories(self) -> list[str]:
        """
        List all cloned repositories.

        Returns:
            List of repository directory names
        """
        if not self.repos_dir.exists():
            return []

        return [
            d.name for d in self.repos_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]


# Global repository downloader instance
repo_downloader = RepositoryDownloader()