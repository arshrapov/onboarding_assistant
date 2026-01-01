"""
Service for downloading Git repositories.
"""

import shutil
from pathlib import Path
import git
from git.exc import GitCommandError

from app.config import settings
from app.core.exceptions import RepositoryDownloadError
from app.utils.github_utils import get_repo_identifier, normalize_github_url


class RepositoryDownloader:
    """Service for downloading Git repositories."""

    def __init__(self):
        """Initialize repository downloader."""
        self.repos_dir = Path(settings.repos_dir)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure the required directories exist."""
        self.repos_dir.mkdir(parents=True, exist_ok=True)

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

    def clone_repository(self, repo_url: str, force: bool = False) -> str:
        """
        Clone a Git repository.

        Args:
            repo_url: GitHub repository URL
            force: If True, remove existing directory and re-clone

        Returns:
            Local path to cloned repository

        Raises:
            RepositoryDownloadError: If cloning fails
        """
        # Normalize URL
        try:
            normalized_url = normalize_github_url(repo_url)
        except Exception as e:
            raise RepositoryDownloadError(f"Invalid repository URL: {str(e)}")

        # Get clone path
        clone_path = self.get_clone_path(normalized_url)

        try:
            # Check if repository already exists
            if clone_path.exists():
                if force:
                    shutil.rmtree(clone_path)
                else:
                    # Repository already exists, return path
                    return str(clone_path)

            # Clone repository
            git.Repo.clone_from(
                normalized_url,
                clone_path,
                depth=1,  # Shallow clone for faster download
                single_branch=True  # Only clone default branch
            )

            return str(clone_path)

        except GitCommandError as e:
            raise RepositoryDownloadError(f"Git clone failed: {str(e)}")

        except Exception as e:
            raise RepositoryDownloadError(f"Unexpected error during cloning: {str(e)}")
