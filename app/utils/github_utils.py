"""
Utility functions for GitHub repository operations.
"""

import re
from typing import Optional, Tuple
from urllib.parse import urlparse


class GitHubURLError(Exception):
    """Custom exception for GitHub URL validation errors."""
    pass


def parse_github_url(url: str) -> Tuple[str, str]:
    """
    Parse GitHub repository URL and extract owner and repository name.

    Args:
        url: GitHub repository URL (HTTP/HTTPS or SSH format)

    Returns:
        Tuple of (owner, repo_name)

    Raises:
        GitHubURLError: If URL is invalid or not a GitHub URL

    Examples:
        >>> parse_github_url("https://github.com/facebook/react")
        ('facebook', 'react')
        >>> parse_github_url("git@github.com:facebook/react.git")
        ('facebook', 'react')
    """
    if not url:
        raise GitHubURLError("URL cannot be empty")

    # Remove trailing slashes and .git extension
    url = url.rstrip('/').rstrip('.git')

    # Pattern for HTTPS URLs: https://github.com/owner/repo
    https_pattern = r'https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?$'

    # Pattern for SSH URLs: git@github.com:owner/repo
    ssh_pattern = r'git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$'

    # Try HTTPS pattern
    match = re.match(https_pattern, url)
    if match:
        owner, repo = match.groups()
        return owner, repo

    # Try SSH pattern
    match = re.match(ssh_pattern, url)
    if match:
        owner, repo = match.groups()
        return owner, repo

    raise GitHubURLError(
        f"Invalid GitHub URL format: {url}. "
        "Expected format: https://github.com/owner/repo or git@github.com:owner/repo"
    )


def validate_github_url(url: str) -> bool:
    """
    Validate if the URL is a valid GitHub repository URL.

    Args:
        url: GitHub repository URL

    Returns:
        True if valid, False otherwise
    """
    try:
        parse_github_url(url)
        return True
    except GitHubURLError:
        return False


def get_repo_identifier(url: str) -> str:
    """
    Get a filesystem-safe identifier for the repository.

    Args:
        url: GitHub repository URL

    Returns:
        String identifier in format "owner-repo"

    Raises:
        GitHubURLError: If URL is invalid

    Examples:
        >>> get_repo_identifier("https://github.com/facebook/react")
        'facebook-react'
    """
    owner, repo = parse_github_url(url)
    return f"{owner}-{repo}"


def is_github_url(url: str) -> bool:
    """
    Check if the URL is a GitHub URL (basic check).

    Args:
        url: URL to check

    Returns:
        True if URL contains github.com, False otherwise
    """
    try:
        parsed = urlparse(url)
        return 'github.com' in parsed.netloc.lower()
    except Exception:
        return False


def normalize_github_url(url: str) -> str:
    """
    Normalize GitHub URL to HTTPS format.

    Args:
        url: GitHub repository URL (any format)

    Returns:
        Normalized HTTPS URL

    Raises:
        GitHubURLError: If URL is invalid

    Examples:
        >>> normalize_github_url("git@github.com:facebook/react.git")
        'https://github.com/facebook/react'
    """
    owner, repo = parse_github_url(url)
    return f"https://github.com/{owner}/{repo}"