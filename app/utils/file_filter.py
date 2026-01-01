"""
Utilities for filtering files during repository processing.
"""

from pathlib import Path
from typing import List, Set
import os

from app.config import settings


def get_supported_extensions() -> Set[str]:
    """
    Get set of supported file extensions from config.

    Returns:
        Set of file extensions (e.g., {'.py', '.js', '.ts'})
    """
    return set(settings.supported_extensions)


def should_process_file(file_path: Path) -> bool:
    """
    Determine if a file should be processed based on extension and size.

    Args:
        file_path: Path to the file

    Returns:
        True if file should be processed, False otherwise
    """
    # Check if it's a file
    if not file_path.is_file():
        return False

    # Check extension
    if file_path.suffix not in get_supported_extensions():
        return False

    # Check file size
    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > settings.max_file_size_mb:
            return False
    except OSError:
        return False

    return True


def get_language_from_extension(extension: str) -> str:
    """
    Map file extension to programming language.

    Args:
        extension: File extension (e.g., '.py', '.js')

    Returns:
        Language name (e.g., 'python', 'javascript')
    """
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.jsx': 'javascript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.rb': 'ruby',
        '.php': 'php',
        '.md': 'markdown',
        '.txt': 'text',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml'
    }

    return extension_map.get(extension.lower(), 'unknown')


def filter_repository_files(repo_path: Path) -> List[Path]:
    """
    Recursively find all processable files in a repository.

    Args:
        repo_path: Root path of the repository

    Returns:
        List of file paths that should be processed
    """
    processable_files = []

    # Directories to skip
    skip_dirs = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv',
        'build', 'dist', '.next', '.cache', 'target', 'bin', 'obj',
        '.idea', '.vscode', 'coverage', '.pytest_cache'
    }

    # Walk through directory tree
    for root, dirs, files in os.walk(repo_path):
        # Remove skip directories from traversal
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        # Check each file
        for file_name in files:
            file_path = Path(root) / file_name

            if should_process_file(file_path):
                processable_files.append(file_path)

    return processable_files


def get_relative_path(file_path: Path, repo_path: Path) -> str:
    """
    Get relative path of file from repository root.

    Args:
        file_path: Absolute path to file
        repo_path: Absolute path to repository root

    Returns:
        Relative path as string with forward slashes
    """
    try:
        relative = file_path.relative_to(repo_path)
        # Convert to forward slashes for consistency
        return str(relative).replace('\\', '/')
    except ValueError:
        # If file is not relative to repo, return the name
        return file_path.name


def count_files_by_language(file_paths: List[Path]) -> dict:
    """
    Count files grouped by programming language.

    Args:
        file_paths: List of file paths

    Returns:
        Dictionary mapping language to file count
    """
    language_counts = {}

    for file_path in file_paths:
        language = get_language_from_extension(file_path.suffix)
        language_counts[language] = language_counts.get(language, 0) + 1

    return language_counts