"""
File parsing utilities for extracting code chunks.
"""

from pathlib import Path
from typing import List
from abc import ABC, abstractmethod
import hashlib
import ast

from app.core.models import CodeChunk, ChunkType
from app.utils.file_filter import get_relative_path, get_language_from_extension


def generate_chunk_id(repo_url: str, file_path: str, start_line: int) -> str:
    """
    Generate unique ID for a code chunk.

    Args:
        repo_url: Repository URL
        file_path: Relative file path
        start_line: Starting line number

    Returns:
        Unique chunk ID
    """
    content = f"{repo_url}:{file_path}:{start_line}"
    print(content)
    return hashlib.md5(content.encode()).hexdigest()


class FileParser(ABC):
    """Abstract base class for language-specific file parsers."""

    def __init__(self, repo_url: str):
        """
        Initialize parser.

        Args:
            repo_url: Repository URL for chunk ID generation
        """
        self.repo_url = repo_url

    @abstractmethod
    def parse(self, file_path: Path, repo_path: Path) -> List[CodeChunk]:
        """
        Parse a file into code chunks.

        Args:
            file_path: Absolute path to file
            repo_path: Repository root path

        Returns:
            List of code chunks
        """
        pass

    def read_file_content(self, file_path: Path) -> str:
        """
        Safely read file content.

        Args:
            file_path: Path to file

        Returns:
            File content as string

        Raises:
            Exception: If file cannot be read
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def create_file_chunk(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> CodeChunk:
        """
        Create a file-level code chunk.

        Args:
            content: File content
            file_path: Relative file path
            language: Programming language

        Returns:
            CodeChunk representing the entire file
        """
        lines = content.splitlines()
        chunk_id = generate_chunk_id(self.repo_url, file_path, 1)

        return CodeChunk(
            id=chunk_id,
            content=content,
            chunk_type=ChunkType.FILE,
            file_path=file_path,
            start_line=1,
            end_line=len(lines) if lines else 1,
            language=language,
            signature=None,
            metadata={"lines": len(lines)}
        )


class PythonParser(FileParser):
    """Parser for Python files using AST."""

    def parse(self, file_path: Path, repo_path: Path) -> List[CodeChunk]:
        """
        Parse Python file into chunks (file + functions/classes/methods).

        Args:
            file_path: Absolute path to file
            repo_path: Repository root path

        Returns:
            List of code chunks
        """
        relative_path = get_relative_path(file_path, repo_path)

        try:
            content = self.read_file_content(file_path)
        except Exception:
            return []

        chunks = []

        # Always create a file-level chunk
        file_chunk = self.create_file_chunk(content, relative_path, 'python')
        chunks.append(file_chunk)

        # Try to parse AST for functions/classes
        try:
            tree = ast.parse(content)

            # Extract top-level functions and classes
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = self._extract_function_chunk(node, content, relative_path)
                    if chunk:
                        chunks.append(chunk)

                elif isinstance(node, ast.ClassDef):
                    # Extract the class itself
                    class_chunk = self._extract_class_chunk(node, content, relative_path)
                    if class_chunk:
                        chunks.append(class_chunk)

                    # Extract methods from the class
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_chunk = self._extract_method_chunk(
                                item, node.name, content, relative_path
                            )
                            if method_chunk:
                                chunks.append(method_chunk)

        except SyntaxError:
            # If AST parsing fails, just return the file chunk
            pass

        return chunks

    def _extract_function_chunk(
        self,
        node: ast.FunctionDef,
        content: str,
        file_path: str
    ) -> CodeChunk:
        """Extract a function as a code chunk."""
        lines = content.splitlines()
        start_line = node.lineno
        end_line = node.end_lineno if node.end_lineno else start_line

        # Extract function source
        function_lines = lines[start_line - 1:end_line]
        function_content = '\n'.join(function_lines)

        # Generate signature
        args = [arg.arg for arg in node.args.args]
        signature = f"def {node.name}({', '.join(args)})"

        chunk_id = generate_chunk_id(self.repo_url, file_path, start_line)

        return CodeChunk(
            id=chunk_id,
            content=function_content,
            chunk_type=ChunkType.FUNCTION,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            language='python',
            signature=signature,
            metadata={
                "name": node.name,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "num_args": len(args)
            }
        )

    def _extract_method_chunk(
        self,
        node: ast.FunctionDef,
        class_name: str,
        content: str,
        file_path: str
    ) -> CodeChunk:
        """Extract a class method as a code chunk."""
        lines = content.splitlines()
        start_line = node.lineno
        end_line = node.end_lineno if node.end_lineno else start_line

        # Extract method source
        method_lines = lines[start_line - 1:end_line]
        method_content = '\n'.join(method_lines)

        # Generate signature
        args = [arg.arg for arg in node.args.args]
        signature = f"{class_name}.{node.name}({', '.join(args)})"

        chunk_id = generate_chunk_id(self.repo_url, file_path, start_line)

        return CodeChunk(
            id=chunk_id,
            content=method_content,
            chunk_type=ChunkType.METHOD,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            language='python',
            signature=signature,
            metadata={
                "class_name": class_name,
                "method_name": node.name,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "num_args": len(args)
            }
        )

    def _extract_class_chunk(
        self,
        node: ast.ClassDef,
        content: str,
        file_path: str
    ) -> CodeChunk:
        """Extract a class as a code chunk."""
        lines = content.splitlines()
        start_line = node.lineno
        end_line = node.end_lineno if node.end_lineno else start_line

        # Extract class source
        class_lines = lines[start_line - 1:end_line]
        class_content = '\n'.join(class_lines)

        # Generate signature
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr)

        if bases:
            signature = f"class {node.name}({', '.join(bases)})"
        else:
            signature = f"class {node.name}"

        chunk_id = generate_chunk_id(self.repo_url, file_path, start_line)

        return CodeChunk(
            id=chunk_id,
            content=class_content,
            chunk_type=ChunkType.CLASS,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            language='python',
            signature=signature,
            metadata={
                "name": node.name,
                "bases": bases,
                "num_methods": sum(1 for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
            }
        )


class GenericParser(FileParser):
    """Generic parser for non-Python files - creates file-level and line-based segment chunks."""

    def __init__(self, repo_url: str, lines_per_segment: int = 50):
        """
        Initialize generic parser.

        Args:
            repo_url: Repository URL for chunk ID generation
            lines_per_segment: Number of lines per segment chunk (default: 50)
        """
        super().__init__(repo_url)
        self.lines_per_segment = lines_per_segment

    def parse(self, file_path: Path, repo_path: Path) -> List[CodeChunk]:
        """
        Parse file into a file-level chunk and line-based segments.

        Args:
            file_path: Absolute path to file
            repo_path: Repository root path

        Returns:
            List containing file-level chunk and segment chunks
        """
        language = get_language_from_extension(file_path.suffix)
        relative_path = get_relative_path(file_path, repo_path)

        try:
            content = self.read_file_content(file_path)
            chunks = []

            # TODO: should we keep file-level chunk for generic files?
            # Always create a file-level chunk
            # file_chunk = self.create_file_chunk(content, relative_path, language)
            # chunks.append(file_chunk)

            # Create segment chunks for every N lines
            segment_chunks = self._create_segment_chunks(content, relative_path, language)
            chunks.extend(segment_chunks)

            return chunks
        except Exception:
            return []

    def _create_segment_chunks(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[CodeChunk]:
        """
        Create segment chunks from file content.

        Args:
            content: File content
            file_path: Relative file path
            language: Programming language

        Returns:
            List of segment chunks
        """
        lines = content.splitlines()
        chunks = []

        # Skip if file is too small to segment
        if len(lines) <= self.lines_per_segment:
            return []

        # Create segments
        for start_idx in range(0, len(lines), self.lines_per_segment):
            end_idx = min(start_idx + self.lines_per_segment, len(lines))
            start_line = start_idx + 1
            end_line = end_idx

            # Extract segment content
            segment_lines = lines[start_idx:end_idx]
            segment_content = '\n'.join(segment_lines)

            # Generate unique ID for this segment
            chunk_id = generate_chunk_id(self.repo_url, file_path, start_line)

            chunk = CodeChunk(
                id=chunk_id,
                content=segment_content,
                chunk_type=ChunkType.SEGMENT,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                language=language,
                signature=None,
                metadata={
                    "lines": len(segment_lines),
                    "segment_index": start_idx // self.lines_per_segment
                }
            )
            chunks.append(chunk)

        return chunks


def create_parser(language: str, repo_url: str) -> FileParser:
    """
    Factory function to create appropriate parser for a language.

    Args:
        language: Programming language
        repo_url: Repository URL

    Returns:
        FileParser instance
    """
    return GenericParser(repo_url)

    # TODO: add more language-specific parsers here
    if language == 'python':
        return PythonParser(repo_url)


def parse_file(file_path: Path, repo_path: Path, repo_url: str) -> List[CodeChunk]:
    """
    Parse a file into code chunks using appropriate parser.

    Args:
        file_path: Absolute path to file
        repo_path: Repository root path
        repo_url: Repository URL

    Returns:
        List of code chunks (empty if parsing fails)
    """
    language = get_language_from_extension(file_path.suffix)
    parser = create_parser(language, repo_url)
    return parser.parse(file_path, repo_path)