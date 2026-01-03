"""
Formatters for UI display - code highlighting and markdown formatting.
"""

import re
from typing import Optional
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer, TextLexer
from pygments.formatters import HtmlFormatter
from pygments.util import ClassNotFound


def format_code_block(code: str, language: Optional[str] = None) -> str:
    """
    Format code with syntax highlighting using Pygments.

    Args:
        code: Source code to highlight
        language: Programming language (e.g., 'python', 'javascript')

    Returns:
        HTML string with syntax-highlighted code
    """
    try:
        if language:
            try:
                lexer = get_lexer_by_name(language, stripall=True)
            except ClassNotFound:
                lexer = TextLexer()
        else:
            try:
                lexer = guess_lexer(code)
            except:
                lexer = TextLexer()

        formatter = HtmlFormatter(
            style='monokai',
            noclasses=True,
            linenos=False,
            cssclass='code-block'
        )
        highlighted = highlight(code, lexer, formatter)
        return highlighted
    except Exception as e:
        # Fallback to plain pre/code block
        return f"<pre><code>{_escape_html(code)}</code></pre>"


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def parse_file_references(text: str) -> str:
    """
    Parse file:line references and make them bold/highlighted.

    Supports patterns:
    - file_path:line_number (e.g., app/main.py:42)
    - file_path:line_start-line_end (e.g., src/utils.ts:10-20)

    Args:
        text: Text containing file references

    Returns:
        Formatted text with highlighted references
    """
    # Match patterns like "app/main.py:42" or "src/utils.ts:10-20"
    # Also match Windows paths like "app\\main.py:42"
    pattern = r'([a-zA-Z0-9_\-./\\]+\.[a-zA-Z]+):(\d+)(?:-(\d+))?'

    def replace_reference(match):
        file_path = match.group(1)
        line_start = match.group(2)
        line_end = match.group(3) or line_start

        if line_end == line_start:
            ref = f"{file_path}:{line_start}"
        else:
            ref = f"{file_path}:{line_start}-{line_end}"

        # Format as bold code in markdown
        return f"**`{ref}`**"

    return re.sub(pattern, replace_reference, text)


def format_answer_with_code(answer: str) -> str:
    """
    Format answer with code blocks and file references.

    Processes:
    - Code blocks in ```language ... ``` format (already markdown)
    - File references as file_path:line_number

    Args:
        answer: Raw answer text from RAG engine

    Returns:
        Formatted markdown text
    """
    # Parse and highlight file references
    formatted = parse_file_references(answer)

    # Code blocks are already in markdown format from RAG
    # Gradio will render them automatically

    return formatted


def extract_code_blocks(text: str) -> list:
    """
    Extract code blocks from markdown text.

    Args:
        text: Markdown text with code blocks

    Returns:
        List of tuples (language, code)
    """
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    code_blocks = []
    for lang, code in matches:
        code_blocks.append((lang or 'text', code.strip()))

    return code_blocks


def format_metadata(metadata: dict) -> str:
    """
    Format metadata dictionary as readable markdown.

    Args:
        metadata: Dictionary of metadata

    Returns:
        Formatted markdown string
    """
    if not metadata:
        return ""

    lines = []
    for key, value in metadata.items():
        # Format key
        formatted_key = key.replace('_', ' ').title()

        # Format value
        if isinstance(value, list):
            formatted_value = ', '.join(str(v) for v in value)
        elif isinstance(value, dict):
            formatted_value = str(value)
        else:
            formatted_value = str(value)

        lines.append(f"- **{formatted_key}:** {formatted_value}")

    return '\n'.join(lines)
