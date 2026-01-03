"""
State management for UI - conversation history and context.
"""

from typing import List, Tuple, Optional


class ConversationManager:
    """Manage conversation history and context for Q&A interface."""

    @staticmethod
    def add_turn(
        history: List[Tuple[str, str]],
        question: str,
        answer: str
    ) -> List[Tuple[str, str]]:
        """
        Add Q&A turn to conversation history.

        Args:
            history: Current conversation history
            question: User's question
            answer: Assistant's answer

        Returns:
            Updated conversation history
        """
        return history + [(question, answer)]

    @staticmethod
    def get_context(
        history: List[Tuple[str, str]],
        max_turns: int = 5
    ) -> str:
        """
        Build context string from recent conversation history.

        Args:
            history: Conversation history
            max_turns: Maximum number of recent turns to include

        Returns:
            Context string for RAG query
        """
        if not history:
            return ""

        # Get last N turns
        recent = history[-max_turns:] if len(history) > max_turns else history

        context_parts = []
        for q, a in recent:
            context_parts.append(f"Q: {q}\nA: {a}\n")

        return "\n".join(context_parts)

    @staticmethod
    def get_last_question(history: List[Tuple[str, str]]) -> Optional[str]:
        """
        Get the last question from conversation history.

        Args:
            history: Conversation history

        Returns:
            Last question or None
        """
        if not history:
            return None
        return history[-1][0]

    @staticmethod
    def get_last_answer(history: List[Tuple[str, str]]) -> Optional[str]:
        """
        Get the last answer from conversation history.

        Args:
            history: Conversation history

        Returns:
            Last answer or None
        """
        if not history:
            return None
        return history[-1][1]

    @staticmethod
    def count_turns(history: List[Tuple[str, str]]) -> int:
        """
        Count number of turns in conversation.

        Args:
            history: Conversation history

        Returns:
            Number of turns
        """
        return len(history)

    @staticmethod
    def clear_history() -> List[Tuple[str, str]]:
        """
        Clear conversation history.

        Returns:
            Empty list
        """
        return []

    @staticmethod
    def format_history_for_display(history: List[Tuple[str, str]]) -> str:
        """
        Format conversation history as markdown for display.

        Args:
            history: Conversation history

        Returns:
            Formatted markdown string
        """
        if not history:
            return "*Нет истории диалога*"

        formatted = []
        for i, (q, a) in enumerate(history, 1):
            formatted.append(f"### Вопрос {i}\n{q}\n\n### Ответ {i}\n{a}\n\n---\n")

        return "\n".join(formatted)
