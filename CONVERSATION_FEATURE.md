# Multi-Turn Conversation Support

## Overview

The onboarding assistant now supports **multi-turn conversations**, allowing users to ask follow-up questions that reference previous parts of the conversation. This creates a more natural and intuitive Q&A experience.

## How It Works

### Architecture

1. **Conversation History Storage**
   - Each conversation is tracked as a list of `(question, answer)` tuples
   - History is maintained in the Gradio UI's state management
   - Managed by `ConversationManager` in `app/ui/utils/state_manager.py`

2. **Context Integration in RAG**
   - The RAG engine's `answer_question` method now accepts a `chat_history` parameter
   - History is used in **two stages**:
     - **Stage 1 (Search Query Generation)**: Last 3 turns used to generate better search queries
     - **Stage 3 (Final Answer)**: Last 5 turns used to provide context for answer generation

3. **Smart Context Windowing**
   - Recent history is truncated to avoid token limits
   - Answers are shortened to ~200-300 characters when passed as context
   - Only the most recent turns are used (3 for search, 5 for final answer)

### Changes Made

#### 1. RAG Engine (`app/services/rag_engine.py`)

**New Parameter:**
```python
def answer_question(
    self,
    collection_name: str,
    question: str,
    project_overview: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[tuple[str, str]]] = None  # NEW!
) -> str:
```

**Stage 1 Enhancement (Search Query Generation):**
```python
# Build chat history context if available
chat_context = ""
if chat_history and len(chat_history) > 0:
    recent_history = chat_history[-3:]  # Last 3 turns
    history_parts = []
    for q, a in recent_history:
        history_parts.append(f"Q: {q}\nA: {a[:200]}...")
    chat_context = f"\n\nRECENT CONVERSATION:\n" + "\n".join(history_parts)
```

**Stage 3 Enhancement (Final Answer Generation):**
```python
# Build conversation history for final prompt
conversation_context = ""
if chat_history and len(chat_history) > 0:
    recent_history = chat_history[-5:]  # Last 5 turns
    history_parts = []
    for i, (q, a) in enumerate(recent_history, 1):
        history_parts.append(f"Turn {i}:\nUser: {q}\nAssistant: {a[:300]}...")
    conversation_context = f"\n\nCONVERSATION HISTORY:\n" + "\n\n".join(history_parts)
```

#### 2. Gradio UI (`app/ui/gradio_app.py`)

**Updated Q&A Handler:**
```python
# OLD - Enhanced question with text context
enhanced_question = f"""Предыдущий контекст диалога:
{context}

Текущий вопрос: {question}"""

# NEW - Pass history directly to RAG engine
answer = rag_engine.answer_question(
    collection_name=job.collection_name,
    question=question,
    project_overview=job.project_overview,
    chat_history=history  # Direct history passing
)
```

## Usage Examples

### Example 1: Follow-up Questions

```python
# Turn 1
User: "What does this project do?"
Assistant: "This is a FastAPI-based web service for..."

# Turn 2 - Uses context from Turn 1
User: "What frameworks does it use?"
Assistant: "The project uses FastAPI for the web framework, as mentioned earlier..."

# Turn 3 - References previous answers
User: "Show me the main API endpoints"
Assistant: "Based on our previous discussion about the FastAPI framework..."
```

### Example 2: Pronoun Resolution

```python
# Turn 1
User: "Find the authentication module"
Assistant: "The authentication is implemented in app/auth/handlers.py..."

# Turn 2 - "it" refers to authentication module
User: "How does it work?"
Assistant: "The authentication module in app/auth/handlers.py works by..."
```

### Example 3: Building Understanding

```python
# Turn 1
User: "What is the project structure?"
Assistant: "The project follows a clean architecture with app/, tests/, config/..."

# Turn 2 - Leverages understanding from Turn 1
User: "Where are the database models?"
Assistant: "Given the structure we discussed, the models are in app/models/..."
```

## Benefits

1. **Natural Conversation Flow**
   - Ask follow-up questions without repeating context
   - Use pronouns like "it", "that", "the previous one"
   - Build on previous answers progressively

2. **Better Search Queries**
   - Context-aware search query generation
   - Understands implicit references
   - Finds more relevant code sections

3. **Contextual Answers**
   - LLM understands the conversation flow
   - Avoids repeating information
   - Provides more targeted responses

4. **Improved User Experience**
   - More intuitive Q&A interface
   - Faster to explore codebase
   - Less repetition needed

## Implementation Details

### Conversation State Management

The conversation state is managed using:
- **Gradio State**: `gr.State([])` stores the history
- **ConversationManager**: Utility class for history operations
  - `add_turn()`: Add new Q&A pair
  - `get_context()`: Get formatted context string
  - `clear_history()`: Reset conversation

### Token Management

To prevent context overflow:
1. **Limited History Window**: Only last 3-5 turns used
2. **Answer Truncation**: Long answers truncated to 200-300 chars
3. **Smart Formatting**: Minimal formatting overhead

### Clearing Conversation

The conversation is automatically cleared when:
- User clicks "🗑️ Очистить историю" button
- User switches to a different repository

## Testing

Run the test script to verify multi-turn conversation:

```bash
python test_conversation.py
```

This will:
1. Find a completed repository
2. Run a series of follow-up questions
3. Display how history is passed between turns
4. Verify the feature works correctly

## API Reference

### RAGEngine.answer_question()

```python
def answer_question(
    self,
    collection_name: str,
    question: str,
    project_overview: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[tuple[str, str]]] = None
) -> str:
    """
    Answer a question using RAG with conversation context.

    Args:
        collection_name: Collection to search
        question: User's question
        project_overview: Project overview for context
        top_k: Number of context chunks to retrieve
        chat_history: List of (question, answer) tuples

    Returns:
        Generated answer string
    """
```

### ConversationManager

```python
class ConversationManager:
    @staticmethod
    def add_turn(history, question, answer) -> List[Tuple[str, str]]:
        """Add Q&A turn to history."""

    @staticmethod
    def get_context(history, max_turns=5) -> str:
        """Get formatted context from history."""

    @staticmethod
    def clear_history() -> List[Tuple[str, str]]:
        """Clear conversation history."""
```

## Future Enhancements

Potential improvements for the conversation feature:

1. **Conversation Summarization**
   - Summarize old turns to save tokens
   - Keep longer history with compressed context

2. **Conversation Persistence**
   - Save conversations to database
   - Resume previous conversations

3. **Conversation Branching**
   - Allow users to fork conversations
   - Explore different question paths

4. **Semantic History Retrieval**
   - Retrieve most relevant turns (not just recent)
   - Better handling of long conversations

5. **Conversation Analytics**
   - Track common question patterns
   - Suggest follow-up questions

## Troubleshooting

### Issue: Answers don't reference previous conversation

**Cause**: Chat history not being passed correctly

**Solution**: Check that `chat_history=history` is passed in `gradio_app.py:545`

### Issue: Token limit errors

**Cause**: Too much conversation history

**Solution**: Reduce the history window size in `rag_engine.py`:
- Change `chat_history[-3:]` to `chat_history[-2:]` for search
- Change `chat_history[-5:]` to `chat_history[-3:]` for final answer

### Issue: History not clearing

**Cause**: State management issue

**Solution**: Verify the clear button handler in `gradio_app.py:612-615`

## Conclusion

Multi-turn conversation support makes the onboarding assistant more powerful and natural to use. Users can now explore codebases through iterative conversations, building understanding progressively without repeating context.
