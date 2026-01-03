"""
Main Gradio application for Onboarding Assistant UI.
"""

import gradio as gr
from typing import List, Tuple, Optional
import pandas as pd

from app.services.onboarding_service import RepositoryOnboardingService
from app.core.models import OnboardingJob, OnboardingState


def create_gradio_interface() -> gr.Blocks:
    """
    Create main Gradio interface with tabs.

    Returns:
        Gradio Blocks application
    """
    # Initialize service
    onboarding_service = RepositoryOnboardingService()

    # Create the main interface
    with gr.Blocks(
        title="Onboarding Assistant",
        theme=gr.themes.Soft(),
        css="""
        .code-block {
            background-color: #282c34;
            border-radius: 5px;
            padding: 10px;
        }
        """
    ) as app:
        # Header
        gr.Markdown("# 🚀 Onboarding Assistant")
        gr.Markdown("AI-powered repository onboarding and code understanding system")
        gr.Markdown("---")

        # Main tabs
        with gr.Tabs() as tabs:
            # Tab 1: Add Repository
            with gr.Tab("➕ Добавить репозиторий"):
                _create_add_repository_tab(onboarding_service)

            # Tab 2: Repository List
            with gr.Tab("📋 Список репозиториев"):
                _create_repository_list_tab(onboarding_service)

            # Tab 3: Q&A Interface
            with gr.Tab("💬 Вопросы и ответы"):
                _create_qa_tab(onboarding_service)

        # Footer
        gr.Markdown("---")
        gr.Markdown("*Powered by LlamaIndex, ChromaDB, and Google Gemini*")

    return app


def _create_add_repository_tab(service: RepositoryOnboardingService) -> None:
    """Create the repository addition tab."""
    gr.Markdown("## Добавить новый репозиторий")
    gr.Markdown("Введите URL GitHub репозитория для индексации")

    # Input section
    repo_url_input = gr.Textbox(
        label="URL репозитория",
        placeholder="https://github.com/owner/repo",
        info="Пример: https://github.com/anthropics/anthropic-sdk-python"
    )

    # Validation feedback
    url_validation_msg = gr.Markdown("")

    # Start button
    start_btn = gr.Button("🚀 Начать индексацию", variant="primary", size="lg")

    # Status and progress section
    with gr.Group(visible=False) as progress_group:
        gr.Markdown("### 🔄 Статус индексации")

        # Current state display
        current_state = gr.Markdown("")

        # Detailed progress information
        with gr.Accordion("📊 Детальный прогресс", open=True):
            progress_details = gr.Markdown("")

    # Overview section (shown after completion)
    with gr.Group(visible=False) as overview_group:
        gr.Markdown("### ✅ Индексация завершена!")
        overview_display = gr.Markdown("")
        view_repo_btn = gr.Button("📋 Перейти к списку репозиториев", variant="secondary")

    # Hidden state to track current job ID
    current_job_id = gr.State(None)

    # Help section
    with gr.Accordion("💡 Информация", open=False):
        gr.Markdown("""
        ### Процесс индексации включает:
        1. **Клонирование** - Загрузка репозитория с GitHub
        2. **Парсинг** - Анализ кода и создание индекса
        3. **Генерация обзора** - Создание AI-описания проекта
        4. **Завершение** - Репозиторий готов для вопросов

        ### Поддерживаемые языки:
        Python, JavaScript, TypeScript, Java, Go, Rust, C++, C, C#, Ruby, PHP и другие

        ### Примечание:
        Процесс индексации может занять от нескольких минут до десятков минут
        в зависимости от размера репозитория.
        """)

    # Event handlers
    def validate_url(url: str) -> str:
        """Validate repository URL."""
        if not url:
            return "⚠️ Введите URL"

        if not url.startswith("https://github.com/"):
            return "⚠️ Поддерживаются только GitHub репозитории (начинающиеся с https://github.com/)"

        parts = url.rstrip('/').split('/')
        if len(parts) < 5:
            return "⚠️ Неверный формат URL. Ожидается: https://github.com/owner/repo"

        return "✅ URL корректен"

    def start_onboarding(repo_url: str):
        """Start repository onboarding process."""
        if not repo_url:
            return {
                progress_group: gr.update(visible=False),
                overview_group: gr.update(visible=False),
                current_job_id: None
            }

        validation = validate_url(repo_url)
        if not validation.startswith("✅"):
            return {
                progress_group: gr.update(visible=False),
                overview_group: gr.update(visible=False),
                current_job_id: None
            }

        try:
            # Create job
            job = service.create_job(repo_url)

            # Start job in background thread to avoid blocking
            import threading
            import asyncio

            def run_job_async():
                """Run async job in new event loop in background thread."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(service._process_job_async(job))
                finally:
                    loop.close()

            thread = threading.Thread(target=run_job_async, daemon=True)
            thread.start()

            # Show progress group
            return {
                progress_group: gr.update(visible=True),
                overview_group: gr.update(visible=False),
                current_job_id: job.job_id,
                current_state: f"🔵 **{job.current_state}**",
                progress_details: f"""
**ID задачи:** `{job.job_id}`
**Репозиторий:** {job.repo_url}
**Создано:** {job.created_at.strftime("%Y-%m-%d %H:%M:%S")}
"""
            }

        except Exception:
            import traceback
            print(f"Error starting onboarding: {traceback.format_exc()}")
            return {
                progress_group: gr.update(visible=False),
                overview_group: gr.update(visible=False),
                current_job_id: None
            }

    def poll_job_status(job_id: str):
        """Poll job status and update UI."""
        import time

        if not job_id:
            return {
                progress_group: gr.update(visible=False),
                overview_group: gr.update(visible=False)
            }

        # Poll for updates
        max_polls = 600  # 10 minutes (600 * 1 second)
        for _ in range(max_polls):
            job = service.get_job_status(job_id)
            if not job:
                break

            # Calculate progress
            progress_percent = job.calculate_progress_percent()

            # State-specific icons and messages
            state_str = job.current_state if isinstance(job.current_state, str) else job.current_state.value
            state_icons = {
                "created": "⚪",
                "cloning": "🔵",
                "parsing": "🟡",
                "generating_overview": "🟠",
                "completed": "✅",
                "failed": "❌"
            }
            icon = state_icons.get(state_str, "⚪")

            # Build progress details
            details = f"""
**ID задачи:** `{job.job_id}`
**Репозиторий:** {job.repo_url}
**Прогресс:** {progress_percent}%
**Текущий статус:** {icon} {state_str}

---

**📊 Статистика:**
- Файлов обработано: {job.total_files}
- Чанков создано: {job.total_chunks}
- Языки: {', '.join(job.languages_detected[:5]) if job.languages_detected else 'определяются...'}
"""

            if job.error:
                details += f"\n\n**❌ Ошибка:**\n```\n{job.error}\n```"

            # Yield progress update
            yield {
                current_state: f"{icon} **{state_str}** ({progress_percent}%)",
                progress_details: details,
                progress_group: gr.update(visible=True),
                overview_group: gr.update(visible=False)
            }

            # Check if completed or failed
            if state_str in ["completed", "failed"]:
                # Show overview if completed
                if state_str == "completed":
                    overview_text = ""
                    if job.project_overview:
                        overview_text = job.project_overview
                    else:
                        overview_text = "*Обзор не был сгенерирован*"

                    yield {
                        current_state: f"✅ **Завершено**",
                        progress_details: details,
                        progress_group: gr.update(visible=False),
                        overview_group: gr.update(visible=True),
                        overview_display: overview_text
                    }
                break

            time.sleep(1)  # Poll every second

    # Wire up events
    repo_url_input.change(
        fn=validate_url,
        inputs=[repo_url_input],
        outputs=[url_validation_msg]
    )

    start_btn.click(
        fn=start_onboarding,
        inputs=[repo_url_input],
        outputs=[progress_group, overview_group, current_job_id, current_state, progress_details]
    ).then(
        fn=poll_job_status,
        inputs=[current_job_id],
        outputs=[current_state, progress_details, progress_group, overview_group, overview_display]
    )


def _create_repository_list_tab(service: RepositoryOnboardingService) -> None:
    """Create the repository list tab."""
    gr.Markdown("## Индексированные репозитории")

    # Refresh button
    refresh_btn = gr.Button("🔄 Обновить список", size="sm")

    # DataFrame display
    repos_table = gr.DataFrame(
        label="Репозитории",
        wrap=True,
        interactive=False
    )

    # Selected repository details
    with gr.Accordion("📄 Детали репозитория", open=False) as details_accordion:
        repo_details = gr.Markdown("*Выберите репозиторий из таблицы для просмотра деталей*")

    def format_jobs_as_dataframe(jobs: List[OnboardingJob]) -> pd.DataFrame:
        """Convert list of jobs to DataFrame for display."""
        if not jobs:
            return pd.DataFrame({
                "Сообщение": ["Нет индексированных репозиториев. Добавьте репозиторий во вкладке '➕ Добавить репозиторий'"]
            })

        data = []
        for job in jobs:
            # Shorten ID for display
            short_id = job.job_id[:8]

            # Extract repo name from URL
            repo_name = job.repo_url.split('/')[-1] if '/' in job.repo_url else job.repo_url

            # Format dates
            created = job.created_at.strftime("%Y-%m-%d %H:%M")
            updated = job.updated_at.strftime("%Y-%m-%d %H:%M")

            # Format languages (limit to 3)
            languages = ", ".join(job.languages_detected[:3]) if job.languages_detected else "-"
            if len(job.languages_detected) > 3:
                languages += f" (+{len(job.languages_detected) - 3})"

            # Status emoji
            status_emoji_map = {
                "created": "⚪",
                "cloning": "🔵",
                "parsing": "🟡",
                "generating_overview": "🟠",
                "completed": "✅",
                "failed": "❌"
            }
            # Handle both string and enum
            state_str = job.current_state if isinstance(job.current_state, str) else job.current_state.value
            status_emoji = status_emoji_map.get(state_str, "⚪")

            data.append({
                "ID": short_id,
                "Репозиторий": repo_name,
                "Статус": f"{status_emoji} {state_str}",
                "Файлов": job.total_files,
                "Чанков": job.total_chunks,
                "Языки": languages,
                "Создано": created,
                "Обновлено": updated,
            })

        return pd.DataFrame(data)

    def load_repositories() -> pd.DataFrame:
        """Load and display all repositories."""
        jobs = service.list_jobs()
        # Sort by updated_at descending
        jobs.sort(key=lambda j: j.updated_at, reverse=True)
        return format_jobs_as_dataframe(jobs)

    def show_repo_details(evt: gr.SelectData) -> str:
        """Show details when user clicks on a row."""
        if evt is None:
            return "*Выберите репозиторий из таблицы*"

        row_index = evt.index[0]
        jobs = service.list_jobs()
        jobs.sort(key=lambda j: j.updated_at, reverse=True)

        if row_index >= len(jobs):
            return "❌ Не удалось загрузить детали"

        job = jobs[row_index]

        # Build detailed view
        details = f"""
### {job.repo_url}

**ID задачи:** `{job.job_id}`
**Статус:** {job.current_state}
**Коллекция:** `{job.collection_name}`

---

#### 📊 Статистика
- **Файлов обработано:** {job.total_files}
- **Чанков создано:** {job.total_chunks}
- **Файлов с ошибками:** {len(job.failed_files)}
- **Языки программирования:** {", ".join(job.languages_detected) if job.languages_detected else "Не определено"}

---

#### 📅 Временные метки
- **Создано:** {job.created_at.strftime("%Y-%m-%d %H:%M:%S")}
- **Обновлено:** {job.updated_at.strftime("%Y-%m-%d %H:%M:%S")}

---

#### 📝 Обзор проекта
"""

        if job.project_overview:
            details += f"\n{job.project_overview}\n"
        else:
            # Check state (handle both string and enum)
            state_str = job.current_state if isinstance(job.current_state, str) else job.current_state.value
            if state_str == "completed":
                details += "\n*Обзор не был сгенерирован*\n"
            else:
                details += "\n*Обзор будет сгенерирован после завершения индексации*\n"

        if job.error:
            details += f"\n---\n\n#### ❌ Ошибка\n```\n{job.error}\n```\n"

        return details

    # Events
    refresh_btn.click(
        fn=load_repositories,
        outputs=[repos_table]
    )

    repos_table.select(
        fn=show_repo_details,
        outputs=[repo_details]
    )

    # Set initial value (will load on first render)
    repos_table.value = load_repositories()


def _create_qa_tab(service: RepositoryOnboardingService) -> None:
    """Create the Q&A tab."""
    gr.Markdown("## Задайте вопрос о кодовой базе")

    # Repository selector
    def get_completed_repos() -> List[Tuple[str, str]]:
        """Get list of completed repositories for dropdown."""
        jobs = service.list_jobs()
        completed = [
            (f"{job.repo_url.split('/')[-1]} ({job.job_id[:8]})", job.job_id)
            for job in jobs
            if (job.current_state == OnboardingState.COMPLETED or
                job.current_state == "completed")
        ]

        if not completed:
            return [("Нет завершенных репозиториев", "")]

        return completed

    with gr.Row():
        repo_selector = gr.Dropdown(
            label="Выберите репозиторий",
            choices=get_completed_repos(),
            interactive=True,
            scale=3
        )
        refresh_repos_btn = gr.Button("🔄 Обновить", size="sm", scale=1)

    # Conversation state
    conversation_state = gr.State([])  # List of (question, answer) tuples

    # Chat display
    chatbot = gr.Chatbot(
        label="Диалог",
        height=500,
        show_label=True,
        bubble_full_width=False
    )

    # Input area
    with gr.Row():
        question_input = gr.Textbox(
            label="Ваш вопрос",
            placeholder="Например: Как работает аутентификация в этом проекте?",
            scale=4,
            lines=1
        )
        submit_btn = gr.Button("📤 Отправить", scale=1, variant="primary")

    clear_btn = gr.Button("🗑️ Очистить историю", size="sm")

    # Help section
    with gr.Accordion("💡 Примеры вопросов", open=False):
        gr.Markdown("""
        ### Общие вопросы:
        - Что делает этот проект?
        - Какая архитектура используется?
        - Какие основные зависимости?

        ### Технические вопросы:
        - Как работает аутентификация?
        - Где определены API endpoints?
        - Покажи главный файл приложения
        - Как устроена база данных?
        - Какие паттерны проектирования используются?

        ### Поиск кода:
        - Найди функцию для обработки платежей
        - Покажи класс User
        - Где происходит валидация данных?
        """)

    # Event handlers
    def ask_question(question: str, repo_id: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """
        Handle question and return updated conversation.

        Args:
            question: User's question
            repo_id: Selected repository ID
            history: Conversation history

        Returns:
            Updated history and empty string (to clear input)
        """
        if not repo_id or repo_id == "":
            error_msg = "❌ Пожалуйста, выберите репозиторий"
            return history + [(question, error_msg)], ""

        if not question.strip():
            return history, ""

        try:
            # Get job and collection name
            job = service.get_job_status(repo_id)
            if not job:
                return history + [(question, "❌ Репозиторий не найден")], ""

            # Check if completed (handle both string and enum)
            state_str = job.current_state if isinstance(job.current_state, str) else job.current_state.value
            if state_str != "completed":
                return history + [(question, f"❌ Репозиторий еще не готов. Статус: {state_str}")], ""

            # Build context from conversation history
            from app.ui.utils.state_manager import ConversationManager
            context = ConversationManager.get_context(history, max_turns=5)

            # Enhance question with context if needed
            enhanced_question = question
            if context and len(history) > 0:
                enhanced_question = f"""Предыдущий контекст диалога:
{context}

Текущий вопрос: {question}

Пожалуйста, учитывай предыдущий контекст при ответе."""

            # Query RAG engine
            from app.services.rag_engine import RAGEngine
            rag_engine = RAGEngine()

            answer = rag_engine.answer_question(
                collection_name=job.collection_name,
                question=question, # TOOD: we should pass the history and the question to that function 
            )

            # Format answer with syntax highlighting and references
            from app.ui.utils.formatters import format_answer_with_code
            formatted_answer = format_answer_with_code(answer)

            # Add to history
            new_history = ConversationManager.add_turn(history, question, formatted_answer)

            return new_history, ""  # Return history and clear input

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in ask_question: {error_details}")
            error_answer = f"❌ Ошибка при обработке вопроса:\n```\n{str(e)}\n```"
            return history + [(question, error_answer)], ""

    def clear_conversation() -> Tuple[List, List]:
        """Clear conversation history."""
        return [], []

    def refresh_repo_list():
        """Refresh the repository dropdown list."""
        new_choices = get_completed_repos()
        return gr.Dropdown(choices=new_choices)

    # Wire up events
    refresh_repos_btn.click(
        fn=refresh_repo_list,
        outputs=[repo_selector]
    )

    submit_btn.click(
        fn=ask_question,
        inputs=[question_input, repo_selector, conversation_state],
        outputs=[chatbot, question_input]
    ).then(
        fn=lambda h: h,
        inputs=[chatbot],
        outputs=[conversation_state]
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input, repo_selector, conversation_state],
        outputs=[chatbot, question_input]
    ).then(
        fn=lambda h: h,
        inputs=[chatbot],
        outputs=[conversation_state]
    )

    clear_btn.click(
        fn=clear_conversation,
        outputs=[chatbot, conversation_state]
    )

    repo_selector.change(
        fn=clear_conversation,
        outputs=[chatbot, conversation_state]
    )


# For standalone testing
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)
