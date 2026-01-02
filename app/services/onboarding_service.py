"""
Repository onboarding service - orchestrates the full onboarding workflow.
"""

import uuid
import json
import asyncio
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from app.core.models import OnboardingJob, OnboardingState
from app.core.exceptions import OnboardingError, StateTransitionError
from app.services.state_machine import OnboardingStateMachine
from app.services.rag_engine import RAGEngine, create_collection_name
from app.utils.file_filter import filter_repository_files, count_files_by_language
from app.config import settings


# Special files to collect for overview generation
OVERVIEW_FILES = {
    # README files
    'readme.md', 'readme.rst', 'readme.txt', 'readme',
    # Package/dependency files
    'requirements.txt', 'pyproject.toml', 'setup.py', 'pipfile',
    'package.json', 'package-lock.json', 'yarn.lock',
    'cargo.toml', 'go.mod', 'go.sum',
    'pom.xml', 'build.gradle', 'build.gradle.kts',
    'gemfile', 'gemfile.lock',
    # Configuration files
    'dockerfile', '.dockerignore',
    'makefile', 'cmakelists.txt',
    '.env.example', 'config.yaml', 'config.yml', 'config.json',
}


class RepositoryOnboardingService:
    """Service for managing repository onboarding jobs."""

    def __init__(self):
        """Initialize the onboarding service."""
        self.jobs: Dict[str, OnboardingJob] = {}
        self.jobs_file = Path(settings.cache_dir) / "onboarding_jobs.json"
        self.rag_engine = RAGEngine()  # Now handles everything: repo loading, indexing, LLM

        self._ensure_cache_dir()
        self._load_jobs()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)

    def _load_jobs(self) -> None:
        """Load jobs from JSON file."""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.jobs = {
                        job_id: OnboardingJob.from_dict(job_data)
                        for job_id, job_data in data.items()
                    }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Could not load onboarding jobs file: {e}")
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
            print(f"Error saving onboarding jobs to file: {e}")

    def _update_job(self, job: OnboardingJob) -> None:
        """Update job in memory and persist to disk."""
        job.updated_at = datetime.utcnow()
        self.jobs[job.job_id] = job
        self._save_jobs()

    def create_job(self, repo_url: str, force_reclone: bool = False) -> OnboardingJob:
        """
        Create a new onboarding job.

        Args:
            repo_url: Repository URL
            force_reclone: If True, remove existing clone and re-clone

        Returns:
            Created OnboardingJob
        """
        job_id = str(uuid.uuid4())
        collection_name = create_collection_name(repo_url)

        job = OnboardingJob(
            job_id=job_id,
            repo_url=repo_url,
            collection_name=collection_name
        )

        self._update_job(job)
        return job

    def start_job(self, job_id: str) -> OnboardingJob:
        """
        Start processing an onboarding job asynchronously.

        Args:
            job_id: Job identifier

        Returns:
            Updated job

        Raises:
            OnboardingError: If job not found or cannot be started
        """
        job = self.get_job_status(job_id)
        if not job:
            raise OnboardingError(f"Job {job_id} not found")

        if job.current_state != OnboardingState.CREATED:
            raise OnboardingError(f"Job {job_id} already started (state: {job.current_state})")

        # Start the workflow in background
        asyncio.create_task(self._process_job_async(job))

        return job

    async def _process_job_async(self, job: OnboardingJob) -> None:
        """
        Process a job through all states asynchronously.

        Args:
            job: The job to process
        """
        state_machine = OnboardingStateMachine(job)

        try:
            # State: CLONING
            if job.current_state == OnboardingState.CREATED:
                state_machine.transition_to(OnboardingState.CLONING)
                self._update_job(job)

            if job.current_state == OnboardingState.CLONING:
                await self._handle_cloning(job, state_machine)

            # State: PARSING (includes filtering, parsing, and indexing to vector store)
            if job.current_state == OnboardingState.PARSING:
                await self._handle_parsing(job, state_machine)

            # State: GENERATING_OVERVIEW
            if job.current_state == OnboardingState.GENERATING_OVERVIEW:
                await self._handle_generating_overview(job, state_machine)

        except Exception as e:
            self._handle_error(job, state_machine, e)

    async def _handle_cloning(self, job: OnboardingJob, sm: OnboardingStateMachine) -> None:
        """Handle the cloning/loading state using RAG engine."""
        sm.start_state(OnboardingState.CLONING)
        self._update_job(job)

        try:
            # Load repository using RAG engine (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            clone_path, documents = await loop.run_in_executor(
                None,
                self.rag_engine.load_repository,
                job.repo_url,
                False,  # force parameter
                settings.github_token  # GitHub token from config
            )
            job.clone_path = clone_path

            # Store documents for later use in parsing
            job.llama_documents = documents

            sm.complete_state(OnboardingState.CLONING, {
                "clone_path": clone_path,
                "documents_loaded": len(documents)
            })

            # Transition to parsing
            sm.transition_to(OnboardingState.PARSING)
            self._update_job(job)

        except Exception as e:
            raise OnboardingError(f"Cloning error: {str(e)}")

    async def _handle_parsing(self, job: OnboardingJob, sm: OnboardingStateMachine) -> None:
        """
        Handle the parsing state - process LlamaIndex documents and stream chunks to vector store.
        This state now includes all indexing operations.
        """
        sm.start_state(OnboardingState.PARSING)
        self._update_job(job)

        try:
            # Use LlamaIndex documents if available
            if job.llama_documents:
                await self._handle_parsing_from_llama_documents(job, sm)
            else:
                # Fallback to file-based parsing for backward compatibility
                await self._handle_parsing_from_files(job, sm)

        except Exception as e:
            raise OnboardingError(f"Parsing/Indexing error: {str(e)}")

    async def _handle_parsing_from_llama_documents(self, job: OnboardingJob, sm: OnboardingStateMachine) -> None:
        """
        Parse using LlamaIndex documents.
        Add documents directly to the vector store without conversion.
        """
        from pathlib import Path

        loop = asyncio.get_event_loop()

        # Get document count and languages
        job.total_files = len(job.llama_documents)
        language_counts = {}

        # Create index if it doesn't exist
        if not self.rag_engine.index_exists(job.collection_name):
            self.rag_engine.create_index(job.repo_url, job.collection_name)

        # Process documents and add to vector store
        processed_files = 0
        total_docs = 0
        doc_batch = []
        BATCH_SIZE = 50  # Process documents in batches

        for doc in job.llama_documents:
            try:
                file_path = doc.metadata.get('file_path', 'unknown')
                file_name = doc.metadata.get('file_name', 'unknown')

                # Detect language from file extension
                file_ext = Path(file_path).suffix.lower()
                language = self._get_language_from_extension(file_ext)
                if language:
                    language_counts[language] = language_counts.get(language, 0) + 1

                # Collect metadata for overview generation
                if self._is_overview_file_name(file_name):
                    try:
                        job.metadata_for_overview[file_path] = doc.text[:500000]  # Limit size
                    except Exception:
                        pass

                # Add document to batch
                doc_batch.append(doc)
                total_docs += 1
                processed_files += 1

                # When batch is full, add to vector store
                if len(doc_batch) >= BATCH_SIZE:
                    await loop.run_in_executor(
                        None,
                        self.rag_engine.add_documents,
                        job.collection_name,
                        doc_batch
                    )
                    doc_batch = []  # Clear batch

                # Update progress periodically
                if processed_files % 10 == 0:
                    metrics = sm.get_state_metrics(OnboardingState.PARSING)
                    metrics.data["processed_files"] = processed_files
                    metrics.data["documents_indexed"] = total_docs
                    self._update_job(job)

            except Exception as e:
                # Skip failed files, log them
                job.failed_files.append(file_path)
                print(f"Error processing {file_path}: {e}")

        # Add remaining documents in batch
        if doc_batch:
            await loop.run_in_executor(
                None,
                self.rag_engine.add_documents,
                job.collection_name,
                doc_batch
            )

        job.total_chunks = total_docs
        job.languages_detected = list(language_counts.keys())

        sm.complete_state(OnboardingState.PARSING, {
            "processed_files": processed_files,
            "documents_indexed": total_docs,
            "failed_files": len(job.failed_files),
            "languages": language_counts,
            "overview_files_collected": len(job.metadata_for_overview)
        })

        # Transition to overview generation
        sm.transition_to(OnboardingState.GENERATING_OVERVIEW)
        self._update_job(job)

    async def _handle_parsing_from_files(self, job: OnboardingJob, sm: OnboardingStateMachine) -> None:
        """
        Parse using file system (fallback/backward compatibility).
        Note: This path is rarely used since LlamaIndex documents are preferred.
        """
        from llama_index.core import Document

        repo_path = Path(job.clone_path)

        # Filter files (only done once!)
        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(
            None,
            filter_repository_files,
            repo_path
        )

        job.total_files = len(files)
        language_counts = count_files_by_language(files)
        job.languages_detected = list(language_counts.keys())
        self._update_job(job)

        # Create index if it doesn't exist
        if not self.rag_engine.index_exists(job.collection_name):
            self.rag_engine.create_index(job.repo_url, job.collection_name)

        # Convert files to documents and add to vector store
        processed_files = 0
        total_docs = 0
        doc_batch = []
        BATCH_SIZE = 50  # Process documents in batches

        for file_path in files:
            try:
                # Collect metadata for overview generation
                if self._is_overview_file(file_path, repo_path):
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        relative_path = str(file_path.relative_to(repo_path))
                        job.metadata_for_overview[relative_path] = content
                    except Exception:
                        pass

                # Read file and create Document
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    relative_path = str(file_path.relative_to(repo_path))

                    # Create LlamaIndex Document
                    doc = Document(
                        text=content,
                        metadata={
                            "file_path": relative_path,
                            "file_name": file_path.name,
                            "language": self._get_language_from_extension(file_path.suffix.lower())
                        }
                    )

                    doc_batch.append(doc)
                    total_docs += 1
                    processed_files += 1

                    # When batch is full, add to vector store
                    if len(doc_batch) >= BATCH_SIZE:
                        await loop.run_in_executor(
                            None,
                            self.rag_engine.add_documents,
                            job.collection_name,
                            doc_batch
                        )
                        doc_batch = []  # Clear batch

                    # Update progress periodically
                    if processed_files % 10 == 0:
                        metrics = sm.get_state_metrics(OnboardingState.PARSING)
                        metrics.data["processed_files"] = processed_files
                        metrics.data["documents_indexed"] = total_docs
                        self._update_job(job)

                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    job.failed_files.append(str(file_path))

            except Exception as e:
                # Skip failed files, log them
                job.failed_files.append(str(file_path))

        # Add remaining documents in batch
        if doc_batch:
            await loop.run_in_executor(
                None,
                self.rag_engine.add_documents,
                job.collection_name,
                doc_batch
            )

        job.total_chunks = total_docs

        sm.complete_state(OnboardingState.PARSING, {
            "processed_files": processed_files,
            "documents_indexed": total_docs,
            "failed_files": len(job.failed_files),
            "languages": language_counts,
            "overview_files_collected": len(job.metadata_for_overview)
        })

        # Transition to overview generation
        sm.transition_to(OnboardingState.GENERATING_OVERVIEW)
        self._update_job(job)

    async def _handle_generating_overview(self, job: OnboardingJob, sm: OnboardingStateMachine) -> None:
        """
        Handle the overview generation state.
        Uses hybrid approach: file analysis + RAG queries.
        """
        sm.start_state(OnboardingState.GENERATING_OVERVIEW)
        self._update_job(job)

        try:
            loop = asyncio.get_event_loop()

            # Phase 1: File Analysis (already collected during parsing)
            file_data = await loop.run_in_executor(
                None,
                self._analyze_collected_files,
                job
            )

            # Phase 2: RAG Queries
            rag_data = await loop.run_in_executor(
                None,
                self._query_rag_for_insights,
                job.collection_name,
                file_data
            )

            # Phase 3: LLM Synthesis
            overview = await loop.run_in_executor(
                None,
                self._synthesize_overview,
                file_data,
                rag_data,
                job
            )

            job.project_overview = overview

            sm.complete_state(OnboardingState.GENERATING_OVERVIEW, {
                "overview_length": len(overview),
                "file_data_keys": list(file_data.keys()),
                "rag_queries_made": len(rag_data)
            })

            # Transition to completed
            sm.transition_to(OnboardingState.COMPLETED)
            self._update_job(job)

        except Exception as e:
            raise OnboardingError(f"Overview generation error: {str(e)}")

    def _analyze_collected_files(self, job: OnboardingJob) -> Dict[str, any]:
        """
        Analyze files collected during parsing.

        Returns:
            Dictionary with extracted information
        """
        file_data = {
            "readme_content": None,
            "dependencies": [],
            "tech_stack": set(job.languages_detected),
            "config_files": [],
            "has_docker": False,
            "package_info": {}
        }

        for file_path, content in job.metadata_for_overview.items():
            filename_lower = Path(file_path).name.lower()

            # Extract README
            if filename_lower.startswith('readme'):
                file_data["readme_content"] = content[:5000]  # Limit to first 5000 chars

            # Extract dependencies from requirements.txt
            elif filename_lower == 'requirements.txt':
                deps = [line.strip().split('==')[0].split('>=')[0].split('<=')[0]
                       for line in content.split('\n')
                       if line.strip() and not line.strip().startswith('#')]
                file_data["dependencies"].extend(deps[:20])  # Limit to 20

            # Extract from package.json
            elif filename_lower == 'package.json':
                try:
                    import json
                    pkg = json.loads(content)
                    file_data["package_info"] = {
                        "name": pkg.get("name"),
                        "description": pkg.get("description"),
                        "scripts": list(pkg.get("scripts", {}).keys())
                    }
                    if "dependencies" in pkg:
                        file_data["dependencies"].extend(list(pkg["dependencies"].keys())[:20])
                except:
                    pass

            # Detect frameworks from package files
            elif filename_lower in ['cargo.toml', 'go.mod', 'pom.xml']:
                file_data["config_files"].append(file_path)

            # Docker detection
            elif filename_lower == 'dockerfile':
                file_data["has_docker"] = True

        file_data["tech_stack"] = list(file_data["tech_stack"])
        return file_data

    def _query_rag_for_insights(self, collection_name: str, file_data: Optional[Dict] = None) -> Dict[str, str]:
        """
        Query RAG engine for architectural insights.

        Args:
            collection_name: Collection to query
            file_data: Data from file analysis

        Returns:
            Dictionary with RAG query results
        """
        rag_data = {}

        queries = [
            ("entry_points", "What are the main entry points of this application? List main files, API endpoints, or CLI commands."),
            ("key_modules", "What are the key modules and components in this codebase? Describe their purpose."),
            ("architecture", "Describe the overall architecture of this project. What patterns are used?")
        ]

        for key, query in queries:
            try:
                results = self.rag_engine.search(
                    collection_name=collection_name,
                    query=query,
                    top_k=5
                )

                # Format results from LlamaIndex NodeWithScore objects
                if results:
                    formatted = "\n".join([
                        f"- {node.metadata.get('file_path', 'unknown')} ({node.metadata.get('language', 'unknown')})"
                        for node in results[:3]
                    ])
                    rag_data[key] = formatted
                else:
                    rag_data[key] = "Нет данных"
            except Exception as e:
                rag_data[key] = f"Ошибка: {str(e)}"

        return rag_data

    def _synthesize_overview(self, file_data: Dict, rag_data: Dict, job: OnboardingJob) -> str:
        """
        Synthesize final overview using LLM.

        Args:
            file_data: Extracted file data
            rag_data: RAG query results
            job: Current job

        Returns:
            Generated overview in Russian
        """
        # Build context for LLM
        context = f"""
Проанализируй следующую информацию о репозитории и создай структурированный обзор.

ИНФОРМАЦИЯ ИЗ ФАЙЛОВ:
README: {file_data.get('readme_content', 'Не найден')[:1000]}
Зависимости: {', '.join(file_data.get('dependencies', [])[:15])}
Языки программирования: {', '.join(file_data.get('tech_stack', []))}
Имя проекта: {file_data.get('package_info', {}).get('name', 'Неизвестно')}
Описание из package.json: {file_data.get('package_info', {}).get('description', 'Нет')}
Docker: {'Да' if file_data.get('has_docker') else 'Нет'}
Количество файлов: {job.total_files}
Количество чанков кода: {job.total_chunks}

ИНФОРМАЦИЯ ИЗ АНАЛИЗА КОДА (RAG):
Точки входа:
{rag_data.get('entry_points', 'Не найдены')}

Ключевые модули:
{rag_data.get('key_modules', 'Не найдены')}

Архитектура:
{rag_data.get('architecture', 'Не определена')}
"""

        prompt = f"""На основе предоставленной информации создай структурированный обзор проекта на русском языке.

{context}

Создай обзор в следующем формате:

## Назначение проекта
[Опиши назначение проекта на основе README и структуры]

## Основной технологический стек
[Перечисли основные технологии, фреймворки и языки]

## Ключевые модули/компоненты
[Опиши основные модули и их назначение]

## Точки входа (entry points)
[Укажи главные точки входа в приложение]

## Краткая архитектурная схема
[Текстовое описание архитектуры проекта]

Требования:
- Пиши кратко и по делу
- Используй русский язык
- Основывайся только на предоставленной информации
- Если какой-то информации нет, так и напиши"""
        print(prompt)
        try:
            # Use RAG engine's LLM (configured in Settings.llm)
            from llama_index.core import Settings
            response = Settings.llm.complete(prompt)
            return str(response)
        except Exception as e:
            return f"Ошибка генерации обзора: {str(e)}"

    def _is_overview_file(self, file_path: Path, repo_path: Path) -> bool:
        """
        Check if a file should be collected for overview generation.

        Args:
            file_path: Path to the file
            repo_path: Repository root path

        Returns:
            True if file should be collected
        """
        # Only collect files in root or first-level directories
        try:
            relative_path = file_path.relative_to(repo_path)
            depth = len(relative_path.parts)
            if depth > 2:  # Skip files too deep in the tree
                return False
        except ValueError:
            return False

        filename_lower = file_path.name.lower()

        # Check if it's a special file
        if filename_lower in OVERVIEW_FILES:
            # Skip if file is too large (> 500KB)
            try:
                if file_path.stat().st_size > 500_000:
                    return False
                return True
            except:
                return False

        return False

    def _is_overview_file_name(self, file_name: str) -> bool:
        """
        Check if a file name indicates it should be collected for overview.

        Args:
            file_name: Name of the file

        Returns:
            True if file should be collected
        """
        filename_lower = file_name.lower()
        return filename_lower in OVERVIEW_FILES

    def _get_language_from_extension(self, extension: str) -> Optional[str]:
        """
        Get language name from file extension.

        Args:
            extension: File extension (e.g., '.py', '.js')

        Returns:
            Language name or None
        """
        extension_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.jsx': 'JavaScript',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++',
            '.cs': 'C#',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.md': 'Markdown',
            '.txt': 'Text',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
        }
        return extension_map.get(extension.lower())

    def _handle_error(self, job: OnboardingJob, sm: OnboardingStateMachine, error: Exception) -> None:
        """
        Handle error during job processing.

        Args:
            job: The job that encountered an error
            sm: State machine
            error: The exception that occurred
        """
        job.error = str(error)
        sm.transition_to(OnboardingState.FAILED, triggered_by="error")
        self._update_job(job)

    def get_job_status(self, job_id: str) -> Optional[OnboardingJob]:
        """
        Get the status of a job.

        Args:
            job_id: Job identifier

        Returns:
            OnboardingJob or None if not found
        """
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[OnboardingJob]:
        """
        List all onboarding jobs.

        Returns:
            List of all jobs
        """
        return list(self.jobs.values())

    async def retry_job_async(self, job_id: str) -> OnboardingJob:
        """
        Retry a failed job asynchronously.

        Args:
            job_id: Job identifier

        Returns:
            Updated job

        Raises:
            OnboardingError: If job cannot be retried
        """
        job = self.get_job_status(job_id)
        if not job:
            raise OnboardingError(f"Job {job_id} not found")

        sm = OnboardingStateMachine(job)

        if not sm.is_retryable():
            raise OnboardingError(
                f"Job {job_id} cannot be retried "
                f"(state: {job.current_state}, retries: {job.retry_count}/{job.max_retries})"
            )

        # Get state to retry from
        retry_state = sm.get_retry_state()

        # Increment retry count
        job.retry_count += 1
        job.error = None

        # Transition to retry state
        sm.transition_to(retry_state, triggered_by="retry")
        self._update_job(job)

        # Resume processing
        await self._process_job_async(job)

        return job

    def retry_job(self, job_id: str) -> OnboardingJob:
        """
        Retry a failed job (synchronous wrapper).

        Args:
            job_id: Job identifier

        Returns:
            Updated job

        Raises:
            OnboardingError: If job cannot be retried
        """
        # Create task in background
        asyncio.create_task(self.retry_job_async(job_id))

        # Return current job state
        job = self.get_job_status(job_id)
        if not job:
            raise OnboardingError(f"Job {job_id} not found")
        return job