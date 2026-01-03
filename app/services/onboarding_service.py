"""
Repository onboarding service - orchestrates the full onboarding workflow.
"""

import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from app.core.models import OnboardingJob, OnboardingState
from app.core.exceptions import OnboardingError, StateTransitionError
from app.services.state_machine import OnboardingStateMachine
from app.services.rag_engine import RAGEngine, create_collection_name
from app.utils.file_filter import filter_repository_files, count_files_by_language
from app.config import settings

logger = logging.getLogger(__name__)


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
        logger.info("Initializing RepositoryOnboardingService")
        self.jobs: Dict[str, OnboardingJob] = {}
        self.jobs_file = Path(settings.cache_dir) / "onboarding_jobs.json"
        self.rag_engine = RAGEngine()  # Now handles everything: repo loading, indexing, LLM

        self._ensure_cache_dir()
        self._load_jobs()
        logger.info(f"RepositoryOnboardingService initialized with {len(self.jobs)} existing jobs")

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        cache_path = Path(settings.cache_dir)
        if not cache_path.exists():
            logger.info(f"Creating cache directory: {cache_path}")
        cache_path.mkdir(parents=True, exist_ok=True)

    def _load_jobs(self) -> None:
        """Load jobs from JSON file."""
        if self.jobs_file.exists():
            try:
                logger.info(f"Loading jobs from {self.jobs_file}")
                with open(self.jobs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.jobs = {
                        job_id: OnboardingJob.from_dict(job_data)
                        for job_id, job_data in data.items()
                    }
                logger.info(f"Successfully loaded {len(self.jobs)} jobs from file")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Could not load onboarding jobs file: {e}", exc_info=True)
                self.jobs = {}
        else:
            logger.info(f"No existing jobs file found at {self.jobs_file}")

    def _save_jobs(self) -> None:
        """Save jobs to JSON file."""
        try:
            logger.debug(f"Saving {len(self.jobs)} jobs to {self.jobs_file}")
            data = {
                job_id: job.to_dict()
                for job_id, job in self.jobs.items()
            }
            with open(self.jobs_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug("Jobs saved successfully")
        except Exception as e:
            logger.error(f"Error saving onboarding jobs to file: {e}", exc_info=True)

    def _update_job(self, job: OnboardingJob) -> None:
        """Update job in memory and persist to disk."""
        logger.debug(f"Updating job {job.job_id} (state: {job.current_state})")
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
        logger.info(f"Creating new onboarding job for repository: {repo_url}")
        job_id = str(uuid.uuid4())
        collection_name = create_collection_name(repo_url)

        job = OnboardingJob(
            job_id=job_id,
            repo_url=repo_url,
            collection_name=collection_name
        )

        self._update_job(job)
        logger.info(f"Job created successfully: {job_id} (collection: {collection_name})")
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
        logger.info(f"Starting onboarding job: {job_id}")
        job = self.get_job_status(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            raise OnboardingError(f"Job {job_id} not found")

        if job.current_state != OnboardingState.CREATED:
            logger.error(f"Job {job_id} cannot be started - already in state {job.current_state}")
            raise OnboardingError(f"Job {job_id} already started (state: {job.current_state})")

        # Start the workflow in background
        logger.info(f"Launching async workflow for job {job_id}")
        asyncio.create_task(self._process_job_async(job))

        return job

    async def _process_job_async(self, job: OnboardingJob) -> None:
        """
        Process a job through all states asynchronously.

        Args:
            job: The job to process
        """
        logger.info(f"Processing job {job.job_id} asynchronously from state {job.current_state}")
        state_machine = OnboardingStateMachine(job)

        try:
            # State: CLONING
            if job.current_state == OnboardingState.CREATED:
                logger.info(f"Job {job.job_id}: Transitioning from CREATED to CLONING")
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

            logger.info(f"Job {job.job_id} completed successfully in state {job.current_state}")

        except Exception as e:
            logger.error(f"Job {job.job_id} encountered error: {str(e)}", exc_info=True)
            self._handle_error(job, state_machine, e)

    async def _handle_cloning(self, job: OnboardingJob, sm: OnboardingStateMachine) -> None:
        """Handle the cloning/loading state using RAG engine."""
        logger.info(f"Job {job.job_id}: Starting CLONING phase for {job.repo_url}")
        sm.start_state(OnboardingState.CLONING)
        self._update_job(job)

        try:
            # Load repository using RAG engine (run in executor to avoid blocking)
            logger.debug(f"Job {job.job_id}: Loading repository using RAG engine")
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

            logger.info(f"Job {job.job_id}: Repository loaded successfully - {len(documents)} documents from {clone_path}")

            sm.complete_state(OnboardingState.CLONING, {
                "clone_path": clone_path,
                "documents_loaded": len(documents)
            })

            # Transition to parsing
            logger.info(f"Job {job.job_id}: Transitioning from CLONING to PARSING")
            sm.transition_to(OnboardingState.PARSING)
            self._update_job(job)

        except Exception as e:
            logger.error(f"Job {job.job_id}: Cloning failed - {str(e)}", exc_info=True)
            raise OnboardingError(f"Cloning error: {str(e)}")

    async def _handle_parsing(self, job: OnboardingJob, sm: OnboardingStateMachine) -> None:
        """
        Handle the parsing state - process LlamaIndex documents and stream chunks to vector store.
        This state now includes all indexing operations and metadata collection.
        """
        logger.info(f"Job {job.job_id}: Starting PARSING phase with {len(job.llama_documents)} documents")
        sm.start_state(OnboardingState.PARSING)
        self._update_job(job)

        try:
            # Step 1: Collect overview metadata from documents
            logger.debug(f"Job {job.job_id}: Collecting overview metadata from documents")
            loop = asyncio.get_event_loop()
            overview_metadata = await loop.run_in_executor(
                None,
                self.rag_engine.collect_overview_metadata,
                job.llama_documents
            )

            # Store metadata in job for later use in overview generation
            job.metadata_for_overview = overview_metadata.get("special_files", {})
            job.total_files = overview_metadata["stats"]["total_files"]
            job.languages_detected = list(overview_metadata["stats"]["languages"].keys())

            # Store additional metadata as JSON in job
            job.overview_metadata = overview_metadata

            logger.info(f"Job {job.job_id}: Collected metadata - {job.total_files} files, "
                       f"{len(job.metadata_for_overview)} special files, "
                       f"languages: {job.languages_detected}")

            # Step 2: Create index in vector store
            logger.debug(f"Job {job.job_id}: Creating index in collection {job.collection_name}")
            await loop.run_in_executor(
                None,
                self.rag_engine.create_index,
                job.repo_url,
                job.collection_name,
                job.llama_documents
            )

            logger.info(f"Job {job.job_id}: Parsing and indexing completed successfully")

            logger.info(f"Job {job.job_id}: Transitioning from PARSING to GENERATING_OVERVIEW")
            sm.transition_to(OnboardingState.GENERATING_OVERVIEW)
            self._update_job(job)

        except Exception as e:
            logger.error(f"Job {job.job_id}: Parsing/Indexing failed - {str(e)}", exc_info=True)
            raise OnboardingError(f"Parsing/Indexing error: {str(e)}")

    async def _handle_generating_overview(self, job: OnboardingJob, sm: OnboardingStateMachine) -> None:
        """
        Handle the overview generation state.
        Uses hybrid approach: file analysis + RAG queries.
        """
        logger.info(f"Job {job.job_id}: Starting GENERATING_OVERVIEW phase")
        sm.start_state(OnboardingState.GENERATING_OVERVIEW)
        self._update_job(job)

        try:
            loop = asyncio.get_event_loop()

            # Phase 1: File Analysis (already collected during parsing)
            logger.debug(f"Job {job.job_id}: Phase 1 - Analyzing collected files")
            file_data = await loop.run_in_executor(
                None,
                self._analyze_collected_files,
                job
            )
            logger.debug(f"Job {job.job_id}: File analysis complete - found {len(file_data)} data points")

            # Phase 2: RAG Queries
            logger.debug(f"Job {job.job_id}: Phase 2 - Querying RAG for insights")
            rag_data = await loop.run_in_executor(
                None,
                self._query_rag_for_insights,
                job.collection_name,
                file_data
            )
            logger.debug(f"Job {job.job_id}: RAG queries complete - {len(rag_data)} queries executed")

            # Phase 3: LLM Synthesis
            logger.debug(f"Job {job.job_id}: Phase 3 - Synthesizing overview with LLM")
            overview = await loop.run_in_executor(
                None,
                self._synthesize_overview,
                file_data,
                rag_data,
                job
            )

            job.project_overview = overview
            logger.info(f"Job {job.job_id}: Overview generated successfully ({len(overview)} characters)")

            sm.complete_state(OnboardingState.GENERATING_OVERVIEW, {
                "overview_length": len(overview),
                "file_data_keys": list(file_data.keys()),
                "rag_queries_made": len(rag_data)
            })

            # Transition to completed
            logger.info(f"Job {job.job_id}: Transitioning from GENERATING_OVERVIEW to COMPLETED")
            sm.transition_to(OnboardingState.COMPLETED)
            self._update_job(job)

        except Exception as e:
            logger.error(f"Job {job.job_id}: Overview generation failed - {str(e)}", exc_info=True)
            raise OnboardingError(f"Overview generation error: {str(e)}")

    def _analyze_collected_files(self, job: OnboardingJob) -> Dict[str, any]:
        """
        Analyze files collected during parsing using the structured metadata.

        Returns:
            Dictionary with extracted information for LLM synthesis
        """
        logger.debug(f"Job {job.job_id}: Analyzing collected metadata")

        # Use the new structured metadata from RAG engine
        metadata = job.overview_metadata

        # Build file_data from the structured metadata
        file_data = {
            "readme_content": "",
            "dependencies": metadata.get("package_info", {}).get("top_dependencies", []),
            "tech_stack": list(metadata.get("stats", {}).get("languages", {}).keys()),
            "config_files": [],
            "has_docker": metadata.get("patterns", {}).get("has_docker", False),
            "package_info": metadata.get("package_info", {}),
            "frameworks": metadata.get("patterns", {}).get("frameworks", []),
            "entry_points": metadata.get("patterns", {}).get("entry_points", []),
            "has_tests": metadata.get("patterns", {}).get("has_tests", False),
            "has_ci_cd": metadata.get("patterns", {}).get("has_ci_cd", False),
            "project_type": metadata.get("patterns", {}).get("project_type", "unknown"),
            "total_files": metadata.get("stats", {}).get("total_files", 0),
            "total_size_mb": round(metadata.get("stats", {}).get("total_size_bytes", 0) / 1024 / 1024, 2),
        }

        # Extract README from special files
        special_files = metadata.get("special_files", {})
        for file_path, content in special_files.items():
            filename_lower = Path(file_path).name.lower()

            if filename_lower.startswith('readme'):
                file_data["readme_content"] = content[:5000]
                logger.debug(f"Job {job.job_id}: Found README: {file_path}")
                break  # Use first README found

        # Collect config files
        file_data["config_files"] = [
            path for path in special_files.keys()
            if Path(path).name.lower() in ['cargo.toml', 'go.mod', 'pom.xml', 'pyproject.toml',
                                           'package.json', 'requirements.txt', 'dockerfile']
        ]

        logger.debug(f"Job {job.job_id}: File analysis complete - "
                    f"tech stack: {file_data['tech_stack']}, "
                    f"frameworks: {file_data['frameworks']}, "
                    f"project type: {file_data['project_type']}")

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
        logger.debug(f"Querying RAG engine for collection: {collection_name}")
        rag_data = {}

        queries = [
            ("entry_points", "What are the main entry points of this application? List main files, API endpoints, or CLI commands."),
            ("key_modules", "What are the key modules and components in this codebase? Describe their purpose."),
            ("architecture", "Describe the overall architecture of this project. What patterns are used?")
        ]

        for key, query in queries:
            try:
                logger.debug(f"Executing RAG query '{key}': {query}")
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
                    logger.debug(f"RAG query '{key}' returned {len(results)} results")
                else:
                    rag_data[key] = "Нет данных"
                    logger.debug(f"RAG query '{key}' returned no results")
            except Exception as e:
                rag_data[key] = f"Ошибка: {str(e)}"
                logger.warning(f"RAG query '{key}' failed: {str(e)}", exc_info=True)

        logger.debug("RAG queries completed")
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
        logger.debug(f"Job {job.job_id}: Synthesizing overview with LLM")

        # Build context for LLM with enhanced metadata
        context = f"""
Проанализируй следующую информацию о репозитории и создай структурированный обзор.

ИНФОРМАЦИЯ ИЗ ФАЙЛОВ:
README: {file_data.get('readme_content', 'Не найден')[:1000]}
Имя проекта: {file_data.get('package_info', {}).get('name', 'Неизвестно')}
Описание: {file_data.get('package_info', {}).get('description', 'Нет')}
Версия: {file_data.get('package_info', {}).get('version', 'Нет')}
Лицензия: {file_data.get('package_info', {}).get('license', 'Нет')}

СТАТИСТИКА:
Количество файлов: {file_data.get('total_files', 0)}
Размер кодовой базы: {file_data.get('total_size_mb', 0)} MB
Тип проекта: {file_data.get('project_type', 'unknown')}

ТЕХНОЛОГИЧЕСКИЙ СТЕК:
Языки программирования: {', '.join(file_data.get('tech_stack', []))}
Фреймворки: {', '.join(file_data.get('frameworks', []))}
Зависимости ({file_data.get('package_info', {}).get('dependencies_count', 0)}): {', '.join(file_data.get('dependencies', [])[:15])}

ИНФРАСТРУКТУРА:
Docker: {'Да' if file_data.get('has_docker') else 'Нет'}
CI/CD: {'Да' if file_data.get('has_ci_cd') else 'Нет'}
Тесты: {'Да' if file_data.get('has_tests') else 'Нет'}

ТОЧКИ ВХОДА:
{', '.join(file_data.get('entry_points', [])) if file_data.get('entry_points') else 'Не найдены'}

ФАЙЛЫ КОНФИГУРАЦИИ:
{', '.join([Path(f).name for f in file_data.get('config_files', [])])}

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

        logger.debug(f"Job {job.job_id}: Sending prompt to LLM ({len(prompt)} characters)")
        try:
            # Use RAG engine's LLM (configured in Settings.llm)
            from llama_index.core import Settings
            response = Settings.llm.complete(prompt)
            overview = str(response)
            logger.debug(f"Job {job.job_id}: LLM synthesis successful ({len(overview)} characters)")
            return overview
        except Exception as e:
            logger.error(f"Job {job.job_id}: LLM synthesis failed: {str(e)}", exc_info=True)
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
        logger.error(f"Job {job.job_id}: Handling error - {str(error)}", exc_info=True)
        job.error = str(error)
        sm.transition_to(OnboardingState.FAILED, triggered_by="error")
        self._update_job(job)
        logger.info(f"Job {job.job_id}: Transitioned to FAILED state")

    def get_job_status(self, job_id: str) -> Optional[OnboardingJob]:
        """
        Get the status of a job.

        Args:
            job_id: Job identifier

        Returns:
            OnboardingJob or None if not found
        """
        logger.debug(f"Retrieving status for job: {job_id}")
        job = self.jobs.get(job_id)
        if job:
            logger.debug(f"Job {job_id} found in state {job.current_state}")
        else:
            logger.debug(f"Job {job_id} not found")
        return job

    def list_jobs(self) -> List[OnboardingJob]:
        """
        List all onboarding jobs.

        Returns:
            List of all jobs
        """
        logger.debug(f"Listing all jobs - total count: {len(self.jobs)}")
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
        logger.info(f"Attempting to retry job: {job_id}")
        job = self.get_job_status(job_id)
        if not job:
            logger.error(f"Cannot retry - job not found: {job_id}")
            raise OnboardingError(f"Job {job_id} not found")

        sm = OnboardingStateMachine(job)

        if not sm.is_retryable():
            logger.error(f"Job {job_id} is not retryable (state: {job.current_state}, retries: {job.retry_count}/{job.max_retries})")
            raise OnboardingError(
                f"Job {job_id} cannot be retried "
                f"(state: {job.current_state}, retries: {job.retry_count}/{job.max_retries})"
            )

        # Get state to retry from
        retry_state = sm.get_retry_state()

        # Increment retry count
        job.retry_count += 1
        job.error = None
        logger.info(f"Job {job_id}: Retry attempt {job.retry_count}/{job.max_retries}, transitioning to {retry_state}")

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
        logger.info(f"Scheduling retry for job: {job_id}")
        # Create task in background
        asyncio.create_task(self.retry_job_async(job_id))

        # Return current job state
        job = self.get_job_status(job_id)
        if not job:
            logger.error(f"Job {job_id} not found after scheduling retry")
            raise OnboardingError(f"Job {job_id} not found")
        return job