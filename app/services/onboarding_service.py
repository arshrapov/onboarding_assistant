"""
Repository onboarding service - orchestrates the full onboarding workflow.
"""

import uuid
import json
import asyncio
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from app.core.models import OnboardingJob, OnboardingState, CodeChunk
from app.core.exceptions import OnboardingError, StateTransitionError
from app.services.state_machine import OnboardingStateMachine
from app.services.repo_downloader import RepositoryDownloader
from app.services.rag_engine import RAGEngine
from app.services.vector_store import create_collection_name
from app.utils.file_filter import filter_repository_files, count_files_by_language
from app.utils.file_parser import parse_file
from app.config import settings


class RepositoryOnboardingService:
    """Service for managing repository onboarding jobs."""

    def __init__(self):
        """Initialize the onboarding service."""
        self.jobs: Dict[str, OnboardingJob] = {}
        self.jobs_file = Path(settings.cache_dir) / "onboarding_jobs.json"
        self.downloader = RepositoryDownloader()
        self.rag_engine = RAGEngine()

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

        except Exception as e:
            self._handle_error(job, state_machine, e)

    async def _handle_cloning(self, job: OnboardingJob, sm: OnboardingStateMachine) -> None:
        """Handle the cloning state."""
        sm.start_state(OnboardingState.CLONING)
        self._update_job(job)

        try:
            # Clone repository (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            clone_path = await loop.run_in_executor(
                None,
                self.downloader.clone_repository,
                job.repo_url
            )
            job.clone_path = clone_path

            sm.complete_state(OnboardingState.CLONING, {
                "clone_path": clone_path
            })

            # Transition to parsing
            sm.transition_to(OnboardingState.PARSING)
            self._update_job(job)

        except Exception as e:
            raise OnboardingError(f"Cloning error: {str(e)}")

    async def _handle_parsing(self, job: OnboardingJob, sm: OnboardingStateMachine) -> None:
        """
        Handle the parsing state - filter files, parse, and stream chunks to vector store.
        This state now includes all indexing operations.
        """
        sm.start_state(OnboardingState.PARSING)
        self._update_job(job)

        try:
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

            # Parse files and stream chunks directly to vector store
            processed_files = 0
            total_chunks = 0
            chunk_batch: List[CodeChunk] = []
            BATCH_SIZE = 50  # Process chunks in batches

            for file_path in files:
                try:
                    # Parse file
                    chunks = await loop.run_in_executor(
                        None,
                        parse_file,
                        file_path,
                        repo_path,
                        job.repo_url
                    )

                    
                    chunk_batch.extend(chunks)
                    total_chunks += len(chunks)
                    processed_files += 1

                    # When batch is full, add to vector store
                    if len(chunk_batch) >= BATCH_SIZE:
                        await loop.run_in_executor(
                            None,
                            self.rag_engine.add_chunks,
                            job.collection_name,
                            chunk_batch
                        )
                        chunk_batch = []  # Clear batch

                    # Update progress periodically
                    if processed_files % 10 == 0:
                        metrics = sm.get_state_metrics(OnboardingState.PARSING)
                        metrics.data["processed_files"] = processed_files
                        metrics.data["chunks_created"] = total_chunks
                        self._update_job(job)

                except Exception as e:
                    # Skip failed files, log them
                    job.failed_files.append(str(file_path))

            # Add remaining chunks in batch
            if chunk_batch:
                await loop.run_in_executor(
                    None,
                    self.rag_engine.add_chunks,
                    job.collection_name,
                    chunk_batch
                )

            job.total_chunks = total_chunks

            sm.complete_state(OnboardingState.PARSING, {
                "processed_files": processed_files,
                "chunks_created": total_chunks,
                "chunks_indexed": total_chunks,
                "failed_files": len(job.failed_files),
                "languages": language_counts
            })

            # Transition directly to completed (no separate indexing state)
            sm.transition_to(OnboardingState.COMPLETED)
            self._update_job(job)

        except Exception as e:
            raise OnboardingError(f"Parsing/Indexing error: {str(e)}")

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