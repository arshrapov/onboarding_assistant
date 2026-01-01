"""
State machine for managing onboarding job state transitions.
"""

from typing import Dict, List
from datetime import datetime

from app.core.models import OnboardingJob, OnboardingState, StateTransition, StateMetrics


# State transition rules
ALLOWED_TRANSITIONS: Dict[OnboardingState, List[OnboardingState]] = {
    OnboardingState.CREATED: [
        OnboardingState.CLONING,
        OnboardingState.FAILED
    ],
    OnboardingState.CLONING: [
        OnboardingState.PARSING,
        OnboardingState.FAILED,
        OnboardingState.CLONING  # Allow retry
    ],
    OnboardingState.PARSING: [
        OnboardingState.COMPLETED,
        OnboardingState.FAILED,
        OnboardingState.PARSING  # Allow retry
    ],
    OnboardingState.FAILED: [
        # Can retry from failed state back to any state
        OnboardingState.CLONING,
        OnboardingState.PARSING
    ],
    OnboardingState.COMPLETED: [
        # Terminal state - no transitions allowed
    ]
}


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class OnboardingStateMachine:
    """Manages state transitions and validation for onboarding jobs."""

    def __init__(self, job: OnboardingJob):
        """
        Initialize state machine for a job.

        Args:
            job: The onboarding job to manage
        """
        self.job = job

    def can_transition_to(self, to_state: OnboardingState) -> bool:
        """
        Check if transition to the given state is allowed.

        Args:
            to_state: Target state

        Returns:
            True if transition is allowed, False otherwise
        """
        allowed = ALLOWED_TRANSITIONS.get(self.job.current_state, [])
        return to_state in allowed

    def transition_to(
        self,
        to_state: OnboardingState,
        triggered_by: str = "system",
        metadata: Dict = None
    ) -> None:
        """
        Execute a state transition.

        Args:
            to_state: Target state
            triggered_by: What triggered this transition
            metadata: Additional metadata for the transition

        Raises:
            StateTransitionError: If transition is not allowed
        """
        if not self.can_transition_to(to_state):
            raise StateTransitionError(
                f"Cannot transition from {self.job.current_state} to {to_state}. "
                f"Allowed transitions: {ALLOWED_TRANSITIONS.get(self.job.current_state, [])}"
            )

        # Record the transition
        transition = StateTransition(
            from_state=self.job.current_state,
            to_state=to_state,
            timestamp=datetime.utcnow(),
            triggered_by=triggered_by,
            metadata=metadata or {}
        )

        # Update job state
        self.job.state_history.append(transition)
        self.job.current_state = to_state
        self.job.updated_at = datetime.utcnow()

        # If transitioning to FAILED, record the error state
        if to_state == OnboardingState.FAILED:
            self.job.last_error_state = transition.from_state

        # If transitioning to COMPLETED, set completion time
        if to_state == OnboardingState.COMPLETED:
            self.job.completed_at = datetime.utcnow()

    def get_allowed_transitions(self) -> List[OnboardingState]:
        """
        Get list of valid next states from current state.

        Returns:
            List of allowed target states
        """
        return ALLOWED_TRANSITIONS.get(self.job.current_state, [])

    def start_state(self, state: OnboardingState) -> None:
        """
        Mark a state as started, initializing its metrics.

        Args:
            state: The state being started
        """
        if state.value not in self.job.state_metrics:
            self.job.state_metrics[state.value] = StateMetrics(state=state)

        metrics = self.job.state_metrics[state.value]
        metrics.started_at = datetime.utcnow()
        metrics.attempts += 1

    def complete_state(self, state: OnboardingState, data: Dict = None) -> None:
        """
        Mark a state as completed with optional result data.

        Args:
            state: The state being completed
            data: Optional state-specific result data
        """
        if state.value not in self.job.state_metrics:
            self.job.state_metrics[state.value] = StateMetrics(state=state)

        metrics = self.job.state_metrics[state.value]
        metrics.completed_at = datetime.utcnow()

        if metrics.started_at:
            duration = (metrics.completed_at - metrics.started_at).total_seconds()
            metrics.duration_seconds = duration

        if data:
            metrics.data.update(data)

    def get_state_metrics(self, state: OnboardingState) -> StateMetrics:
        """
        Get metrics for a specific state.

        Args:
            state: The state to get metrics for

        Returns:
            StateMetrics object, or a new empty one if not found
        """
        if state.value not in self.job.state_metrics:
            self.job.state_metrics[state.value] = StateMetrics(state=state)

        return self.job.state_metrics[state.value]

    def is_terminal_state(self) -> bool:
        """
        Check if the job is in a terminal state.

        Returns:
            True if in COMPLETED or FAILED state
        """
        return self.job.current_state in [
            OnboardingState.COMPLETED,
            OnboardingState.FAILED
        ]
