# ai_agent_framework/core/workflow/__init__.py

"""
Workflow Package

Defines workflow patterns (Chain, Router, Parallel, etc.) and orchestration components.
"""

from .base import BaseWorkflow
from .chain import PromptChain
from .router import Router
from .parallel import ParallelWorkflow, BranchDefinition # Updated import
from .evaluator import EvaluatorOptimizer, EvaluationCriterion, EvaluationMetric # Add evaluator components if used directly
# Import other workflow components as needed (e.g., Task, WorkerPool, Orchestrator are often used by agents/interfaces)
from .task import Task, TaskStatus
from .workflow import Workflow
from .worker_pool import WorkerPool, Worker, WorkerStatus
from .orchestrator import Orchestrator
# Import metrics collector if needed
from .metrics_collector import MetricsCollector

__all__ = [
    # Base
    "BaseWorkflow",
    # Patterns
    "PromptChain",
    "Router",
    "ParallelWorkflow", # Ensure ParallelWorkflow is exported
    "BranchDefinition", # Ensure BranchDefinition is exported
    "EvaluatorOptimizer",
    "EvaluationCriterion",
    "EvaluationMetric",
    # Orchestration
    "Task",
    "TaskStatus",
    "Workflow",
    "WorkerPool",
    "Worker",
    "WorkerStatus",
    "Orchestrator",
    # Metrics
    "MetricsCollector",
]