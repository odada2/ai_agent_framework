"""
Asynchronous Task Queue

This module provides a task queue system for asynchronous and parallel
execution of tasks by agents in the workflow.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Set, Tuple
from dataclasses import dataclass, field
import uuid
import heapq

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status values for tasks in the queue."""
    PENDING = "pending"     # Task is waiting to be executed
    RUNNING = "running"     # Task is currently running
    COMPLETED = "completed" # Task completed successfully
    FAILED = "failed"       # Task failed with an error
    CANCELED = "canceled"   # Task was canceled before execution
    TIMEOUT = "timeout"     # Task timed out during execution


@dataclass(order=True)
class Task:
    """
    A task to be executed asynchronously.
    
    The Task class represents a unit of work that can be queued for
    execution. Tasks include metadata for tracking and priorities
    for execution order.
    """
    priority: int  # Lower numbers have higher priority
    created_at: float
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    func: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    result: Any = field(default=None, compare=False)
    error: Optional[Exception] = field(default=None, compare=False)
    timeout: Optional[float] = field(default=None, compare=False)
    tags: List[str] = field(default_factory=list, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    submitted_by: Optional[str] = field(default=None, compare=False)
    dependencies: List[str] = field(default_factory=list, compare=False)
    retries: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    started_at: Optional[float] = field(default=None, compare=False)
    completed_at: Optional[float] = field(default=None, compare=False)
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time if task has completed."""
        if self.started_at is not None and self.completed_at is not None:
            return self.completed_at - self.started_at
        return None
    
    @property
    def waiting_time(self) -> float:
        """Calculate waiting time in queue."""
        start = self.started_at or time.time()
        return start - self.created_at
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed (successfully or not)."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, 
                              TaskStatus.CANCELED, TaskStatus.TIMEOUT)
    
    @property
    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def can_execute(self) -> bool:
        """Check if task can be executed based on dependencies."""
        return self.status == TaskStatus.PENDING and not self.dependencies
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": self.execution_time,
            "waiting_time": self.waiting_time,
            "result": str(self.result) if self.result is not None else None,
            "error": str(self.error) if self.error is not None else None,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "submitted_by": self.submitted_by,
            "metadata": self.metadata
        }


class TaskQueue:
    """
    Asynchronous task queue for managing task execution.
    
    The TaskQueue handles task prioritization, dependency resolution,
    parallel execution, error handling, and task tracking.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        default_timeout: Optional[float] = None,
        retry_failed: bool = True
    ):
        """
        Initialize the task queue.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrently running tasks
            default_timeout: Default timeout for tasks in seconds, None for no timeout
            retry_failed: Whether to automatically retry failed tasks
        """
        self.queue: List[Task] = []  # Priority queue
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}  # task_id -> set of dependent task_ids
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.retry_failed = retry_failed
        self.running = False
        self.worker_task = None
        self._task_added_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the task queue worker."""
        if self.running:
            return
            
        self.running = True
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("Task queue worker started")
    
    async def stop(self) -> None:
        """Stop the task queue worker and wait for completion."""
        if not self.running:
            return
            
        self.running = False
        if self.worker_task:
            self._task_added_event.set()  # Wake up worker if it's waiting
            await self.worker_task
            self.worker_task = None
            
        logger.info("Task queue worker stopped")
    
    async def add_task(
        self,
        func: Callable[..., Awaitable[Any]],
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        name: Optional[str] = None,
        priority: int = 10,
        timeout: Optional[float] = None,
        task_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        submitted_by: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """
        Add a task to the queue.
        
        Args:
            func: Coroutine function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            name: Optional name for the task
            priority: Priority (lower numbers = higher priority)
            timeout: Timeout in seconds, None for default
            task_id: Optional custom task ID
            tags: Optional list of tags for categorization
            metadata: Optional additional metadata
            dependencies: Optional list of task IDs this task depends on
            submitted_by: Optional identifier of the task submitter
            max_retries: Maximum number of retry attempts
            
        Returns:
            Task ID
        """
        # Generate task ID if not provided
        task_id = task_id or str(uuid.uuid4())
        
        # Create a descriptive name if not provided
        if name is None:
            name = func.__name__ if hasattr(func, "__name__") else "anonymous_task"
        
        # Use default values for optional parameters
        kwargs = kwargs or {}
        tags = tags or []
        metadata = metadata or {}
        dependencies = dependencies or []
        
        # Filter out dependencies that don't exist or are already completed
        valid_dependencies = []
        for dep_id in dependencies:
            if dep_id in self.completed_tasks and self.completed_tasks[dep_id].is_successful:
                # Dependency already completed successfully, skip it
                continue
                
            # Check if dependency exists in the queue or running tasks
            dep_exists = False
            for task in self.queue:
                if task.task_id == dep_id:
                    dep_exists = True
                    break
                    
            if dep_id in self.running_tasks or dep_exists:
                valid_dependencies.append(dep_id)
            else:
                logger.warning(f"Dependency {dep_id} not found for task {task_id}, ignoring")
        
        # Create task
        task = Task(
            priority=priority,
            created_at=time.time(),
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            status=TaskStatus.PENDING,
            timeout=timeout if timeout is not None else self.default_timeout,
            tags=tags,
            metadata=metadata,
            dependencies=valid_dependencies,
            submitted_by=submitted_by,
            max_retries=max_retries
        )
        
        # Add to queue
        with self._lock():
            heapq.heappush(self.queue, task)
            
            # Update dependency graph
            for dep_id in valid_dependencies:
                if dep_id not in self.dependency_graph:
                    self.dependency_graph[dep_id] = set()
                self.dependency_graph[dep_id].add(task_id)
        
        # Signal worker thread
        self._task_added_event.set()
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            Task status dictionary or None if not found
        """
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
            
        # Check running tasks
        if task_id in self.running_tasks:
            for task in self.queue:
                if task.task_id == task_id and task.status == TaskStatus.RUNNING:
                    return task.to_dict()
        
        # Check queued tasks
        for task in self.queue:
            if task.task_id == task_id:
                return task.to_dict()
                
        return None
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for a specific task to complete.
        
        Args:
            task_id: ID of the task to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Task status dictionary
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
            KeyError: If task is not found
        """
        start_time = time.time()
        
        while True:
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
            
            # Check if task exists
            status = await self.get_task_status(task_id)
            if status is None:
                raise KeyError(f"Task {task_id} not found")
            
            # Check if task is completed
            if status["status"] in ["completed", "failed", "canceled", "timeout"]:
                return status
            
            # Wait a short time before checking again
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if it's still in the queue or running.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was found and canceled
        """
        # Check if task is running
        if task_id in self.running_tasks:
            # Cancel the task
            self.running_tasks[task_id].cancel()
            
            # Wait for cancellation to take effect
            try:
                await self.running_tasks[task_id]
            except asyncio.CancelledError:
                pass
            
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            # Find task in queue and mark as canceled
            for task in self.queue:
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELED
                    task.completed_at = time.time()
                    
                    # Move to completed tasks
                    self.completed_tasks[task_id] = task
                    
                    # Remove task's dependencies from other tasks
                    self._resolve_dependencies(task_id)
                    return True
                    
            return True
        
        # Check if task is in queue
        with self._lock():
            # Find task in queue
            for i, task in enumerate(self.queue):
                if task.task_id == task_id and task.status == TaskStatus.PENDING:
                    # Mark as canceled
                    task.status = TaskStatus.CANCELED
                    task.completed_at = time.time()
                    
                    # Move to completed tasks
                    self.completed_tasks[task_id] = task
                    
                    # Remove from queue (rebuild heap without this task)
                    self.queue = [t for t in self.queue if t.task_id != task_id]
                    heapq.heapify(self.queue)
                    
                    # Remove task's dependencies from other tasks
                    self._resolve_dependencies(task_id)
                    return True
        
        return False
    
    async def list_tasks(
        self,
        status: Optional[Union[TaskStatus, str]] = None,
        tag: Optional[str] = None,
        submitted_by: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List tasks filtered by criteria.
        
        Args:
            status: Filter by task status
            tag: Filter by tag
            submitted_by: Filter by submitter
            limit: Maximum number of tasks to return
            
        Returns:
            List of task dictionaries
        """
        results = []
        
        # Convert string status to enum if needed
        if isinstance(status, str):
            try:
                status = TaskStatus(status)
            except ValueError:
                status = None
        
        # Helper function to check if task matches filters
        def matches_filters(task: Task) -> bool:
            if status and task.status != status:
                return False
            if tag and tag not in task.tags:
                return False
            if submitted_by and task.submitted_by != submitted_by:
                return False
            return True
        
        # Add completed tasks matching filters
        for task_id, task in list(self.completed_tasks.items())[-limit:]:
            if matches_filters(task):
                results.append(task.to_dict())
                if len(results) >= limit:
                    break
        
        # Add running tasks
        for task in self.queue:
            if task.status == TaskStatus.RUNNING and matches_filters(task):
                results.append(task.to_dict())
                if len(results) >= limit:
                    break
                    
        # Add pending tasks
        for task in sorted(self.queue):  # Sorted by priority
            if task.status == TaskStatus.PENDING and matches_filters(task):
                results.append(task.to_dict())
                if len(results) >= limit:
                    break
        
        return results
    
    def _lock(self):
        """Simple context manager for modifying the queue safely."""
        # In a real system, you would use a proper lock here
        # For asyncio, consider using asyncio.Lock
        class DummyLock:
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
        return DummyLock()
    
    async def _worker(self) -> None:
        """Main worker loop for processing tasks."""
        while self.running:
            # Get all tasks that can be executed (no dependencies and within concurrency limit)
            executable_tasks = []
            
            with self._lock():
                # Check each task in priority order
                for task in sorted(self.queue):
                    if (task.status == TaskStatus.PENDING and 
                        not task.dependencies and 
                        len(self.running_tasks) + len(executable_tasks) < self.max_concurrent_tasks):
                        executable_tasks.append(task)
                        
                # Mark selected tasks as running
                for task in executable_tasks:
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
            
            # Start each executable task
            for task in executable_tasks:
                asyncio.create_task(self._execute_task(task))
            
            # If no tasks could be executed now, wait for a task to be added or completed
            if not executable_tasks:
                # Reset event before waiting
                self._task_added_event.clear()
                
                # Wait for new task or timeout
                try:
                    await asyncio.wait_for(self._task_added_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Just continue the loop
                    pass
    
    async def _execute_task(self, task: Task) -> None:
        """
        Execute a single task with timeout and error handling.
        
        Args:
            task: Task to execute
        """
        # Record in running tasks
        self.running_tasks[task.task_id] = asyncio.current_task()
        
        try:
            # Run with timeout if specified
            if task.timeout is not None:
                task.result = await asyncio.wait_for(
                    task.func(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                task.result = await task.func(*task.args, **task.kwargs)
                
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            
        except asyncio.TimeoutError:
            # Handle timeout
            task.status = TaskStatus.TIMEOUT
            task.error = asyncio.TimeoutError(f"Task timed out after {task.timeout} seconds")
            logger.warning(f"Task {task.task_id} ({task.name}) timed out")
            
        except asyncio.CancelledError:
            # Handle cancellation
            task.status = TaskStatus.CANCELED
            logger.info(f"Task {task.task_id} ({task.name}) was canceled")
            
        except Exception as e:
            # Handle other errors
            task.status = TaskStatus.FAILED
            task.error = e
            logger.error(f"Task {task.task_id} ({task.name}) failed: {str(e)}")
            
            # Handle retries
            if self.retry_failed and task.retries < task.max_retries:
                await self._retry_task(task)
                return
        
        finally:
            # Record completion time
            task.completed_at = time.time()
            
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            # Move to completed tasks if not being retried
            if task.status != TaskStatus.PENDING:
                self.completed_tasks[task.task_id] = task
                
                # Clean up old tasks if there are too many
                self._cleanup_old_tasks()
                
                # Resolve dependencies
                self._resolve_dependencies(task.task_id)
                
                # Signal that a task has completed
                self._task_added_event.set()
    
    async def _retry_task(self, task: Task) -> None:
        """
        Retry a failed task.
        
        Args:
            task: Failed task to retry
        """
        # Increment retry counter
        task.retries += 1
        
        # Reset status to pending
        task.status = TaskStatus.PENDING
        task.started_at = None
        task.completed_at = None
        task.error = None
        
        # Add delay before retry based on retry count (exponential backoff)
        backoff_time = 2 ** (task.retries - 1)  # 1, 2, 4, 8, ...
        await asyncio.sleep(backoff_time)
        
        logger.info(f"Retrying task {task.task_id} ({task.name}), attempt {task.retries} of {task.max_retries}")
        
        # Signal that a task is ready to be executed
        self._task_added_event.set()
    
    def _resolve_dependencies(self, task_id: str) -> None:
        """
        Resolve dependencies for a completed task.
        
        Args:
            task_id: ID of the completed task
        """
        # Check if any tasks depend on this one
        dependent_tasks = self.dependency_graph.get(task_id, set())
        
        # Remove this dependency from dependent tasks
        for dep_task_id in dependent_tasks:
            # Find the task
            for task in self.queue:
                if task.task_id == dep_task_id:
                    # Remove the dependency
                    if task_id in task.dependencies:
                        task.dependencies.remove(task_id)
                    break
        
        # Remove from dependency graph
        if task_id in self.dependency_graph:
            del self.dependency_graph[task_id]
            
        # Signal that dependencies have changed
        self._task_added_event.set()
    
    def _cleanup_old_tasks(self, max_completed: int = 1000) -> None:
        """
        Clean up old completed tasks to prevent memory leaks.
        
        Args:
            max_completed: Maximum number of completed tasks to keep
        """
        if len(self.completed_tasks) > max_completed:
            # Sort by completion time and keep only the most recent
            sorted_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].completed_at or 0,
                reverse=True
            )
            
            # Keep only the most recent tasks
            self.completed_tasks = dict(sorted_tasks[:max_completed])


# Global task queue for module-level access
global_task_queue = TaskQueue()


async def schedule_task(
    func: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> str:
    """
    Schedule a task for execution using the global task queue.
    
    This is a convenience function for adding tasks to the global queue.
    All parameters are passed to TaskQueue.add_task().
    
    Returns:
        Task ID
    """
    # Extract task parameters from kwargs
    task_kwargs = {
        "name": kwargs.pop("name", None),
        "priority": kwargs.pop("priority", 10),
        "timeout": kwargs.pop("timeout", None),
        "task_id": kwargs.pop("task_id", None),
        "tags": kwargs.pop("tags", None),
        "metadata": kwargs.pop("metadata", None),
        "dependencies": kwargs.pop("dependencies", None),
        "submitted_by": kwargs.pop("submitted_by", None),
        "max_retries": kwargs.pop("max_retries", 3)
    }
    
    # Ensure global queue is running
    if not global_task_queue.running:
        await global_task_queue.start()
    
    # Add task to queue
    return await global_task_queue.add_task(func, args, kwargs, **task_kwargs)


async def wait_for_task(task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    Wait for a task to complete using the global task queue.
    
    Args:
        task_id: ID of the task to wait for
        timeout: Maximum time to wait in seconds
        
    Returns:
        Task status dictionary
    """
    return await global_task_queue.wait_for_task(task_id, timeout)


async def cancel_task(task_id: str) -> bool:
    """
    Cancel a task using the global task queue.
    
    Args:
        task_id: ID of the task to cancel
        
    Returns:
        True if task was found and canceled
    """
    return await global_task_queue.cancel_task(task_id)


async def list_tasks(
    status: Optional[Union[TaskStatus, str]] = None,
    tag: Optional[str] = None,
    submitted_by: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List tasks using the global task queue.
    
    Args:
        status: Filter by task status
        tag: Filter by tag
        submitted_by: Filter by submitter
        limit: Maximum number of tasks to return
        
    Returns:
        List of task dictionaries
    """
    return await global_task_queue.list_tasks(status, tag, submitted_by, limit)