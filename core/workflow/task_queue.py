"""
Asynchronous Task Queue

This module provides a task queue system for asynchronous and parallel
execution of tasks by agents in the workflow. Uses asyncio.Lock for safety.
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
    func: Callable[..., Awaitable[Any]] = field(compare=False) # Ensure func is Awaitable
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
        start = self.started_at or time.time() # Use monotonic time if possible
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
        # Ensure error is serializable
        error_str = None
        if self.error is not None:
            try:
                error_str = str(self.error)
            except Exception:
                error_str = f"Unserializable error of type {type(self.error).__name__}"

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
            # Attempt simple str conversion for result, avoid complex objects
            "result": str(self.result) if self.result is not None else None,
            "error": error_str,
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
    parallel execution, error handling, and task tracking using asyncio.Lock.
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
        self.queue: List[Task] = []  # Priority queue (min-heap)
        # Use asyncio.Lock for async safety
        self._queue_lock = asyncio.Lock()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        # completed_tasks should also be protected by lock if accessed/modified concurrently
        self.completed_tasks: Dict[str, Task] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}  # task_id -> set of dependent task_ids
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.retry_failed = retry_failed
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
        # Event to signal when tasks are added or completed
        self._condition = asyncio.Condition(lock=self._queue_lock)

    async def start(self) -> None:
        """Start the task queue worker."""
        if self.running:
            return

        self.running = True
        self.worker_task = asyncio.create_task(self._worker(), name="TaskQueueWorker")
        logger.info("Task queue worker started")

    async def stop(self) -> None:
        """Stop the task queue worker and wait for completion."""
        if not self.running:
            return

        self.running = False
        async with self._condition: # Acquire lock to notify
            self._condition.notify_all() # Wake up worker if waiting

        if self.worker_task:
            try:
                await asyncio.wait_for(self.worker_task, timeout=5.0) # Give worker time to finish current cycle
            except asyncio.TimeoutError:
                 logger.warning("Task queue worker did not stop gracefully, cancelling.")
                 self.worker_task.cancel()
            except asyncio.CancelledError:
                 pass # Expected if stop is called rapidly
            self.worker_task = None

        logger.info("Task queue worker stopped")

    async def add_task(
        self,
        func: Callable[..., Awaitable[Any]],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None, # Use Optional
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
            func: Coroutine function to execute.
            args: Positional arguments for the function.
            kwargs: Keyword arguments for the function.
            name: Optional name for the task.
            priority: Priority (lower numbers = higher priority).
            timeout: Timeout in seconds, None for default.
            task_id: Optional custom task ID.
            tags: Optional list of tags for categorization.
            metadata: Optional additional metadata.
            dependencies: Optional list of task IDs this task depends on.
            submitted_by: Optional identifier of the task submitter.
            max_retries: Maximum number of retry attempts.

        Returns:
            Task ID.
        """
        if not asyncio.iscoroutinefunction(func):
             raise TypeError("Task function must be an awaitable coroutine.")

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

        async with self._queue_lock: # Acquire lock for checking dependencies and adding task
            # Filter out dependencies that don't exist or are already completed
            valid_dependencies = []
            for dep_id in dependencies:
                completed_task = self.completed_tasks.get(dep_id)
                if completed_task and completed_task.is_successful:
                    # Dependency already completed successfully, skip it
                    continue

                # Check if dependency exists in the queue or running tasks
                dep_exists_in_queue = any(t.task_id == dep_id for t in self.queue)

                if dep_id in self.running_tasks or dep_exists_in_queue:
                    valid_dependencies.append(dep_id)
                else:
                    # Check completed tasks again in case it finished between checks
                    if not (completed_task and completed_task.is_successful):
                        logger.warning(f"Dependency '{dep_id}' not found or not successful for task '{task_id}', ignoring dependency.")

            # Create task
            task = Task(
                priority=priority,
                created_at=time.monotonic(), # Use monotonic time
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
            heapq.heappush(self.queue, task)
            logger.debug(f"Added task '{task_id}' ({name}) to queue.")

            # Update dependency graph
            for dep_id in valid_dependencies:
                if dep_id not in self.dependency_graph:
                    self.dependency_graph[dep_id] = set()
                self.dependency_graph[dep_id].add(task_id)

            # Signal worker thread that a task was added
            self._condition.notify()

        return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.

        Args:
            task_id: ID of the task to check

        Returns:
            Task status dictionary or None if not found
        """
        async with self._queue_lock: # Lock needed to access shared completed_tasks and queue
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].to_dict()

            # Check running tasks (task object is still in self.queue when running)
            # and queued tasks
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
            KeyError: If task is not found initially
        """
        start_time = time.monotonic()

        # First check if task exists at all
        initial_status = await self.get_task_status(task_id)
        if initial_status is None:
             raise KeyError(f"Task '{task_id}' not found in queue or completed list.")

        # Now wait for completion
        while True:
            # Calculate remaining time
            elapsed = time.monotonic() - start_time
            remaining_timeout = None
            if timeout is not None:
                remaining_timeout = timeout - elapsed
                if remaining_timeout <= 0:
                     raise asyncio.TimeoutError(f"Timeout waiting for task '{task_id}'")

            async with self._condition: # Use condition to wait efficiently
                status = await self.get_task_status(task_id) # Re-check status under lock
                if status is None: # Should not happen if initial check passed, but handle defensively
                     raise KeyError(f"Task '{task_id}' disappeared unexpectedly.")

                if status["status"] in ["completed", "failed", "canceled", "timeout"]:
                    return status

                # Wait for notification or timeout
                try:
                    # Wait on the condition variable. This releases the lock while waiting.
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining_timeout)
                except asyncio.TimeoutError:
                     # If wait_for times out, check status one last time before raising
                     final_status = await self.get_task_status(task_id)
                     if final_status["status"] in ["completed", "failed", "canceled", "timeout"]:
                         return final_status
                     else:
                          raise asyncio.TimeoutError(f"Timeout waiting for task '{task_id}'")


    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if it's still pending or running.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if task was found and action was taken (marked canceled or cancellation requested).
        """
        async with self._queue_lock:
            # Check if task is pending in the queue
            task_in_queue = None
            task_index = -1
            for i, t in enumerate(self.queue):
                if t.task_id == task_id:
                    task_in_queue = t
                    task_index = i
                    break

            if task_in_queue and task_in_queue.status == TaskStatus.PENDING:
                logger.info(f"Cancelling pending task '{task_id}'.")
                task_in_queue.status = TaskStatus.CANCELED
                task_in_queue.completed_at = time.monotonic()
                # Move to completed tasks
                self.completed_tasks[task_id] = task_in_queue
                # Remove from queue (using index for efficiency if heap property not critical here)
                self.queue.pop(task_index)
                heapq.heapify(self.queue) # Restore heap property
                # Resolve dependencies that depended on this canceled task
                await self._resolve_dependencies(task_id, success=False) # Indicate failure for dependents?
                self._condition.notify_all() # Notify worker loop
                return True

            # Check if task is running
            if task_id in self.running_tasks:
                 # Note: The task object itself is still in self.queue while running
                 running_task_instance = self.running_tasks[task_id]
                 if not running_task_instance.done():
                     logger.info(f"Requesting cancellation for running task '{task_id}'.")
                     was_cancelled = running_task_instance.cancel()
                     # The actual status update (to CANCELED/FAILED etc.) will happen
                     # in _execute_task when the cancellation is processed.
                     # We don't remove it from running_tasks here; _execute_task handles that.
                     return was_cancelled # Return if cancellation was successfully requested
                 else:
                     logger.warning(f"Task '{task_id}' is in running_tasks but already done. Cannot cancel.")
                     return False # Already finished, cannot cancel

        # Check completed tasks (cannot cancel)
        async with self._queue_lock:
            if task_id in self.completed_tasks:
                 logger.warning(f"Cannot cancel task '{task_id}': already completed with status {self.completed_tasks[task_id].status.value}")
                 return False

        logger.warning(f"Task '{task_id}' not found for cancellation.")
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
            status: Filter by task status (enum or string).
            tag: Filter by tag.
            submitted_by: Filter by submitter.
            limit: Maximum number of tasks to return.

        Returns:
            List of task dictionaries.
        """
        results = []

        # Convert string status to enum if needed
        status_enum = None
        if isinstance(status, str):
            try:
                status_enum = TaskStatus(status.lower())
            except ValueError:
                logger.warning(f"Invalid status string '{status}' provided to list_tasks. Ignoring status filter.")
                status_enum = None
        elif isinstance(status, TaskStatus):
             status_enum = status


        # Helper function to check if task matches filters
        def matches_filters(task: Task) -> bool:
            if status_enum and task.status != status_enum:
                return False
            if tag and tag not in task.tags:
                return False
            if submitted_by and task.submitted_by != submitted_by:
                return False
            return True

        async with self._queue_lock: # Lock for accessing shared lists
            # Combine queued, running, and completed tasks for filtering
            all_tasks = list(self.queue) + list(self.completed_tasks.values())
            # Sort primarily by creation time descending to show recent tasks first
            all_tasks.sort(key=lambda t: t.created_at, reverse=True)

            for task in all_tasks:
                if matches_filters(task):
                    results.append(task.to_dict())
                    if len(results) >= limit:
                        break

        return results


    async def _worker(self) -> None:
        """Main worker loop for processing tasks."""
        while self.running:
            executable_task: Optional[Task] = None
            async with self._condition: # Acquire lock for accessing queue
                while not executable_task and self.running: # Loop until a task is found or stopped
                    # Find the highest priority task that is ready
                    ready_candidates = []
                    temp_removed = []
                    while self.queue:
                        task = heapq.heappop(self.queue)
                        if task.status == TaskStatus.PENDING and not task.dependencies:
                            if len(self.running_tasks) < self.max_concurrent_tasks:
                                ready_candidates.append(task)
                                # Check if this is the best candidate so far
                                if not executable_task or task.priority < executable_task.priority:
                                     executable_task = task
                                # Break if we found one and don't need to check lower priority ones?
                                # No, keep checking for same priority but earlier created_at
                            else:
                                # Put back if no capacity, still pending
                                temp_removed.append(task)
                        elif task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                            # Put back pending (with deps) or running tasks
                            temp_removed.append(task)
                        # Ignore completed/failed/canceled tasks in the heap

                    # Put back tasks that were not selected or are not ready/running
                    for t in temp_removed:
                        heapq.heappush(self.queue, t)
                    # Put back candidates not chosen (if any)
                    for t in ready_candidates:
                         if t != executable_task:
                              heapq.heappush(self.queue, t)

                    if executable_task:
                        logger.debug(f"Worker found executable task: {executable_task.task_id}")
                        executable_task.status = TaskStatus.RUNNING
                        executable_task.started_at = time.monotonic()
                        # Add to running_tasks dict - no lock needed as only worker modifies it here
                        self.running_tasks[executable_task.task_id] = asyncio.current_task() # Store worker task ref? Or task obj ref? Store asyncio task.
                        break # Exit wait loop

                    # If no task found, wait for notification
                    logger.debug("Worker waiting for tasks or completion...")
                    try:
                        # Wait for a notification (task added/completed) or timeout
                        await asyncio.wait_for(self._condition.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass # Timeout is fine, just re-check queue

            # Execute the task outside the lock
            if executable_task:
                # Create the task execution coroutine
                exec_coro = self._execute_task(executable_task)
                # Schedule it to run independently
                asyncio.create_task(exec_coro, name=f"Task-{executable_task.task_id}")
            elif not self.running:
                 break # Exit worker loop if stopped


    async def _execute_task(self, task: Task) -> None:
        """
        Execute a single task with timeout and error handling.

        Args:
            task: Task to execute
        """
        logger.info(f"Executing task '{task.task_id}' ({task.name})...")
        task_future = None
        try:
            # Create the coroutine for the task function
            task_coro = task.func(*task.args, **task.kwargs)

            # Run with timeout if specified
            if task.timeout is not None:
                task_future = asyncio.wait_for(task_coro, timeout=task.timeout)
            else:
                task_future = task_coro

            # Await the future
            task.result = await task_future
            task.status = TaskStatus.COMPLETED
            logger.info(f"Task '{task.task_id}' ({task.name}) completed successfully.")

        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = asyncio.TimeoutError(f"Task timed out after {task.timeout} seconds")
            logger.warning(f"Task '{task.task_id}' ({task.name}) timed out.")
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELED
            task.error = asyncio.CancelledError(f"Task '{task.task_id}' was cancelled.")
            logger.info(f"Task '{task.task_id}' ({task.name}) was canceled.")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = e
            logger.error(f"Task '{task.task_id}' ({task.name}) failed: {e}", exc_info=self.running) # Log traceback if running
            # Retry logic moved here
            if self.retry_failed and task.retries < task.max_retries:
                await self._retry_task(task) # This will reset status to PENDING and notify
                return # Don't proceed to finalization steps below if retrying

        # --- Finalization (runs if task completed, failed, canceled, or timed out AND not retrying) ---
        task.completed_at = time.monotonic()

        async with self._condition: # Acquire lock to update shared state
            # Remove from running tasks dict
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]

            # Move to completed tasks (this handles final failed/canceled/timeout states too)
            self.completed_tasks[task_id] = task

            # Clean up old tasks if needed (keep lock minimal)
            # Maybe move cleanup to a separate periodic task?
            self._cleanup_old_tasks_nolock(max_completed=500) # Limit completed task history

            # Resolve dependencies only on successful completion or if configured differently
            # Should dependents run if a dependency fails/is cancelled? Depends on workflow.
            # Assuming dependents only run on SUCCESSFUL completion for now.
            if task.status == TaskStatus.COMPLETED:
                await self._resolve_dependencies(task.task_id, success=True)
            else:
                 # If a task fails/is cancelled, mark downstream dependents as failed/blocked?
                 await self._resolve_dependencies(task.task_id, success=False) # Notify dependents of failure/cancellation


            # Notify worker loop that a task slot might be free or dependencies resolved
            self._condition.notify()


    async def _retry_task(self, task: Task) -> None:
        """
        Prepare a failed task for retry. Resets status and re-queues.

        Args:
            task: Failed task to retry.
        """
        task.retries += 1
        backoff_time = min(60, 2 ** task.retries) # Exponential backoff with cap
        logger.info(f"Retrying task '{task.task_id}' ({task.name}) (attempt {task.retries}/{task.max_retries}) after {backoff_time:.1f}s delay.")

        await asyncio.sleep(backoff_time)

        async with self._condition: # Acquire lock
            # Reset status for re-queueing
            task.status = TaskStatus.PENDING
            task.started_at = None
            task.completed_at = None
            task.error = None
            task.result = None # Clear previous result/error

            # Remove from running_tasks if somehow still there
            if task.task_id in self.running_tasks:
                 del self.running_tasks[task.task_id]

            # Re-add to the priority queue
            heapq.heappush(self.queue, task)

            # Notify worker that a task is ready
            self._condition.notify()

    async def _resolve_dependencies(self, completed_task_id: str, success: bool) -> None:
        """
        Update tasks that depended on the completed/failed task.

        Args:
            completed_task_id: ID of the task that finished.
            success: Whether the completed task was successful.
        """
        # This method assumes the lock (_condition's lock) is already held

        dependent_task_ids = self.dependency_graph.pop(completed_task_id, set())
        if not dependent_task_ids:
            return # No tasks depended on this one

        logger.debug(f"Resolving dependencies for {len(dependent_task_ids)} tasks dependent on {completed_task_id} (Success: {success})")

        tasks_to_notify = False
        for task in self.queue: # Iterate through tasks still in queue (pending or running)
            if task.task_id in dependent_task_ids:
                if success:
                    if completed_task_id in task.dependencies:
                        task.dependencies.remove(completed_task_id)
                        logger.debug(f"Removed dependency '{completed_task_id}' from task '{task.task_id}'. Remaining: {task.dependencies}")
                        # Check if this task is now ready
                        if task.status == TaskStatus.PENDING and not task.dependencies:
                            tasks_to_notify = True # Signal worker to check queue
                else:
                    # Handle dependency failure/cancellation
                    if task.status == TaskStatus.PENDING:
                         logger.warning(f"Marking task '{task.task_id}' as FAILED due to failed/canceled dependency '{completed_task_id}'.")
                         task.status = TaskStatus.FAILED
                         task.error = Exception(f"Dependency '{completed_task_id}' failed or was canceled.")
                         task.completed_at = time.monotonic()
                         self.completed_tasks[task.task_id] = task # Move to completed
                         # Need to remove this failed task from the heap
                         # This is tricky, requires rebuilding or marking and filtering later.
                         # For simplicity, we might rely on the worker ignoring FAILED tasks.
                         # Let's filter it out during candidate selection in _worker instead.

        if tasks_to_notify:
            self._condition.notify() # Notify worker if tasks became ready

    def _cleanup_old_tasks_nolock(self, max_completed: int = 1000) -> None:
        """
        Clean up old completed tasks (requires lock to be held externally).

        Args:
            max_completed: Maximum number of completed tasks to keep.
        """
        num_completed = len(self.completed_tasks)
        if num_completed > max_completed:
            # Sort by completion time (oldest first)
            # Note: completed_at might be None if cleanup runs unexpectedly early
            sorted_ids = sorted(
                self.completed_tasks.keys(),
                key=lambda tid: self.completed_tasks[tid].completed_at or 0
            )
            num_to_remove = num_completed - max_completed
            ids_to_remove = sorted_ids[:num_to_remove]
            for task_id in ids_to_remove:
                del self.completed_tasks[task_id]
            logger.debug(f"Cleaned up {num_to_remove} old completed tasks.")


# --- Global Task Queue Instance ---
# Consider if a global instance is appropriate or if it should be managed/injected.
# For simplicity in examples, a global instance is often used.
global_task_queue = TaskQueue()


# --- Convenience Functions for Global Queue ---

async def schedule_task(
    func: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> str:
    """
    Schedule a task for execution using the global task queue.

    Args:
        func: The coroutine function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments, including task config (name, priority, etc.)
                  and function keyword arguments.

    Returns:
        Task ID.
    """
    # Extract task configuration kwargs
    task_config_keys = ["name", "priority", "timeout", "task_id", "tags", "metadata", "dependencies", "submitted_by", "max_retries"]
    task_kwargs = {k: kwargs.pop(k) for k in task_config_keys if k in kwargs}

    # Remaining kwargs are for the function itself
    func_kwargs = kwargs

    # Ensure global queue is running
    if not global_task_queue.running:
        await global_task_queue.start()

    # Add task to queue
    return await global_task_queue.add_task(func, args, func_kwargs, **task_kwargs)


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
        True if task was found and cancellation was attempted/marked.
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
        status: Filter by task status (enum or string).
        tag: Filter by tag.
        submitted_by: Filter by submitter.
        limit: Maximum number of tasks to return.

    Returns:
        List of task dictionaries.
    """
    return await global_task_queue.list_tasks(status, tag, submitted_by, limit)