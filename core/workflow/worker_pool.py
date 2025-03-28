# ai_agent_framework/core/workflow/worker_pool.py

"""
Worker Pool Management

Defines Worker status, Worker representation, and the WorkerPool class
for managing and selecting workers for task execution in an async environment.
"""

import time
import asyncio
import logging
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class WorkerStatus(Enum):
    """Possible statuses for a worker."""
    OFFLINE = "offline"     # Worker is not reachable or unregistered
    ONLINE = "online"       # Worker is registered and available
    BUSY = "busy"         # Worker is online but at max concurrent task capacity
    ERROR = "error"         # Worker reported an error or is unresponsive

@dataclass
class Worker:
    """Represents a worker node capable of executing tasks."""
    id: str
    endpoint: str # URL for communication
    capabilities: List[str] = field(default_factory=list)
    status: WorkerStatus = WorkerStatus.ONLINE
    max_concurrent_tasks: int = 1
    # Use a set to track unique IDs of currently assigned tasks
    active_tasks: Set[str] = field(default_factory=set)
    last_heartbeat: float = field(default_factory=time.monotonic)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_load(self) -> int:
        """Number of tasks currently assigned."""
        return len(self.active_tasks)

    def is_available(self) -> bool:
        """Check if the worker is online and has capacity."""
        return self.status == WorkerStatus.ONLINE and self.current_load < self.max_concurrent_tasks

    def update_status(self):
        """Update status based on current load."""
        if self.status == WorkerStatus.ONLINE and self.current_load >= self.max_concurrent_tasks:
             self.status = WorkerStatus.BUSY
        elif self.status == WorkerStatus.BUSY and self.current_load < self.max_concurrent_tasks:
             self.status = WorkerStatus.ONLINE
        # Note: ERROR/OFFLINE statuses usually need explicit external updates


class WorkerPool:
    """
    Manages a collection of Worker instances for the Orchestrator.

    Provides methods for adding, removing, tracking status, and selecting
    appropriate workers based on different strategies in an async environment.
    """
    def __init__(self):
        self.workers: Dict[str, Worker] = {}
        self._lock = asyncio.Lock() # Use asyncio.Lock for managing access to workers dict
        self._round_robin_index = 0

    async def add_worker(self, worker: Worker):
        """Add or update a worker in the pool."""
        async with self._lock:
            if worker.id in self.workers:
                 logger.warning(f"Updating existing worker: {worker.id}")
                 # Preserve active tasks if updating
                 worker.active_tasks = self.workers[worker.id].active_tasks
            else:
                 logger.info(f"Adding new worker: {worker.id} with caps {worker.capabilities}")
            self.workers[worker.id] = worker
            worker.update_status() # Ensure initial status is correct

    async def remove_worker(self, worker_id: str):
        """Remove a worker from the pool."""
        async with self._lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Removed worker: {worker_id}")
                # Adjust round-robin index if necessary
                if self._round_robin_index >= len(self.workers):
                    self._round_robin_index = 0
            else:
                # Raising error as per original Orchestrator expectation
                raise ValueError(f"Worker '{worker_id}' not found in pool.")

    async def get_worker(self, worker_id: str) -> Optional[Worker]:
        """Get a worker instance by ID."""
        async with self._lock:
            return self.workers.get(worker_id)

    async def update_worker_status(self, worker_id: str, status: WorkerStatus, error: Optional[str] = None):
        """Explicitly update the status of a worker."""
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker:
                 if worker.status != status:
                      logger.info(f"Updating worker '{worker_id}' status to {status.name}")
                      worker.status = status
                 if status == WorkerStatus.ERROR and error:
                      worker.metadata["last_error"] = error
                 worker.last_heartbeat = time.monotonic() # Update heartbeat on status change
                 worker.update_status() # Re-evaluate BUSY/ONLINE based on load
            else:
                 logger.warning(f"Cannot update status for unknown worker: {worker_id}")

    async def assign_task_to_worker(self, worker_id: str, task_id: str):
        """Assign a task to a worker and update load."""
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker:
                 if not worker.is_available():
                      logger.warning(f"Assigning task {task_id} to worker {worker_id} that is not available (Status: {worker.status.name}, Load: {worker.current_load}/{worker.max_concurrent_tasks}).")
                 worker.active_tasks.add(task_id)
                 worker.last_heartbeat = time.monotonic()
                 worker.update_status() # Check if status becomes BUSY
                 logger.debug(f"Assigned task {task_id} to worker {worker_id}. New load: {worker.current_load}")
            else:
                 logger.error(f"Cannot assign task {task_id}, worker {worker_id} not found.")

    async def release_task_from_worker(self, worker_id: str, task_id: str):
        """Release a completed/failed task from a worker and update load."""
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker:
                 if task_id in worker.active_tasks:
                      worker.active_tasks.discard(task_id)
                      worker.last_heartbeat = time.monotonic()
                      worker.update_status() # Check if status becomes ONLINE
                      logger.debug(f"Released task {task_id} from worker {worker_id}. New load: {worker.current_load}")
                 else:
                      logger.warning(f"Task {task_id} not found in active tasks of worker {worker_id}. Cannot release.")
            # else: No warning if worker disappeared, might have been unregistered

    async def get_available_workers(self, required_capabilities: Optional[List[str]] = None) -> List[Worker]:
        """Get a list of workers that are ONLINE and have capacity, matching capabilities if specified."""
        async with self._lock:
             available = []
             for worker in self.workers.values():
                  if worker.is_available(): # Checks ONLINE and capacity
                       if required_capabilities:
                           # Check if worker has all required capabilities
                           if all(cap in worker.capabilities for cap in required_capabilities):
                                available.append(worker)
                       else:
                           # No capabilities required, just check availability
                           available.append(worker)
             return available

    # --- Worker Selection Strategies ---

    async def get_next_available_worker(self, required_capabilities: Optional[List[str]] = None) -> Optional[Worker]:
        """Selects the next available worker using round-robin."""
        async with self._lock: # Lock needed to safely access workers and _round_robin_index
            available = await self.get_available_workers(required_capabilities) # Get currently available matching workers
            if not available:
                return None

            num_available = len(available)
            # Ensure index is within bounds of the *currently available* list
            start_index = self._round_robin_index % num_available
            for i in range(num_available):
                 # Cycle through available list starting from current index
                 current_check_index = (start_index + i) % num_available
                 selected_worker = available[current_check_index]

                 # Update global index for next call relative to the full worker list size if possible,
                 # otherwise just cycle through available ones. This is imperfect round robin.
                 # A better round robin needs stable indexing or worker list.
                 self._round_robin_index = (self._round_robin_index + 1) % max(1, len(self.workers))

                 return selected_worker

            return None # Should not happen if available is not empty

    async def get_least_loaded_worker(self, required_capabilities: Optional[List[str]] = None) -> Optional[Worker]:
        """Selects the available worker with the minimum current load."""
        async with self._lock:
            available = await self.get_available_workers(required_capabilities)
            if not available:
                return None

            # Find worker with the minimum load
            return min(available, key=lambda w: w.current_load)

    async def get_best_capability_match_worker(self, required_capabilities: Optional[List[str]] = None) -> Optional[Worker]:
        """Selects the best available worker based on capability match (most specialized) and then load."""
        required_caps = set(required_capabilities or [])
        async with self._lock:
            available = await self.get_available_workers(required_capabilities) # Already filters by required caps
            if not available:
                return None

            candidates: List[Tuple[int, int, Worker]] = [] # (extra_caps, load, worker)
            for worker in available:
                 worker_caps = set(worker.capabilities)
                 extra_caps = len(worker_caps - required_caps)
                 candidates.append((extra_caps, worker.current_load, worker))

            # Sort by: 1. Fewest extra capabilities, 2. Lowest current load
            candidates.sort(key=lambda x: (x[0], x[1]))

            return candidates[0][2] # Return the worker from the best candidate