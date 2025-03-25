"""
Base Workflow Class

This module defines the abstract base class for all workflow patterns.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BaseWorkflow(ABC):
    """
    Abstract base class for all workflow patterns.
    
    This class defines the interface and common functionality for workflow implementations.
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        max_steps: int = 10,
        verbose: bool = False
    ):
        """
        Initialize the BaseWorkflow.
        
        Args:
            name: A unique name for this workflow
            description: Optional description of what this workflow does
            max_steps: Maximum number of steps the workflow can execute
            verbose: Whether to log detailed workflow execution information
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description or f"Workflow: {name}"
        self.max_steps = max_steps
        self.verbose = verbose
        
        # Execution state
        self.current_step = 0
        self.steps_executed = []
        self.execution_log = []
        self.finished = False
        self.success = False
        self.error = None
    
    @abstractmethod
    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute the workflow with the given input.
        
        This method must be implemented by all concrete workflow classes.
        
        Args:
            input_data: The input data for the workflow
            **kwargs: Additional execution parameters
            
        Returns:
            A dictionary containing the workflow result and execution metadata
        """
        pass
    
    def _log_step(self, step_name: str, input_data: Any, output_data: Any, error: Optional[Exception] = None) -> None:
        """
        Log a step execution to the workflow's execution log.
        
        Args:
            step_name: Name of the step
            input_data: Input data for the step
            output_data: Output data from the step
            error: Optional error that occurred during execution
        """
        step_log = {
            "step_number": self.current_step,
            "step_name": step_name,
            "timestamp": import_time_and_return_time(),
            "success": error is None,
            "error": str(error) if error else None
        }
        
        # Log detailed input/output if verbose
        if self.verbose:
            step_log["input"] = input_data
            step_log["output"] = output_data
        
        self.execution_log.append(step_log)
        self.steps_executed.append(step_name)
        
        if self.verbose:
            if error:
                logger.error(f"Step {self.current_step} ({step_name}) failed: {str(error)}")
            else:
                logger.info(f"Step {self.current_step} ({step_name}) completed successfully")
    
    def _increment_step(self) -> bool:
        """
        Increment the step counter and check if max steps has been reached.
        
        Returns:
            True if execution can continue, False if max steps reached
        """
        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            logger.warning(f"Workflow '{self.name}' reached maximum steps ({self.max_steps})")
            self._mark_finished(success=False, error="Maximum step count reached")
            return False
        
        return True
    
    def _mark_finished(self, success: bool = True, error: Optional[str] = None) -> None:
        """
        Mark the workflow as finished.
        
        Args:
            success: Whether the workflow completed successfully
            error: Optional error message
        """
        self.finished = True
        self.success = success
        self.error = error
        
        if self.verbose:
            if success:
                logger.info(f"Workflow '{self.name}' completed successfully")
            else:
                logger.error(f"Workflow '{self.name}' failed: {error}")
    
    def reset(self) -> None:
        """Reset the workflow's execution state."""
        self.current_step = 0
        self.steps_executed = []
        self.execution_log = []
        self.finished = False
        self.success = False
        self.error = None
        
        if self.verbose:
            logger.info(f"Workflow '{self.name}' reset")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the workflow's execution.
        
        Returns:
            Dictionary containing execution summary
        """
        return {
            "workflow_id": self.id,
            "workflow_name": self.name,
            "steps_executed": self.steps_executed,
            "total_steps": self.current_step,
            "finished": self.finished,
            "success": self.success,
            "error": self.error
        }


def import_time_and_return_time():
    """Helper function to handle time import and return current time."""
    import time
    return time.time()