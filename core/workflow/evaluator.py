"""
Evaluator-Optimizer Workflow Pattern

This module implements the Evaluator-Optimizer pattern for enabling iterative improvement
through feedback loops. This pattern is crucial for self-improving agents that can
critique and optimize their own outputs based on evaluation metrics.

The pattern consists of:
1. Executor: Component that generates initial outputs
2. Evaluator: Component that assesses outputs against quality criteria
3. Optimizer: Component that improves outputs based on evaluations
4. Coordinator: Component that manages the iterative improvement cycle

This pattern enables agents to:
- Generate initial outputs
- Evaluate outputs against defined criteria
- Generate improvement suggestions
- Refine outputs iteratively until quality thresholds are met
"""

import os
import time
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from core.workflow.task import Task, TaskStatus
from core.workflow.workflow import Workflow
from core.exceptions import EvaluationError, OptimizationError

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Evaluation metrics that can be used to assess outputs."""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    CONSISTENCY = "consistency"
    CONCISENESS = "conciseness"
    CORRECTNESS = "correctness"
    HARMLESSNESS = "harmlessness"
    HELPFULNESS = "helpfulness"
    CREATIVITY = "creativity"
    CUSTOM = "custom"


@dataclass
class EvaluationCriterion:
    """
    Defines a criterion for evaluating outputs.
    
    Attributes:
        metric: The evaluation metric to use
        weight: The weight of this criterion in the overall evaluation (0-1)
        threshold: The minimum acceptable score for this criterion (0-1)
        description: Human-readable description of the criterion
        custom_evaluator: Optional function for custom evaluation
    """
    metric: EvaluationMetric
    weight: float = 1.0
    threshold: float = 0.7
    description: str = ""
    custom_evaluator: Optional[Callable[[Any], float]] = None


@dataclass
class EvaluationResult:
    """
    Result of evaluating an output against a set of criteria.
    
    Attributes:
        output_id: Identifier of the evaluated output
        scores: Dictionary mapping criteria to scores (0-1)
        overall_score: Weighted average of all scores
        passed: Whether all criteria passed their thresholds
        feedback: Specific feedback for improvement
        timestamp: When the evaluation was performed
    """
    output_id: str
    scores: Dict[str, float]
    overall_score: float
    passed: bool
    feedback: Dict[str, str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationResult:
    """
    Result of optimizing an output based on evaluation feedback.
    
    Attributes:
        original_output_id: Identifier of the original output
        optimized_output: The improved output
        improvement_summary: Summary of improvements made
        optimization_method: Method used for optimization
        timestamp: When the optimization was performed
    """
    original_output_id: str
    optimized_output: Any
    improvement_summary: str
    optimization_method: str
    timestamp: float = field(default_factory=time.time)


class EvaluatorOptimizer:
    """
    Implements the Evaluator-Optimizer workflow pattern for iterative improvement.
    
    This class coordinates the evaluation and optimization process, managing the
    feedback loop until outputs meet quality criteria or max iterations are reached.
    """
    
    def __init__(
        self,
        evaluator: Optional[Callable] = None,
        optimizer: Optional[Callable] = None,
        executor: Optional[Callable] = None,
        criteria: Optional[List[EvaluationCriterion]] = None,
        max_iterations: int = 3,
        improvement_threshold: float = 0.05,
        metrics_collector: Optional[Callable] = None
    ):
        """
        Initialize the EvaluatorOptimizer.
        
        Args:
            evaluator: Function that evaluates outputs against criteria
            optimizer: Function that improves outputs based on evaluation
            executor: Function that generates initial outputs
            criteria: List of evaluation criteria
            max_iterations: Maximum number of optimization iterations
            improvement_threshold: Minimum improvement required to continue
            metrics_collector: Function to collect and store metrics
        """
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.executor = executor
        self.criteria = criteria or []
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.metrics_collector = metrics_collector
        
        # Initialize metrics tracking
        self.metrics: Dict[str, List[Any]] = {
            "evaluation_results": [],
            "optimization_results": [],
            "improvement_trajectory": []
        }
    
    def set_evaluator(self, evaluator: Callable) -> None:
        """
        Set the evaluator function.
        
        Args:
            evaluator: Function that evaluates outputs against criteria
        """
        self.evaluator = evaluator
    
    def set_optimizer(self, optimizer: Callable) -> None:
        """
        Set the optimizer function.
        
        Args:
            optimizer: Function that improves outputs based on evaluation
        """
        self.optimizer = optimizer
    
    def set_executor(self, executor: Callable) -> None:
        """
        Set the executor function.
        
        Args:
            executor: Function that generates initial outputs
        """
        self.executor = executor
    
    def add_criterion(self, criterion: EvaluationCriterion) -> None:
        """
        Add an evaluation criterion.
        
        Args:
            criterion: The criterion to add
        """
        self.criteria.append(criterion)
    
    def evaluate(self, output: Any, criteria: Optional[List[EvaluationCriterion]] = None) -> EvaluationResult:
        """
        Evaluate an output against criteria.
        
        Args:
            output: The output to evaluate
            criteria: Specific criteria to use (defaults to self.criteria)
            
        Returns:
            Evaluation result
            
        Raises:
            EvaluationError: If evaluation fails
        """
        if not self.evaluator and not criteria:
            raise EvaluationError("No evaluator function or criteria provided")
        
        criteria = criteria or self.criteria
        
        try:
            # If a custom evaluator is provided, use it
            if self.evaluator:
                return self.evaluator(output, criteria)
            
            # Otherwise, perform default evaluation
            return self._default_evaluate(output, criteria)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise EvaluationError(f"Evaluation failed: {str(e)}") from e
    
    def _default_evaluate(self, output: Any, criteria: List[EvaluationCriterion]) -> EvaluationResult:
        """
        Default evaluation implementation.
        
        Args:
            output: The output to evaluate
            criteria: Criteria to evaluate against
            
        Returns:
            Evaluation result
        """
        output_id = getattr(output, "id", str(uuid.uuid4()))
        scores = {}
        feedback = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        passed = True
        
        for criterion in criteria:
            # Determine the score using custom evaluator if provided
            if criterion.custom_evaluator:
                score = criterion.custom_evaluator(output)
            else:
                # Simple placeholder - in a real implementation, this would use
                # more sophisticated evaluation methods based on the metric type
                score = 0.7  # Default placeholder score
            
            # Record score and determine if it passes threshold
            metric_name = criterion.metric.value
            scores[metric_name] = score
            criterion_passed = score >= criterion.threshold
            
            if not criterion_passed:
                passed = False
                feedback[metric_name] = f"Score {score:.2f} below threshold {criterion.threshold:.2f}"
            
            # Update weighted total
            total_weighted_score += score * criterion.weight
            total_weight += criterion.weight
        
        # Calculate overall score
        overall_score = total_weighted_score / max(total_weight, 1e-10)
        
        # Create evaluation result
        result = EvaluationResult(
            output_id=output_id,
            scores=scores,
            overall_score=overall_score,
            passed=passed,
            feedback=feedback
        )
        
        # Record metrics
        self.metrics["evaluation_results"].append(result)
        
        return result
    
    def optimize(self, output: Any, evaluation_result: EvaluationResult) -> OptimizationResult:
        """
        Optimize an output based on evaluation results.
        
        Args:
            output: The output to optimize
            evaluation_result: Evaluation results with feedback
            
        Returns:
            Optimization result
            
        Raises:
            OptimizationError: If optimization fails
        """
        if not self.optimizer:
            raise OptimizationError("No optimizer function provided")
        
        try:
            # Use the provided optimizer
            optimization_result = self.optimizer(output, evaluation_result)
            
            # Record metrics
            self.metrics["optimization_results"].append(optimization_result)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise OptimizationError(f"Optimization failed: {str(e)}") from e
    
    def execute(self, input_data: Any) -> Any:
        """
        Execute the initial task to generate an output.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Generated output
            
        Raises:
            RuntimeError: If execution fails
        """
        if not self.executor:
            raise RuntimeError("No executor function provided")
        
        try:
            return self.executor(input_data)
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Execution failed: {str(e)}") from e
    
    def run_improvement_cycle(self, input_data: Any) -> Tuple[Any, List[EvaluationResult]]:
        """
        Run the full evaluation-optimization cycle to iteratively improve output.
        
        Args:
            input_data: Input data for the initial execution
            
        Returns:
            Tuple of (final_output, evaluation_history)
            
        Raises:
            RuntimeError: If the improvement cycle fails
        """
        if not all([self.executor, self.evaluator, self.optimizer]):
            raise RuntimeError("Executor, evaluator, and optimizer must all be set")
        
        try:
            # Generate initial output
            current_output = self.execute(input_data)
            logger.info(f"Initial output generated")
            
            evaluation_history = []
            last_score = 0.0
            
            # Improvement cycle
            for iteration in range(self.max_iterations):
                # Evaluate current output
                evaluation = self.evaluate(current_output)
                evaluation_history.append(evaluation)
                
                logger.info(f"Iteration {iteration+1}/{self.max_iterations}: " 
                           f"Score {evaluation.overall_score:.4f}")
                
                # Check if all criteria are met
                if evaluation.passed:
                    logger.info(f"All criteria passed. Stopping improvement cycle.")
                    break
                
                # Check if we've reached the max iterations
                if iteration == self.max_iterations - 1:
                    logger.info(f"Reached maximum iterations. Stopping improvement cycle.")
                    break
                
                # Check if improvement is significant enough to continue
                improvement = evaluation.overall_score - last_score
                self.metrics["improvement_trajectory"].append(improvement)
                
                if iteration > 0 and improvement < self.improvement_threshold:
                    logger.info(f"Improvement ({improvement:.4f}) below threshold "
                               f"({self.improvement_threshold:.4f}). Stopping cycle.")
                    break
                
                # Optimize based on evaluation
                optimization = self.optimize(current_output, evaluation)
                current_output = optimization.optimized_output
                last_score = evaluation.overall_score
                
                logger.info(f"Output optimized: {optimization.improvement_summary}")
            
            # Collect final metrics
            if self.metrics_collector:
                self.metrics_collector(self.metrics)
            
            return current_output, evaluation_history
            
        except Exception as e:
            logger.error(f"Improvement cycle failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Improvement cycle failed: {str(e)}") from e


class EvaluatorOptimizerWorkflow:
    """
    Integrates the EvaluatorOptimizer pattern with the task-based workflow system.
    
    This class wraps the EvaluatorOptimizer to provide a workflow-compatible
    interface, allowing it to be used within the larger orchestration framework.
    """
    
    def __init__(
        self,
        name: str,
        evaluator_task: Task,
        optimizer_task: Task,
        executor_task: Task,
        criteria: List[EvaluationCriterion],
        max_iterations: int = 3,
        improvement_threshold: float = 0.05
    ):
        """
        Initialize the EvaluatorOptimizerWorkflow.
        
        Args:
            name: Name of the workflow
            evaluator_task: Task that evaluates outputs
            optimizer_task: Task that optimizes outputs
            executor_task: Task that generates initial outputs
            criteria: List of evaluation criteria
            max_iterations: Maximum number of optimization iterations
            improvement_threshold: Minimum improvement required to continue
        """
        self.name = name
        self.evaluator_task = evaluator_task
        self.optimizer_task = optimizer_task
        self.executor_task = executor_task
        self.criteria = criteria
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        
        # Initialize the workflow
        self.workflow = self._create_workflow()
        
        # Initialize metrics tracking
        self.metrics = {
            "iterations": 0,
            "final_score": 0.0,
            "improvement_trajectory": [],
            "evaluation_history": [],
            "optimization_history": []
        }
    
    def _create_workflow(self) -> Workflow:
        """
        Create the workflow structure.
        
        Returns:
            Configured Workflow instance
        """
        # Create tasks for the workflow
        tasks = [
            self.executor_task,
            self.evaluator_task,
            self.optimizer_task
        ]
        
        # Create the workflow
        workflow = Workflow(
            id=f"evaluator-optimizer-{self.name}",
            tasks=tasks,
            metadata={
                "type": "evaluator-optimizer",
                "max_iterations": self.max_iterations,
                "improvement_threshold": self.improvement_threshold,
                "criteria": [c.metric.value for c in self.criteria]
            }
        )
        
        return workflow
    
    def get_workflow(self) -> Workflow:
        """
        Get the configured workflow.
        
        Returns:
            The workflow instance
        """
        return self.workflow
    
    def process_results(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and organize the results from all tasks in the workflow.
        
        Args:
            task_results: Dictionary mapping task IDs to their results
            
        Returns:
            Processed and structured results
        """
        processed_results = {
            "final_output": None,
            "evaluation_history": self.metrics["evaluation_history"],
            "optimization_history": self.metrics["optimization_history"],
            "iterations": self.metrics["iterations"],
            "final_score": self.metrics["final_score"],
            "improvement_trajectory": self.metrics["improvement_trajectory"]
        }
        
        # Extract the final output from the last optimization or initial execution
        if self.metrics["iterations"] > 0:
            last_optimization = self.metrics["optimization_history"][-1]
            processed_results["final_output"] = last_optimization.get("optimized_output")
        else:
            executor_result = task_results.get(self.executor_task.id, {})
            processed_results["final_output"] = executor_result.get("output")
        
        return processed_results
    
    def update_metrics(self, evaluation_result: Dict[str, Any], optimization_result: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the internal metrics with new evaluation and optimization results.
        
        Args:
            evaluation_result: Results from an evaluation iteration
            optimization_result: Results from an optimization iteration
        """
        # Update evaluation history
        self.metrics["evaluation_history"].append(evaluation_result)
        
        # Update final score
        self.metrics["final_score"] = evaluation_result.get("overall_score", 0.0)
        
        # Update optimization history if available
        if optimization_result:
            self.metrics["optimization_history"].append(optimization_result)
            
            # Increment iteration counter
            self.metrics["iterations"] += 1
            
            # Calculate improvement for this iteration
            if len(self.metrics["evaluation_history"]) >= 2:
                prev_score = self.metrics["evaluation_history"][-2].get("overall_score", 0.0)
                current_score = evaluation_result.get("overall_score", 0.0)
                improvement = current_score - prev_score
                self.metrics["improvement_trajectory"].append(improvement)


# Utility functions for creating common evaluation criteria

def create_accuracy_criterion(weight: float = 1.0, threshold: float = 0.7, custom_evaluator: Optional[Callable] = None) -> EvaluationCriterion:
    """Create an accuracy evaluation criterion."""
    return EvaluationCriterion(
        metric=EvaluationMetric.ACCURACY,
        weight=weight,
        threshold=threshold,
        description="Assesses factual correctness and precision of the output",
        custom_evaluator=custom_evaluator
    )

def create_relevance_criterion(weight: float = 1.0, threshold: float = 0.7, custom_evaluator: Optional[Callable] = None) -> EvaluationCriterion:
    """Create a relevance evaluation criterion."""
    return EvaluationCriterion(
        metric=EvaluationMetric.RELEVANCE,
        weight=weight,
        threshold=threshold,
        description="Evaluates how well the output addresses the specific query or need",
        custom_evaluator=custom_evaluator
    )

def create_completeness_criterion(weight: float = 1.0, threshold: float = 0.7, custom_evaluator: Optional[Callable] = None) -> EvaluationCriterion:
    """Create a completeness evaluation criterion."""
    return EvaluationCriterion(
        metric=EvaluationMetric.COMPLETENESS,
        weight=weight,
        threshold=threshold,
        description="Measures whether the output covers all aspects of the requested information",
        custom_evaluator=custom_evaluator
    )

def create_custom_criterion(name: str, weight: float = 1.0, threshold: float = 0.7, description: str = "", evaluator: Callable = None) -> EvaluationCriterion:
    """Create a custom evaluation criterion."""
    return EvaluationCriterion(
        metric=EvaluationMetric.CUSTOM,
        weight=weight,
        threshold=threshold,
        description=description or f"Custom criterion: {name}",
        custom_evaluator=evaluator
    )