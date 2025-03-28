# ai_agent_framework/core/workflow/metrics_collector.py

"""
Metrics Collection for Evaluator-Optimizer Workflow

This module provides components for collecting, storing, and analyzing metrics
from the Evaluator-Optimizer workflow pattern. These metrics enable tracking
improvement over iterations and assessing the effectiveness of self-improving agents.
"""

import os
import time
import json
import logging
import traceback
import statistics
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

# Assuming EvaluatorOptimizer related dataclasses are imported from evaluator.py
# If they are defined elsewhere, adjust the import path.
# from .evaluator import EvaluationResult, OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class ImprovementMetrics:
    """
    Metrics describing the improvements made during an optimization cycle.

    Attributes:
        workflow_id: Identifier of the workflow
        start_time: When the improvement cycle started
        end_time: When the improvement cycle ended
        iterations: Number of iterations performed
        initial_score: Initial evaluation score
        final_score: Final evaluation score
        overall_improvement: Difference between final and initial scores
        average_improvement_per_iteration: Average improvement per iteration
        time_taken: Total time taken for the improvement cycle
        metrics_by_criterion: Detailed metrics for each criterion
        trajectory: Score trajectory over iterations
        success: Whether the workflow met all criteria
    """
    workflow_id: str
    start_time: float
    end_time: float
    iterations: int
    initial_score: float
    final_score: float
    overall_improvement: float
    average_improvement_per_iteration: float
    time_taken: float
    metrics_by_criterion: Dict[str, Dict[str, float]]
    trajectory: List[float]
    success: bool


class MetricsCollector:
    """
    Collects and stores metrics from Evaluator-Optimizer workflows.

    Provides methods for storing metrics locally, calculating aggregate statistics,
    and analyzing improvement patterns over time.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        store_locally: bool = True,
        # Removed remote telemetry for simplification based on current implementation
        # enable_remote_telemetry: bool = False,
        # remote_endpoint: Optional[str] = None
    ):
        """
        Initialize the MetricsCollector.

        Args:
            storage_path: Path to store metrics data (defaults to './metrics').
            store_locally: Whether to store metrics locally.
        """
        self.store_locally = store_locally
        # self.enable_remote_telemetry = enable_remote_telemetry
        # self.remote_endpoint = remote_endpoint
        self.storage_path = None

        if self.store_locally:
            self.storage_path = Path(storage_path or "./metrics").resolve()
            try:
                self.storage_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create metrics storage directory {self.storage_path}: {e}")
                self.store_locally = False # Disable local storage if creation fails

        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.aggregate_metrics: Dict[str, Any] = {
            "total_workflows": 0,
            "total_iterations": 0,
            "average_iterations_per_workflow": 0.0,
            "average_improvement": 0.0,
            "average_time_per_iteration": 0.0,
            "success_rate": 0.0,
        }
        # Load existing aggregate metrics if available
        if self.store_locally:
            self._load_aggregate_metrics()

    def collect_workflow_metrics(self, workflow_id: str, raw_metrics: Dict[str, Any]) -> None:
        """
        Collect metrics from a completed workflow run.

        Args:
            workflow_id: Identifier of the workflow.
            raw_metrics: Raw metrics collected during workflow execution,
                         expected to contain 'evaluation_results' and 'optimization_results'.
        """
        try:
            processed_metrics = self._process_metrics(workflow_id, raw_metrics)
            if not processed_metrics: # Avoid processing if metrics are invalid
                logger.warning(f"Skipping metrics collection for {workflow_id} due to processing error.")
                return

            self.workflows[workflow_id] = processed_metrics

            self._update_aggregate_metrics()

            if self.store_locally:
                self._store_metrics_locally(workflow_id, processed_metrics)
                self._save_aggregate_metrics() # Save updated aggregates

            # Removed telemetry sending logic

            logger.info(f"Successfully collected metrics for workflow {workflow_id}")

        except Exception as e:
            logger.error(f"Error collecting metrics for workflow {workflow_id}: {e}", exc_info=True)

    def _process_metrics(self, workflow_id: str, raw_metrics: Dict[str, Any]) -> Optional[ImprovementMetrics]:
        """
        Process raw metrics into the standardized ImprovementMetrics dataclass.

        Args:
            workflow_id: Identifier of the workflow.
            raw_metrics: Raw metrics containing 'evaluation_results'.

        Returns:
            An ImprovementMetrics object or None if processing fails.
        """
        # Ensure evaluation_results exist and are not empty
        evaluation_results = raw_metrics.get("evaluation_results", [])
        if not evaluation_results or not isinstance(evaluation_results, list):
            logger.warning(f"Missing or invalid 'evaluation_results' for workflow {workflow_id}.")
            return None

        # Ensure evaluation results contain necessary data (basic check)
        if not all(hasattr(r, 'timestamp') and hasattr(r, 'overall_score') and hasattr(r, 'scores') for r in evaluation_results):
             logger.warning(f"Evaluation results for workflow {workflow_id} are missing required attributes.")
             # Attempt to proceed with available data, but might be incomplete
             # Fallback values
             start_time = end_time = time.time()
             initial_score = final_score = 0.0
             trajectory = [0.0]
        else:
            try:
                start_time = min(r.timestamp for r in evaluation_results)
                end_time = max(r.timestamp for r in evaluation_results)
                initial_score = evaluation_results[0].overall_score
                final_score = evaluation_results[-1].overall_score
                trajectory = [r.overall_score for r in evaluation_results]
            except (AttributeError, IndexError, TypeError) as e:
                 logger.error(f"Error extracting basic stats from evaluation results for {workflow_id}: {e}")
                 return None # Cannot proceed without basic scores/timestamps


        # Optimization results are optional for calculating iterations
        optimization_results = raw_metrics.get("optimization_results", [])
        iterations = len(optimization_results) if isinstance(optimization_results, list) else 0

        overall_improvement = final_score - initial_score
        time_taken = end_time - start_time
        avg_improvement = overall_improvement / max(1, iterations) if iterations > 0 else 0.0

        # Calculate metrics by criterion
        metrics_by_criterion = {}
        try:
            if evaluation_results:
                first_scores = evaluation_results[0].scores
                last_scores = evaluation_results[-1].scores
                if isinstance(first_scores, dict) and isinstance(last_scores, dict):
                     for criterion, initial_crit_score in first_scores.items():
                         final_crit_score = last_scores.get(criterion, initial_crit_score)
                         improvement = final_crit_score - initial_crit_score
                         metrics_by_criterion[criterion] = {
                             "initial_score": initial_crit_score,
                             "final_score": final_crit_score,
                             "improvement": improvement
                         }
        except (AttributeError, IndexError, TypeError) as e:
             logger.warning(f"Could not calculate detailed criterion metrics for {workflow_id}: {e}")


        # Determine success status
        success = getattr(evaluation_results[-1], 'passed', False) if evaluation_results else False

        try:
            return ImprovementMetrics(
                workflow_id=workflow_id,
                start_time=start_time,
                end_time=end_time,
                iterations=iterations,
                initial_score=initial_score,
                final_score=final_score,
                overall_improvement=overall_improvement,
                average_improvement_per_iteration=avg_improvement,
                time_taken=time_taken,
                metrics_by_criterion=metrics_by_criterion,
                trajectory=trajectory,
                success=success
            )
        except Exception as e:
            logger.error(f"Error creating ImprovementMetrics object for {workflow_id}: {e}")
            return None


    def _update_aggregate_metrics(self) -> None:
        """Update aggregate metrics based on all collected workflow metrics."""
        if not self.workflows:
            return

        total_workflows = len(self.workflows)
        total_iterations = sum(wf.get("iterations", 0) for wf in self.workflows.values())
        total_improvement = sum(wf.get("overall_improvement", 0.0) for wf in self.workflows.values())
        total_time = sum(wf.get("time_taken", 0.0) for wf in self.workflows.values())
        successful_workflows = sum(1 for wf in self.workflows.values() if wf.get("success", False))

        self.aggregate_metrics["total_workflows"] = total_workflows
        self.aggregate_metrics["total_iterations"] = total_iterations
        self.aggregate_metrics["average_iterations_per_workflow"] = total_iterations / total_workflows if total_workflows else 0.0
        self.aggregate_metrics["average_improvement"] = total_improvement / total_workflows if total_workflows else 0.0
        self.aggregate_metrics["average_time_per_iteration"] = total_time / max(1, total_iterations) if total_iterations > 0 else 0.0
        self.aggregate_metrics["success_rate"] = (successful_workflows / total_workflows) * 100 if total_workflows else 0.0


    def _get_aggregate_metrics_path(self) -> Path:
        """Returns the path for the aggregate metrics file."""
        return self.storage_path / "aggregate_metrics.json"

    def _load_aggregate_metrics(self) -> None:
        """Load aggregate metrics from the local storage file."""
        if not self.storage_path:
            return
        agg_path = self._get_aggregate_metrics_path()
        if agg_path.exists():
            try:
                with agg_path.open("r") as f:
                    loaded_metrics = json.load(f)
                    # Basic validation before updating
                    if isinstance(loaded_metrics, dict) and "total_workflows" in loaded_metrics:
                        self.aggregate_metrics = loaded_metrics
                        logger.info(f"Loaded aggregate metrics from {agg_path}")
                    else:
                        logger.warning(f"Invalid format in aggregate metrics file: {agg_path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Error loading aggregate metrics from {agg_path}: {e}")


    def _save_aggregate_metrics(self) -> None:
        """Save the current aggregate metrics to local storage."""
        if not self.storage_path:
            return
        agg_path = self._get_aggregate_metrics_path()
        try:
            with agg_path.open("w") as f:
                json.dump(self.aggregate_metrics, f, indent=2)
            logger.debug(f"Saved aggregate metrics to {agg_path}")
        except OSError as e:
            logger.error(f"Error saving aggregate metrics to {agg_path}: {e}")

    def _store_metrics_locally(self, workflow_id: str, metrics: ImprovementMetrics) -> None:
        """
        Store processed workflow metrics locally.

        Args:
            workflow_id: Identifier of the workflow.
            metrics: Processed ImprovementMetrics object to store.
        """
        if not self.storage_path:
            logger.warning("Local storage path not configured, skipping local metrics storage.")
            return

        try:
            workflow_dir = self.storage_path / workflow_id
            workflow_dir.mkdir(exist_ok=True)

            # Store metrics as JSON using dataclass asdict
            metrics_path = workflow_dir / "metrics.json"
            with metrics_path.open("w") as f:
                json.dump(asdict(metrics), f, indent=2)

            # Store trajectory as CSV
            trajectory_path = workflow_dir / "trajectory.csv"
            with trajectory_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Iteration", "Score"])
                # Iteration numbers start from 0 (initial score)
                for i, score in enumerate(metrics.trajectory):
                    writer.writerow([i, score])

            logger.debug(f"Stored metrics locally for workflow {workflow_id} in {workflow_dir}")

        except OSError as e:
            logger.error(f"Error storing metrics locally for workflow {workflow_id}: {e}", exc_info=True)

    # Removed _send_telemetry method

    def get_workflow_metrics(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processed metrics for a specific workflow.

        Args:
            workflow_id: Identifier of the workflow.

        Returns:
            Metrics dictionary for the workflow or None if not found.
        """
        # Return a copy to prevent external modification
        metrics = self.workflows.get(workflow_id)
        return metrics.copy() if metrics else None

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate metrics across all processed workflows.

        Returns:
            Dictionary of aggregate metrics.
        """
        # Return a copy to prevent external modification
        return self.aggregate_metrics.copy()

    def get_improvement_summary(self, workflow_id: str) -> str:
        """
        Generate a human-readable summary of improvement for a workflow.

        Args:
            workflow_id: Identifier of the workflow.

        Returns:
            Summary string describing the improvement process or an error message.
        """
        metrics_data = self.get_workflow_metrics(workflow_id)
        if not metrics_data:
            return f"No metrics found for workflow {workflow_id}"

        try:
            # Use dictionary access for processed metrics
            iterations = metrics_data.get("iterations", 0)
            initial_score = metrics_data.get("initial_score", 0.0)
            final_score = metrics_data.get("final_score", 0.0)
            improvement = metrics_data.get("overall_improvement", 0.0)
            time_taken = metrics_data.get("time_taken", 0.0)
            success = metrics_data.get("success", False)

            # Handle potential division by zero for improvement percentage
            improvement_percent = (improvement / initial_score * 100) if initial_score else 0.0

            summary = [
                f"Workflow {workflow_id} completed {iterations} iteration(s) in {time_taken:.2f} seconds.",
                f"Initial score: {initial_score:.4f}, Final score: {final_score:.4f}",
                f"Overall improvement: {improvement:.4f} ({improvement_percent:.1f}%)"
            ]

            summary.append("All evaluation criteria met." if success else "Not all evaluation criteria met.")

            metrics_by_criterion = metrics_data.get("metrics_by_criterion", {})
            if metrics_by_criterion:
                summary.append("\nImprovement by criterion:")
                for criterion, criterion_metrics in metrics_by_criterion.items():
                    initial = criterion_metrics.get("initial_score", 0.0)
                    final = criterion_metrics.get("final_score", 0.0)
                    criterion_improvement = criterion_metrics.get("improvement", 0.0)
                    # Handle division by zero for criterion improvement percentage
                    crit_improvement_percent = (criterion_improvement / initial * 100) if initial else 0.0

                    summary.append(f"  - {criterion}: {initial:.4f} -> {final:.4f} "
                                  f"(+{criterion_improvement:.4f}, {crit_improvement_percent:.1f}%)")

            return "\n".join(summary)
        except Exception as e:
            logger.error(f"Error generating summary for workflow {workflow_id}: {e}")
            return f"Error generating summary for workflow {workflow_id}: {e}"

    def analyze_improvement_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in improvement across all workflows.

        Returns:
            Dictionary of analysis results.
        """
        if not self.workflows:
            return {"error": "No workflow data available for analysis"}

        valid_workflows = [wf for wf in self.workflows.values() if wf is not None]
        if not valid_workflows:
            return {"error": "No valid workflow data available for analysis"}

        improvements = [wf.get("overall_improvement", 0.0) for wf in valid_workflows]
        iterations_counts = [wf.get("iterations", 0) for wf in valid_workflows]
        times = [wf.get("time_taken", 0.0) for wf in valid_workflows]
        trajectories = [wf.get("trajectory", []) for wf in valid_workflows if wf.get("trajectory")]

        # Calculate basic statistics safely
        avg_improvement = statistics.mean(improvements) if improvements else 0.0
        median_improvement = statistics.median(improvements) if improvements else 0.0
        std_improvement = statistics.stdev(improvements) if len(improvements) > 1 else 0.0

        avg_iterations = statistics.mean(iterations_counts) if iterations_counts else 0.0
        median_iterations = statistics.median(iterations_counts) if iterations_counts else 0.0

        avg_time = statistics.mean(times) if times else 0.0
        avg_time_per_iteration = avg_time / avg_iterations if avg_iterations > 0 else 0.0

        # Analyze improvement trajectory patterns
        diminishing_returns = 0
        linear_improvement = 0
        accelerating_improvement = 0 # Renamed from exponential for clarity
        total_analyzed = 0

        for trajectory in trajectories:
            if len(trajectory) < 3:
                continue
            total_analyzed +=1

            midpoint_idx = len(trajectory) // 2
            # Ensure indices are valid
            if midpoint_idx == 0 or midpoint_idx >= len(trajectory) -1:
                continue

            try:
                first_half_improvement = trajectory[midpoint_idx] - trajectory[0]
                second_half_improvement = trajectory[-1] - trajectory[midpoint_idx]

                # Add tolerance for floating point comparisons
                tolerance = 1e-6
                if second_half_improvement < first_half_improvement * 0.5 - tolerance:
                    diminishing_returns += 1
                elif abs(second_half_improvement - first_half_improvement) < tolerance:
                     linear_improvement += 1 # Consider roughly equal as linear
                elif second_half_improvement > first_half_improvement * 1.5 + tolerance:
                     accelerating_improvement += 1
                else:
                     linear_improvement += 1 # Default to linear if not clearly diminishing/accelerating

            except (TypeError, IndexError) as e:
                 logger.warning(f"Could not analyze trajectory {trajectory}: {e}")
                 continue # Skip trajectory if scores are invalid

        # Calculate percentages safely
        total_analyzed = max(1, total_analyzed) # Avoid division by zero
        diminishing_returns_pct = (diminishing_returns / total_analyzed) * 100
        linear_improvement_pct = (linear_improvement / total_analyzed) * 100
        accelerating_improvement_pct = (accelerating_improvement / total_analyzed) * 100

        # Compile analysis results
        analysis = {
            "total_workflows_analyzed": len(valid_workflows),
            "improvement_statistics": {
                "average": avg_improvement,
                "median": median_improvement,
                "standard_deviation": std_improvement
            },
            "iteration_statistics": {
                "average": avg_iterations,
                "median": median_iterations
            },
            "time_statistics": {
                "average_total_time": avg_time,
                "average_time_per_iteration": avg_time_per_iteration
            },
            "improvement_patterns": {
                "diminishing_returns": {"count": diminishing_returns, "percentage": diminishing_returns_pct},
                "linear_improvement": {"count": linear_improvement, "percentage": linear_improvement_pct},
                "accelerating_improvement": {"count": accelerating_improvement, "percentage": accelerating_improvement_pct}
            }
        }
        return analysis

    def export_all_metrics(self, export_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Export all collected metrics to a JSON file.

        Args:
            export_path: Path to export metrics to (defaults to timestamped file).

        Returns:
            Path object to the exported file or None if export failed.
        """
        if not self.store_locally or not self.storage_path:
            logger.error("Local storage is not enabled or path not set, cannot export metrics.")
            return None

        export_path = Path(export_path) if export_path else \
                      self.storage_path / f"all_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "aggregate_metrics": self.aggregate_metrics,
            "workflows": self.workflows, # workflows already contains processed data
            "analysis": self.analyze_improvement_patterns()
        }

        try:
            export_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with export_path.open("w") as f:
                # Use default=str for any non-serializable objects (like timestamps if not float)
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Successfully exported all metrics to {export_path}")
            return export_path

        except (OSError, TypeError) as e:
            logger.error(f"Error exporting metrics to {export_path}: {e}", exc_info=True)
            return None

    def plot_improvement_trajectory(self, workflow_id: str, output_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Plot the improvement trajectory for a workflow using Matplotlib.

        Args:
            workflow_id: Identifier of the workflow.
            output_path: Path to save the plot (defaults to workflow directory).

        Returns:
            Path object to the saved plot or None if plotting failed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib is required for plotting. Install with: pip install matplotlib")
            return None

        metrics = self.get_workflow_metrics(workflow_id)
        if not metrics or not metrics.get("trajectory"):
            logger.warning(f"No trajectory data found for workflow {workflow_id}")
            return None

        trajectory = metrics["trajectory"]
        if not isinstance(trajectory, list) or not trajectory:
             logger.warning(f"Invalid or empty trajectory data for workflow {workflow_id}")
             return None

        iterations = range(len(trajectory)) # Iteration 0 is initial score

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, trajectory, marker='o', linestyle='-', linewidth=2, label="Overall Score")
            plt.title(f"Improvement Trajectory for Workflow {workflow_id}")
            plt.xlabel("Evaluation Iteration (0 = Initial)")
            plt.ylabel("Overall Evaluation Score")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(iterations) # Ensure integer ticks for iterations
            plt.ylim(min(0, min(trajectory) - 0.1), max(1.05, max(trajectory) + 0.1)) # Dynamic Y limits
            plt.legend()

            # Annotate start and end points
            plt.annotate(f"Start: {trajectory[0]:.3f}", xy=(0, trajectory[0]),
                         xytext=(5, -15), textcoords="offset points")
            if len(trajectory) > 1:
                 plt.annotate(f"End: {trajectory[-1]:.3f}", xy=(len(trajectory)-1, trajectory[-1]),
                              xytext=(-10, 10), textcoords="offset points")

            # Determine output path
            if output_path:
                output_path = Path(output_path)
            elif self.storage_path:
                 output_path = self.storage_path / workflow_id / "trajectory_plot.png"
            else:
                 logger.warning("No output path specified and local storage disabled, cannot save plot.")
                 plt.close() # Close plot figure
                 return None

            output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close() # Close plot figure to free memory

            logger.info(f"Saved trajectory plot for workflow {workflow_id} to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error during plotting for workflow {workflow_id}: {e}", exc_info=True)
            plt.close() # Ensure plot is closed on error
            return None