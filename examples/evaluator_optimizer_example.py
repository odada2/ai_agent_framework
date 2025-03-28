# ai_agent_framework/examples/evaluator_optimizer_example.py

"""
Evaluator-Optimizer Pattern Example (Async Updates)

Demonstrates the Evaluator-Optimizer workflow pattern with corrected imports
and adapted for asynchronous execution context.
"""

import os
import sys
import time
import json
import logging
import random
import argparse
import asyncio # Added asyncio
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import asdict, dataclass

# Add parent directory to path to allow imports (assuming structure)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Framework components (using absolute imports)
try:
    from ai_agent_framework.core.workflow.evaluator import (
        EvaluatorOptimizer,
        EvaluationCriterion,
        EvaluationMetric,
        EvaluationResult,
        OptimizationResult,
        create_accuracy_criterion,
        create_relevance_criterion,
        create_completeness_criterion
    )
    from ai_agent_framework.core.workflow.metrics_collector import MetricsCollector
    # Assuming BaseExampleConfig exists or defining it here
    # from ai_agent_framework.examples.base_example import BaseExampleConfig # If exists
except ImportError as e:
    print(f"Error importing framework components: {e}. Please ensure paths are correct.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
METRICS_BASE_PATH = "./metrics" # Relative path, ensure it exists or is created
MAX_ITERATIONS_DEFAULT = 5
IMPROVEMENT_THRESHOLD_DEFAULT = 0.05

# --- Utility Functions (Synchronous Simulations - OK for Example) ---
# (Keep the synchronous evaluate_coherence_basic, evaluate_conciseness_basic as before)
def evaluate_coherence_basic(content: str, version: int) -> float:
    """Evaluate coherence using basic heuristics (shared logic)."""
    transition_words = ["however", "therefore", "thus", "consequently", "furthermore",
                      "moreover", "nevertheless", "in addition", "for example",
                      "specifically", "because", "since", "although", "despite"]
    transition_count = sum(1 for word in transition_words if word in content.lower())
    transition_score = min(0.4, transition_count * 0.06)
    sentences = [s.strip() for s in content.split(".") if s.strip()]
    variance_score = 0
    if sentences and len(sentences) > 1: # Avoid division by zero or variance of single item
        sentence_lengths = [len(s) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentences)
        length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentences)
        variance_score = 0.2 * max(0, 1 - (length_variance / 1000))
    base_score = 0.4 + min(0.2, (version - 1) * 0.08)
    score = base_score + transition_score + variance_score
    return max(0.0, min(1.0, score))

def evaluate_conciseness_basic(content: str, ideal_length: int) -> float:
    """Evaluate conciseness using basic heuristics (shared logic)."""
    content_len = len(content)
    if content_len == 0: return 0.0
    if ideal_length <= 0: return 0.0

    length_score = 0.5
    if content_len < ideal_length * 0.5: length_score = 0.5 * (content_len / (ideal_length * 0.5))
    elif content_len > ideal_length * 1.2: length_score = 0.5 * max(0, 1 - (content_len - ideal_length * 1.2) / (ideal_length * 2))

    filler_words = ["very", "really", "quite", "basically", "actually", "simply", "just", "indeed", "certainly", "definitely"]
    words = content.lower().split()
    word_count = len(words)
    filler_score = 0.3
    if word_count > 0:
        filler_count = sum(1 for word in filler_words if word in words)
        filler_ratio = filler_count / word_count
        filler_score = 0.3 * max(0, 1 - filler_ratio * 10)

    sentences = [s.strip() for s in content.split(".") if s.strip()]
    sentence_score = 0.2
    if sentences:
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
        sentence_score = 0.2 * max(0, 1 - (avg_sentence_length / 150)) # Adjusted penalty threshold

    score = length_score + filler_score + sentence_score
    return max(0.0, min(1.0, score))


# --- Base Example Class (Async Updates) ---

# Assuming BaseExampleConfig dataclass exists or define simply
@dataclass
class BaseExampleConfig:
    """Configuration for base example."""
    metrics_path: str
    criteria: List[EvaluationCriterion]
    max_iterations: int = MAX_ITERATIONS_DEFAULT
    improvement_threshold: float = IMPROVEMENT_THRESHOLD_DEFAULT

class BaseEvaluatorOptimizerExample:
    """Base class for Evaluator-Optimizer examples (adapted for async context)."""

    def __init__(self, config: BaseExampleConfig, executor: Callable, evaluator: Callable, optimizer: Callable):
        self.config = config
        # Ensure metrics directory exists if storing locally
        if METRICS_BASE_PATH:
             os.makedirs(config.metrics_path, exist_ok=True)
        self.metrics_collector = MetricsCollector(storage_path=config.metrics_path)

        # Note: EvaluatorOptimizer itself might need to be async if executor/evaluator/optimizer are async
        # Assuming EvaluatorOptimizer can work with sync functions for this simulation
        self.evaluator_optimizer = EvaluatorOptimizer(
            evaluator=evaluator, # Can be sync or async
            optimizer=optimizer, # Can be sync or async
            executor=executor,   # Can be sync or async
            criteria=config.criteria,
            max_iterations=config.max_iterations,
            improvement_threshold=config.improvement_threshold,
            metrics_collector=self.metrics_collector.collect_workflow_metrics # Pass the method
        )
        self._executor = executor # Store executor for direct call if needed

    def _base_evaluate(self, output_obj: Dict[str, Any], criteria: List[EvaluationCriterion]) -> EvaluationResult:
        """Generic sync evaluation loop (adapt if evaluators become async)."""
        output_id = output_obj.get("id", f"output_{uuid.uuid4()}")
        version = output_obj.get("version", 1)
        logger.info(f"Evaluating output '{output_id}' (version {version})")

        scores = {}
        feedback = {}
        passed = True

        for criterion in criteria:
            eval_method_name = f"_evaluate_{criterion.metric.value.lower()}"
            eval_method = getattr(self, eval_method_name, None)
            score = 0.5 # Default score

            try:
                if eval_method and callable(eval_method):
                     # Assuming eval methods are synchronous for simulation
                     score = eval_method(output_obj)
                elif criterion.custom_evaluator and callable(criterion.custom_evaluator):
                     # Assuming custom evaluator is synchronous
                     score = criterion.custom_evaluator(output_obj)
                else:
                     logger.warning(f"No specific evaluator found for metric {criterion.metric.value}. Using default score {score}.")
                     feedback[criterion.metric.value] = "Evaluation logic missing or incomplete."

                # Validate score is a float between 0 and 1
                if not isinstance(score, (float, int)) or not (0.0 <= score <= 1.0):
                    logger.error(f"Invalid score {score} returned for metric {criterion.metric.value}. Clamping to 0.5.")
                    score = 0.5
                    feedback[criterion.metric.value] = "Invalid score returned by evaluator."

            except Exception as e:
                 logger.error(f"Error during evaluation for metric {criterion.metric.value}: {e}", exc_info=True)
                 score = 0.0 # Assign low score on error
                 feedback[criterion.metric.value] = f"Evaluation error: {e}"


            scores[criterion.metric.value] = score
            criterion_passed = score >= criterion.threshold
            if not criterion_passed:
                passed = False
                if criterion.metric.value not in feedback:
                     feedback[criterion.metric.value] = f"Score {score:.2f} below threshold {criterion.threshold:.2f}."

        # Calculate overall score
        total_weight = sum(c.weight for c in criteria if c.metric.value in scores)
        overall_score = 0.0
        if total_weight > 0:
             weighted_sum = sum(scores[c.metric.value] * c.weight for c in criteria if c.metric.value in scores)
             overall_score = weighted_sum / total_weight

        return EvaluationResult(
            output_id=output_id, scores=scores, overall_score=overall_score,
            passed=passed, feedback=feedback, timestamp=time.time() # Add timestamp
        )

    # Note: Making run async as it involves the potentially async EvaluatorOptimizer loop
    async def run(self, initial_input: Any) -> Dict[str, Any]:
        """Runs the improvement cycle asynchronously."""
        logger.info(f"Running {self.__class__.__name__}...")

        # Assume run_improvement_cycle might be async if its components are async
        # If EvaluatorOptimizer needs async, change its methods and use await here.
        # For now, assuming sync simulation functions allow sync run_improvement_cycle
        # But we run the executor simulation within run_in_executor if it's not async
        executor_func = self.evaluator_optimizer.executor
        if asyncio.iscoroutinefunction(executor_func):
             initial_output = await executor_func(initial_input)
        else:
             # Run sync executor in thread pool to avoid blocking event loop
             loop = asyncio.get_running_loop()
             initial_output = await loop.run_in_executor(None, executor_func, initial_input)

        # Assuming run_improvement_cycle remains sync for this simulation example:
        # final_output, evaluation_history = self.evaluator_optimizer.run_improvement_cycle(initial_input)
        # If it becomes async:
        final_output, evaluation_history = await self.evaluator_optimizer.run_improvement_cycle(initial_input)


        workflow_id = initial_output.get("id", "unknown") # Use ID from initial output

        # Gather results
        # Note: Metrics collector methods might need to be async if they do I/O
        metrics = self.metrics_collector.get_workflow_metrics(workflow_id) # Assuming sync get
        summary = self.metrics_collector.get_improvement_summary(workflow_id) # Assuming sync get

        result = {
            "initial_input": initial_input,
            "initial_output": initial_output, # Capture the actual first output
            "final_output": final_output,
            "iterations": len(evaluation_history),
            # Convert evaluation results to dicts for JSON serialization if needed
            "evaluation_history": [asdict(e) if hasattr(e, '__dataclass_fields__') else e for e in evaluation_history],
            "improvement_summary": summary,
            "metrics": metrics
        }

        logger.info(f"{self.__class__.__name__} complete. Final score: {evaluation_history[-1].overall_score:.4f} after {len(evaluation_history)-1} iterations.")
        return result


# --- Example Implementations (Mostly Synchronous Simulations) ---

class TextRefinementExample(BaseEvaluatorOptimizerExample):
    """Example 1: Text refinement (Simulated, Sync)."""
    def __init__(self):
        criteria = [
            create_accuracy_criterion(weight=1.0, threshold=0.8),
            create_relevance_criterion(weight=0.8, threshold=0.7),
            create_completeness_criterion(weight=0.6, threshold=0.75),
            EvaluationCriterion(metric=EvaluationMetric.COHERENCE, weight=0.7, threshold=0.7),
            EvaluationCriterion(metric=EvaluationMetric.CONCISENESS, weight=0.5, threshold=0.6)
        ]
        config = BaseExampleConfig(metrics_path=os.path.join(METRICS_BASE_PATH, "text_refinement"), criteria=criteria)
        # Provide sync simulation functions
        super().__init__(config=config, executor=self._generate_initial_text,
                         evaluator=self._evaluate_text, optimizer=self._optimize_text)

    # --- Sync Simulation Methods ---
    def _generate_initial_text(self, topic: str) -> Dict[str, Any]:
        logger.info(f"Generating initial text for topic: {topic}"); time.sleep(0.1) # Simulate work
        texts = {"climate change": "Climate change is bad. Earth hot. Ice melts.",
                 "ai": "AI is computers thinking. It learns.",
                 "space": "Space is big. Rockets go up."}
        text = texts.get(topic.lower(), f"Text about {topic}.")
        return {"id": f"text_{topic.replace(' ', '_')}_{int(time.time())}", "topic": topic, "content": text, "version": 1}

    def _evaluate_text(self, text_obj: Dict[str, Any], criteria: List[EvaluationCriterion]) -> EvaluationResult:
        return self._base_evaluate(text_obj, criteria) # Use base sync evaluator

    def _optimize_text(self, text_obj: Dict[str, Any], evaluation: EvaluationResult) -> OptimizationResult:
        logger.info(f"Optimizing text version {text_obj['version']}"); time.sleep(0.15) # Simulate work
        current_text = text_obj["content"]; version = text_obj["version"]
        improved_text = current_text + f" (Improved v{version+1}: based on feedback - {list(evaluation.feedback.keys())})"
        improved_obj = {**text_obj, "content": improved_text, "version": version + 1}
        return OptimizationResult(original_output_id=text_obj["id"], optimized_output=improved_obj,
                                  improvement_summary="Applied feedback", optimization_method="simulated")

    # --- Specific Sync Evaluation Simulations ---
    def _evaluate_accuracy(self, obj: Dict) -> float: return min(1.0, 0.5 + obj["version"] * 0.1 + len(obj["content"]) / 2000)
    def _evaluate_relevance(self, obj: Dict) -> float: return min(1.0, 0.6 + obj["content"].lower().count(obj["topic"].lower()) * 0.1)
    def _evaluate_completeness(self, obj: Dict) -> float: return min(1.0, 0.4 + obj["version"] * 0.08 + len(obj["content"]) / 1500)
    def _evaluate_coherence(self, obj: Dict) -> float: return evaluate_coherence_basic(obj["content"], obj["version"])
    def _evaluate_conciseness(self, obj: Dict) -> float: return evaluate_conciseness_basic(obj["content"], 500)


class QueryOptimizationExample(BaseEvaluatorOptimizerExample):
    """Example 2: Query optimization (Simulated, Sync)."""
    # (Keep KNOWLEDGE_BASE as before)
    KNOWLEDGE_BASE = { "customer": [{"id": 1, "name": "Alice", "segment": "premium"}, {"id": 2, "name": "Bob", "segment": "standard"}], # ... etc
                    }
    def __init__(self):
         criteria = [ create_accuracy_criterion(weight=1.0, threshold=0.8),
                      create_relevance_criterion(weight=0.9, threshold=0.75),
                      create_completeness_criterion(weight=0.7, threshold=0.7) ]
         config = BaseExampleConfig(metrics_path=os.path.join(METRICS_BASE_PATH, "query_optimization"), criteria=criteria, max_iterations=4)
         super().__init__(config=config, executor=self._generate_initial_query,
                          evaluator=self._evaluate_query, optimizer=self._optimize_query)
    # --- Sync Simulation Methods ---
    def _generate_initial_query(self, req: Dict) -> Dict:
         logger.info(f"Generating query for: {req.get('question')}"); time.sleep(0.05)
         q = req.get("question", "").lower(); query = "SELECT * FROM customer"
         if "product" in q: query = "SELECT * FROM product"
         elif "sales" in q: query = "SELECT * FROM sales"
         return {"id": f"query_{int(time.time())}", "question": req.get("question"), "query": query, "version": 1, "results": []}

    def _execute_query_simulation(self, query: str) -> List[Dict]:
        # (Keep synchronous query simulation logic as before)
        logger.debug(f"Simulating execution: {query}")
        try:
             parts = query.lower().split(); table = "customer"
             if "from" in parts: table = parts[parts.index("from") + 1].split()[0]
             data = self.KNOWLEDGE_BASE.get(table, [])
             if "where" in parts and "=" in parts: # Very basic WHERE
                  w_idx = parts.index("where"); field = parts[w_idx+1]; value = parts[w_idx+3].strip("'\"")
                  data = [r for r in data if str(r.get(field,"")).lower() == value]
             if "select" in parts and "*" not in parts: # Basic SELECT fields
                  s_idx = parts.index("select"); f_idx = parts.index("from")
                  fields = [f.strip() for f in " ".join(parts[s_idx+1:f_idx]).split(",")]
                  data = [{f: r.get(f) for f in fields} for r in data]
             return data if data else [{"info": "No matching rows found"}]
        except Exception as e: return [{"error": str(e)}]

    def _evaluate_query(self, query_obj: Dict, criteria: List[EvaluationCriterion]) -> EvaluationResult:
         if not query_obj.get("results"): query_obj["results"] = self._execute_query_simulation(query_obj["query"])
         return self._base_evaluate(query_obj, criteria)

    def _optimize_query(self, query_obj: Dict, evaluation: EvaluationResult) -> OptimizationResult:
        logger.info(f"Optimizing query version {query_obj['version']}"); time.sleep(0.1)
        current_query = query_obj["query"]; version = query_obj["version"]
        improved_query = current_query
        improvements = []
        # Simple optimization: Add WHERE clause if accuracy/relevance low
        if ("accuracy" in evaluation.feedback or "relevance" in evaluation.feedback) and "where" not in current_query.lower():
             if "premium" in query_obj["question"].lower(): improved_query += " WHERE segment = 'premium'"; improvements.append("Added WHERE")
        if not improvements: improvements.append("Minor tweaks")
        improved_obj = {**query_obj, "query": improved_query, "version": version + 1, "results": []}
        return OptimizationResult(original_output_id=query_obj["id"], optimized_output=improved_obj,
                                  improvement_summary=", ".join(improvements), optimization_method="simulated")

    # --- Specific Sync Evaluation Simulations ---
    def _evaluate_accuracy(self, obj: Dict) -> float:
        # Basic check if results seem plausible for question
        q = obj["question"].lower(); res = obj["results"]
        if not res or "error" in res[0]: return 0.1
        score = 0.6
        if "premium" in q and any(r.get("segment")=="premium" for r in res): score = 0.9
        if "alice" in q and any(r.get("name")=="Alice" for r in res): score = 0.95
        return min(1.0, score + obj["version"]*0.05)

    def _evaluate_relevance(self, obj: Dict) -> float: return min(1.0, 0.7 + obj["version"]*0.1 - (0.1 if "select *" in obj["query"].lower() else 0))
    def _evaluate_completeness(self, obj: Dict) -> float: return min(1.0, 0.5 + len(obj["results"])*0.1)


class CreativeContentExample(BaseEvaluatorOptimizerExample):
     """Example 3: Creative content (Simulated, Sync)."""
     def __init__(self):
         criteria = [ EvaluationCriterion(metric=EvaluationMetric.COHERENCE, weight=0.8, threshold=0.7),
                      EvaluationCriterion(metric=EvaluationMetric.CREATIVITY, weight=1.0, threshold=0.75),
                      EvaluationCriterion(metric=EvaluationMetric.HELPFULNESS, weight=0.7, threshold=0.7) ] # Helpfulness as topic adherence proxy
         config = BaseExampleConfig(metrics_path=os.path.join(METRICS_BASE_PATH, "creative_content"), criteria=criteria)
         super().__init__(config=config, executor=self._generate_initial_content,
                          evaluator=self._evaluate_content, optimizer=self._optimize_content)

     # --- Sync Simulation Methods ---
     def _generate_initial_content(self, req: Dict) -> Dict:
         logger.info(f"Generating content: {req}"); time.sleep(0.1)
         topic=req.get("topic", "cats"); style=req.get("style","plain"); type=req.get("type","story")
         content = f"A {style} {type} about {topic}. Once upon a time."
         return {"id": f"creative_{int(time.time())}", "topic":topic, "style":style, "type":type, "content":content, "version": 1}

     def _evaluate_content(self, content_obj: Dict, criteria: List[EvaluationCriterion]) -> EvaluationResult:
         return self._base_evaluate(content_obj, criteria)

     def _optimize_content(self, content_obj: Dict, evaluation: EvaluationResult) -> OptimizationResult:
        logger.info(f"Optimizing content version {content_obj['version']}"); time.sleep(0.15)
        current_content = content_obj["content"]; version = content_obj["version"]
        improved_content = current_content + f" Then something creative happened (v{version+1}). Feedback keys: {list(evaluation.feedback.keys())}."
        improved_obj = {**content_obj, "content": improved_content, "version": version + 1}
        return OptimizationResult(original_output_id=content_obj["id"], optimized_output=improved_obj,
                                  improvement_summary="Added creativity", optimization_method="simulated")

     # --- Specific Sync Evaluation Simulations ---
     def _evaluate_coherence(self, obj: Dict) -> float: return evaluate_coherence_basic(obj["content"], obj["version"])
     def _evaluate_creativity(self, obj: Dict) -> float: return min(1.0, 0.4 + obj["version"]*0.1 + len(obj["content"])/1000)
     def _evaluate_helpfulness(self, obj: Dict) -> float: return min(1.0, 0.6 + (0.2 if obj["topic"].lower() in obj["content"].lower() else 0))


# --- Plotting Utility (Requires matplotlib) ---
def plot_results(results: List[Dict[str, Any]], title: str, save_path: str):
    """Generates and saves a plot of scores over iterations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("Matplotlib not installed. Skipping plot generation. `pip install matplotlib`")
        return

    # Extract overall scores from evaluation history dictionaries
    # Assuming iteration 0 is the initial state *before* first evaluation
    scores = [res.get('overall_score', 0.0) for res in results]
    initial_score = results[0].get('initial_score_for_plot', scores[0]) # Use first eval score as initial unless provided
    plot_scores = [initial_score] + scores
    iterations = range(len(plot_scores)) # 0 to N evaluations

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, plot_scores, marker='o', linestyle='-', linewidth=2)
    plt.title(title)
    plt.xlabel("Evaluation Iteration (0 = Initial)")
    plt.ylabel("Overall Score")
    plt.xticks(iterations)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    try:
        # Ensure directory exists and save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Plot saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {save_path}: {e}")
    finally:
        plt.close() # Close the plot to free memory


# --- Async Main Execution Logic ---

async def run_example(example_class, initial_input, name: str):
    """Runs a single example instance asynchronously."""
    logger.info(f"\n--- Running {name} Example (Async Context) ---")
    try:
        example = example_class()
        # Run the example's async run method
        result = await example.run(initial_input)

        # Display summary
        logger.info(f"\n{name} Summary:")
        logger.info(f"  Initial Input: {result.get('initial_input', 'N/A')}")
        logger.info(f"  Iterations (Optimization Cycles): {result.get('iterations', 'N/A')}")
        logger.info(f"  Improvement Summary: {result.get('improvement_summary', 'N/A')}")

        # Display initial/final content (truncated)
        initial_output = result.get('initial_output', {})
        final_output = result.get('final_output', {})
        initial_content = str(initial_output.get('content', initial_output.get('query', 'N/A')))
        final_content = str(final_output.get('content', final_output.get('query', 'N/A')))
        logger.info(f"  Initial Output (v1): {initial_content[:100]}{'...' if len(initial_content)>100 else ''}")
        logger.info(f"  Final Output (v{final_output.get('version','?')}) : {final_content[:100]}{'...' if len(final_content)>100 else ''}")

        # Plot results if history exists
        if result.get("evaluation_history"):
             # Add initial score to history for plotting iteration 0
             history_for_plot = result["evaluation_history"]
             history_for_plot[0]['initial_score_for_plot'] = result.get('initial_output',{}).get('initial_score', history_for_plot[0].get('overall_score', 0.0))

             plot_title = f"{name} Improvement Over Iterations"
             plot_filename = f"plot_{name.lower().replace(' ', '_').replace('(','').replace(')','')}.png"
             plot_save_path = os.path.join(example.config.metrics_path, plot_filename)
             # Run sync plotting function in executor to avoid blocking
             loop = asyncio.get_running_loop()
             await loop.run_in_executor(None, plot_results, history_for_plot, plot_title, plot_save_path)

    except Exception as e:
        logger.error(f"Error running {name} example: {e}", exc_info=True)


async def main():
     """Parses args and runs selected examples asynchronously."""
     parser = argparse.ArgumentParser(description="Run Async Evaluator-Optimizer Examples")
     parser.add_argument("--example", type=str, choices=["text", "query", "creative", "all"], default="all")
     parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
     args = parser.parse_args()

     logging.getLogger().setLevel(args.log_level.upper())
     logging.getLogger('matplotlib').setLevel(logging.WARNING) # Quiet matplotlib logs

     tasks = []
     if args.example == "text" or args.example == "all":
         tasks.append(run_example(TextRefinementExample, "artificial intelligence", "Text Refinement (AI)"))
     if args.example == "query" or args.example == "all":
         tasks.append(run_example(QueryOptimizationExample, {"question": "Find premium customers"}, "Query Optimization"))
     if args.example == "creative" or args.example == "all":
         tasks.append(run_example(CreativeContentExample, {"type": "story", "topic": "mystery", "style": "dramatic"}, "Creative Content (Story)"))

     if tasks:
         await asyncio.gather(*tasks)
         logger.info("\n--- All selected async examples finished ---")
     else:
          logger.info("No examples selected to run.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Example execution interrupted.")