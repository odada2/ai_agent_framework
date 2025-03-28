"""
Evaluator-Optimizer Pattern Example (Refactored)

This module provides refactored example implementations and usage scenarios for the
Evaluator-Optimizer workflow pattern, demonstrating how agents can critique
and improve their own outputs through feedback loops.

Examples include:
1. Text refinement with quality evaluations
2. Query optimization for data accuracy
3. Creative content improvement with feedback

Refactoring focuses on reducing redundancy and improving clarity while maintaining
the core demonstration logic.
"""

import os
import sys
import time
import json
import logging
import random
import argparse
from typing import Dict, List, Any, Callable
from dataclasses import asdict, dataclass

# Add parent directory to path to allow imports
# Note: For larger projects, proper packaging is recommended over sys.path manipulation.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import framework components (assuming they exist in the specified paths)
try:
    from core.workflow.evaluator import (
        EvaluatorOptimizer,
        EvaluationCriterion,
        EvaluationMetric,
        EvaluationResult,
        OptimizationResult,
        create_accuracy_criterion,
        create_relevance_criterion,
        create_completeness_criterion
    )
    from core.workflow.metrics_collector import MetricsCollector
except ImportError as e:
    print(f"Error importing framework components: {e}")
    print("Please ensure the 'core' directory is correctly structured and accessible.")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
METRICS_BASE_PATH = "./metrics"
MAX_ITERATIONS_DEFAULT = 5
IMPROVEMENT_THRESHOLD_DEFAULT = 0.05


# --- Utility Functions for Evaluation (Refactored) ---

def evaluate_coherence_basic(content: str, version: int) -> float:
    """Evaluate coherence using basic heuristics (shared logic)."""
    # Count transition words that indicate logical flow
    transition_words = ["however", "therefore", "thus", "consequently", "furthermore",
                      "moreover", "nevertheless", "in addition", "for example",
                      "specifically", "because", "since", "although", "despite"]
    transition_count = sum(1 for word in transition_words if word in content.lower())
    transition_score = min(0.4, transition_count * 0.06)

    # Texts with balanced sentence lengths are often more coherent
    sentences = [s.strip() for s in content.split(".") if s.strip()]
    variance_score = 0
    if sentences:
        sentence_lengths = [len(s) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentences)
        # Avoid division by zero if len(sentences) is 0, though caught by 'if sentences:'
        length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentences)
        variance_score = 0.2 * max(0, 1 - (length_variance / 1000)) # Penalize high variance

    # Base score improves artificially with version for simulation
    base_score = 0.4 + min(0.2, (version - 1) * 0.08)
    score = base_score + transition_score + variance_score
    return max(0.0, min(1.0, score))


def evaluate_conciseness_basic(content: str, ideal_length: int) -> float:
    """Evaluate conciseness using basic heuristics (shared logic)."""
    if len(content) == 0:
        return 0.0

    # Score based on deviation from ideal length
    if ideal_length <= 0: # Prevent division by zero
        length_score = 0.0
    elif len(content) < ideal_length * 0.5: # Too short
        length_score = 0.5 * (len(content) / (ideal_length * 0.5))
    elif len(content) <= ideal_length * 1.2: # Ideal range
        length_score = 0.5
    else: # Too verbose
        length_score = 0.5 * max(0, 1 - (len(content) - ideal_length * 1.2) / (ideal_length * 2))

    # Penalize filler words
    filler_words = ["very", "really", "quite", "basically", "actually",
                  "simply", "just", "indeed", "certainly", "definitely"]
    words = content.lower().split()
    word_count = len(words)
    filler_score = 0.3
    if word_count > 0:
        filler_count = sum(1 for word in filler_words if word in words)
        filler_ratio = filler_count / word_count
        filler_score = 0.3 * max(0, 1 - filler_ratio * 10) # Penalize high filler ratio

    # Penalize overly long sentences
    sentences = [s.strip() for s in content.split(".") if s.strip()]
    sentence_score = 0.2
    if sentences:
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
        sentence_score = 0.2 * max(0, 1 - (avg_sentence_length / 100)) # Penalize long avg length

    score = length_score + filler_score + sentence_score
    return max(0.0, min(1.0, score))


# --- Base Example Class ---

@dataclass
class BaseExampleConfig:
    """Configuration for base example."""
    metrics_path: str
    criteria: List[EvaluationCriterion]
    max_iterations: int = MAX_ITERATIONS_DEFAULT
    improvement_threshold: float = IMPROVEMENT_THRESHOLD_DEFAULT


class BaseEvaluatorOptimizerExample:
    """Base class for Evaluator-Optimizer examples."""

    def __init__(self, config: BaseExampleConfig, executor: Callable, evaluator: Callable, optimizer: Callable):
        self.config = config
        self.metrics_collector = MetricsCollector(storage_path=config.metrics_path)
        self.evaluator_optimizer = EvaluatorOptimizer(
            evaluator=evaluator,
            optimizer=optimizer,
            executor=executor,
            criteria=config.criteria,
            max_iterations=config.max_iterations,
            improvement_threshold=config.improvement_threshold,
            metrics_collector=self.metrics_collector.collect_workflow_metrics
        )
        self._executor = executor # Store executor for direct call if needed

    def _base_evaluate(self, output_obj: Dict[str, Any], criteria: List[EvaluationCriterion]) -> EvaluationResult:
        """Generic evaluation loop (can be overridden)."""
        output_id = output_obj.get("id", f"output_{int(time.time())}")
        version = output_obj.get("version", 1)
        logger.info(f"Evaluating output '{output_id}' (version {version})")

        scores = {}
        feedback = {}
        passed = True

        for criterion in criteria:
            # Find the specific evaluation method in the subclass
            eval_method_name = f"_evaluate_{criterion.metric.value.lower()}"
            eval_method = getattr(self, eval_method_name, None)

            if eval_method and callable(eval_method):
                score = eval_method(output_obj)
            elif criterion.custom_evaluator: # Fallback to criterion's evaluator if provided
                 score = criterion.custom_evaluator(output_obj)
            else:
                logger.warning(f"No evaluator found for metric {criterion.metric.value}. Defaulting score.")
                score = 0.5 # Default score if no specific method found
                feedback[criterion.metric.value] = "Evaluation logic missing or incomplete."

            scores[criterion.metric.value] = score
            criterion_passed = score >= criterion.threshold
            if not criterion_passed:
                passed = False
                # Generic feedback if specific method didn't add any
                if criterion.metric.value not in feedback:
                     feedback[criterion.metric.value] = f"Score {score:.2f} is below threshold {criterion.threshold:.2f} for {criterion.metric.value}."

        # Calculate overall score
        total_weight = sum(c.weight for c in criteria)
        overall_score = 0
        if total_weight > 0:
             weighted_sum = sum(scores[c.metric.value] * c.weight for c in criteria)
             overall_score = weighted_sum / total_weight

        return EvaluationResult(
            output_id=output_id,
            scores=scores,
            overall_score=overall_score,
            passed=passed,
            feedback=feedback
        )

    def run(self, initial_input: Any) -> Dict[str, Any]:
        """Runs the improvement cycle and returns results."""
        logger.info(f"Running {self.__class__.__name__}...")

        final_output, evaluation_history = self.evaluator_optimizer.run_improvement_cycle(initial_input)
        workflow_id = final_output.get("id", "unknown")

        initial_output = self._executor(initial_input) # Generate initial again for comparison

        # Gather results
        metrics = self.metrics_collector.get_workflow_metrics(workflow_id)
        summary = self.metrics_collector.get_improvement_summary(workflow_id)

        result = {
            "initial_input": initial_input,
            "initial_output": initial_output,
            "final_output": final_output,
            "iterations": len(evaluation_history),
            "evaluation_history": [asdict(e) for e in evaluation_history],
            "improvement_summary": summary,
            "metrics": metrics
        }

        logger.info(f"{self.__class__.__name__} complete. Final version: {final_output.get('version', 'N/A')}")
        return result


# --- Example 1: Text Refinement ---

class TextRefinementExample(BaseEvaluatorOptimizerExample):
    """Example 1: Text refinement with quality evaluations."""

    def __init__(self):
        criteria = [
            create_accuracy_criterion(weight=1.0, threshold=0.8), # Using default evaluator for simplicity now
            create_relevance_criterion(weight=0.8, threshold=0.7),
            create_completeness_criterion(weight=0.6, threshold=0.75),
            EvaluationCriterion(
                metric=EvaluationMetric.COHERENCE, weight=0.7, threshold=0.7,
                description="Measures logical flow and consistency."
            ),
            EvaluationCriterion(
                metric=EvaluationMetric.CONCISENESS, weight=0.5, threshold=0.6,
                description="Evaluates appropriate conciseness."
            )
        ]
        config = BaseExampleConfig(
            metrics_path=os.path.join(METRICS_BASE_PATH, "text_refinement"),
            criteria=criteria
        )
        super().__init__(
            config=config,
            executor=self._generate_initial_text,
            evaluator=self._evaluate_text,
            optimizer=self._optimize_text
        )

    def _generate_initial_text(self, topic: str) -> Dict[str, Any]:
        """Generates initial text (simulation)."""
        logger.info(f"Generating initial text for topic: {topic}")
        initial_texts = {
            "climate change":
                "Climate change is a big problem. The earth is getting hotter. "
                "This is causing ice to melt. Climate change happens because humans do things. "
                "Scientists agree. We need to fix it.",
            "artificial intelligence":
                "AI means computers can think. It gets smarter all the time. "
                "AI is used for many things. Some people worry about AI. "
                "AI is fast. Neural networks are used.",
            "space exploration":
                "Space is cool. NASA sends rockets. We went to the moon. Mars next? "
                "Space is big. Telescopes see far. No gravity in space."
        }
        text = initial_texts.get(topic.lower(), f"Initial text about {topic}. Needs work. Basic info only.")
        return {
            "id": f"text_{topic.replace(' ', '_')}_{int(time.time())}",
            "topic": topic,
            "content": text,
            "version": 1,
            "metadata": {"word_count": len(text.split()), "char_count": len(text)}
        }

    def _evaluate_text(self, text_obj: Dict[str, Any], criteria: List[EvaluationCriterion]) -> EvaluationResult:
        """Evaluates text using specific metric functions."""
        # Use the base evaluation loop, which will call specific _evaluate_* methods
        return self._base_evaluate(text_obj, criteria)

    def _optimize_text(self, text_obj: Dict[str, Any], evaluation: EvaluationResult) -> OptimizationResult:
        """Optimizes text based on feedback (simulation)."""
        logger.info(f"Optimizing text (version {text_obj['version']}) based on feedback")
        current_text = text_obj["content"]
        topic = text_obj["topic"]
        version = text_obj["version"]
        feedback = evaluation.feedback

        # Simulate improvement using an LLM would go here
        # Simple simulation: Make text longer and add topic words/details based on feedback
        improvements = []
        improved_text = current_text
        if "accuracy" in feedback:
            improved_text += f" More research shows specific data points about {topic}."
            improvements.append("Added factual details")
        if "relevance" in feedback:
            improved_text += f" Focusing specifically on {topic} is key."
            improvements.append("Improved topic focus")
        if "completeness" in feedback:
            improved_text += f" Considering other aspects of {topic} provides a fuller picture."
            improvements.append("Enhanced completeness")
        if "coherence" in feedback:
            improved_text = improved_text.replace(". ", ". Therefore, ") # Basic transition simulation
            improvements.append("Improved flow")
        if "conciseness" in feedback:
             # Simulate removing filler (very basic)
            words = improved_text.split()
            non_fillers = [w for w in words if w.lower() not in ["very", "really", "just"]]
            improved_text = " ".join(non_fillers)
            improvements.append("Improved conciseness")

        # Fallback if no specific feedback led to changes
        if not improvements and version < self.config.max_iterations:
             improved_text += f" Further refinement on {topic} was applied."
             improvements.append("General refinement")

        improved_text_obj = {
            **text_obj,
            "content": improved_text,
            "version": version + 1,
            "metadata": {
                "word_count": len(improved_text.split()),
                "char_count": len(improved_text)
            }
        }
        return OptimizationResult(
            original_output_id=text_obj["id"],
            optimized_output=improved_text_obj,
            improvement_summary=", ".join(improvements) or "Minor adjustments",
            optimization_method="simulated_text_refinement"
        )

    # --- Specific Evaluation Methods for Text Refinement ---
    def _evaluate_accuracy(self, text_obj: Dict[str, Any]) -> float:
        # Simulation: improves with version and length
        score = 0.5 + min(0.3, (text_obj["version"] - 1) * 0.1) + min(0.2, len(text_obj["content"]) / 1000)
        return max(0.0, min(1.0, score))

    def _evaluate_relevance(self, text_obj: Dict[str, Any]) -> float:
        # Simulation: score based on topic mentions
        topic_count = text_obj["content"].lower().count(text_obj["topic"].lower())
        score = 0.5 + min(0.5, topic_count * 0.1)
        return max(0.0, min(1.0, score))

    def _evaluate_completeness(self, text_obj: Dict[str, Any]) -> float:
        # Simulation: improves with version and length
        score = 0.4 + min(0.3, (text_obj["version"] - 1) * 0.1) + min(0.3, len(text_obj["content"]) / 1500)
        return max(0.0, min(1.0, score))

    def _evaluate_coherence(self, text_obj: Dict[str, Any]) -> float:
        # Use shared basic coherence evaluation
        return evaluate_coherence_basic(text_obj["content"], text_obj["version"])

    def _evaluate_conciseness(self, text_obj: Dict[str, Any]) -> float:
        # Use shared basic conciseness evaluation with an ideal length guess
        topic = text_obj.get("topic", "")
        ideal_length = 500 if topic in ["climate change", "artificial intelligence"] else 300
        return evaluate_conciseness_basic(text_obj["content"], ideal_length)


# --- Example 2: Query Optimization ---

class QueryOptimizationExample(BaseEvaluatorOptimizerExample):
    """Example 2: Query optimization for data accuracy."""

    # Simulated database
    KNOWLEDGE_BASE = {
        "customer": [{"id": 1, "name": "Alice", "segment": "premium"}, {"id": 2, "name": "Bob", "segment": "standard"}],
        "product": [{"id": 101, "name": "Gadget", "price": 100}, {"id": 102, "name": "Widget", "price": 50}],
        "sales": [{"id": 1, "cust_id": 1, "prod_id": 101, "amount": 100}, {"id": 2, "cust_id": 2, "prod_id": 102, "amount": 50}]
    }

    def __init__(self):
        criteria = [
            create_accuracy_criterion(weight=1.0, threshold=0.8),
            create_relevance_criterion(weight=0.9, threshold=0.75),
            create_completeness_criterion(weight=0.7, threshold=0.7),
        ]
        config = BaseExampleConfig(
            metrics_path=os.path.join(METRICS_BASE_PATH, "query_optimization"),
            criteria=criteria,
            max_iterations=4 # Limit iterations for query example
        )
        super().__init__(
            config=config,
            executor=self._generate_initial_query,
            evaluator=self._evaluate_query,
            optimizer=self._optimize_query
        )

    def _generate_initial_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generates an initial SQL query (simulation)."""
        question = request.get("question", "")
        logger.info(f"Generating initial query for: {question}")
        # Very basic keyword-based query generation
        query = "SELECT * FROM customer" # Default
        if "product" in question.lower():
            query = "SELECT * FROM product"
        if "sales" in question.lower():
            query = "SELECT * FROM sales"
        if "premium" in question.lower():
             query = "SELECT name FROM customer WHERE segment = 'premium'"

        return {
            "id": f"query_{int(time.time())}",
            "question": question,
            "query": query,
            "version": 1,
            "results": [], # Results populated upon execution
            "metadata": {"estimated_cost": random.uniform(1, 5)}
        }

    def _execute_query_simulation(self, query: str) -> List[Dict]:
        """Simulates executing a SQL query."""
        logger.debug(f"Simulating execution: {query}")
        # WARNING: Extremely basic simulation, not a real SQL engine!
        try:
            parts = query.lower().split()
            table = "customer" # Default
            if "from" in parts:
                from_idx = parts.index("from")
                if from_idx + 1 < len(parts):
                    table_name = parts[from_idx + 1].split("where")[0].split("order")[0].strip()
                    if table_name in self.KNOWLEDGE_BASE:
                        table = table_name

            data = self.KNOWLEDGE_BASE.get(table, [])

            # Basic WHERE simulation
            if "where" in parts:
                 where_idx = parts.index("where")
                 if where_idx + 3 < len(parts) and parts[where_idx+2] == "=":
                     field = parts[where_idx + 1]
                     value_str = parts[where_idx + 3].strip("'\"")
                     # Try to infer type (very brittle)
                     value = value_str
                     try: value = int(value_str)
                     except ValueError: pass
                     try: value = float(value_str)
                     except ValueError: pass

                     data = [row for row in data if str(row.get(field)).lower() == str(value).lower()]

            # Basic SELECT simulation
            select_fields = ["*"]
            if "select" in parts:
                 select_idx = parts.index("select")
                 from_idx = parts.index("from") if "from" in parts else len(parts)
                 fields_str = " ".join(parts[select_idx+1:from_idx])
                 if fields_str and fields_str != '*':
                     select_fields = [f.strip() for f in fields_str.split(',')]

            if "*" not in select_fields:
                 data = [{f: row.get(f) for f in select_fields if f in row} for row in data]

            return data
        except Exception as e:
            logger.error(f"Query simulation error for '{query}': {e}")
            return [{"error": str(e)}]


    def _evaluate_query(self, query_obj: Dict[str, Any], criteria: List[EvaluationCriterion]) -> EvaluationResult:
        """Evaluates the query based on simulated execution results."""
        # Execute simulation if results are missing
        if not query_obj.get("results"):
             query_obj["results"] = self._execute_query_simulation(query_obj["query"])

        # Use the base evaluation loop
        return self._base_evaluate(query_obj, criteria)


    def _optimize_query(self, query_obj: Dict[str, Any], evaluation: EvaluationResult) -> OptimizationResult:
        """Optimizes the SQL query based on feedback (simulation)."""
        logger.info(f"Optimizing query (version {query_obj['version']}) based on feedback")
        current_query = query_obj["query"]
        version = query_obj["version"]
        feedback = evaluation.feedback
        question = query_obj["question"]

        # Simulate improvement (very basic)
        # In reality, an LLM might rewrite the query based on feedback/schema/question
        improved_query = current_query
        improvements = []

        # Example: Add WHERE clause if accuracy/relevance is low and keywords exist
        if ("accuracy" in feedback or "relevance" in feedback) and "WHERE" not in current_query.upper():
             if "premium" in question.lower() and "customer" in current_query.lower():
                 improved_query += " WHERE segment = 'premium'"
                 improvements.append("Added WHERE clause for segment")
             elif "gadget" in question.lower() and "product" in current_query.lower():
                  improved_query += " WHERE name = 'Gadget'"
                  improvements.append("Added WHERE clause for product name")

        # Example: Select specific fields if completeness score is okay but relevance is low
        if "relevance" in feedback and "completeness" not in feedback and "SELECT *" in current_query.upper():
             if "name" in question.lower() and "customer" in current_query.lower():
                 improved_query = improved_query.replace("SELECT *", "SELECT name, segment")
                 improvements.append("Selected specific customer fields")
             elif "price" in question.lower() and "product" in current_query.lower():
                 improved_query = improved_query.replace("SELECT *", "SELECT name, price")
                 improvements.append("Selected specific product fields")

        # Fallback if no specific feedback led to changes
        if not improvements and version < self.config.max_iterations:
             # Could add ORDER BY or LIMIT as a generic improvement attempt
             if "ORDER BY" not in current_query.upper():
                 id_field = "id"
                 if "cust_id" in current_query.lower(): id_field = "cust_id"
                 elif "prod_id" in current_query.lower(): id_field = "prod_id"
                 improved_query += f" ORDER BY {id_field} DESC"
                 improvements.append("Added ORDER BY clause")

        improved_query_obj = {
            **query_obj,
            "query": improved_query,
            "version": version + 1,
            "results": [], # Clear results, will be re-evaluated
            "metadata": {
                "estimated_cost": query_obj["metadata"]["estimated_cost"] * random.uniform(0.8, 1.5) # Update cost
            }
        }

        return OptimizationResult(
            original_output_id=query_obj["id"],
            optimized_output=improved_query_obj,
            improvement_summary=", ".join(improvements) or "Minor query adjustments",
            optimization_method="simulated_query_refinement"
        )

    # --- Specific Evaluation Methods for Query Optimization ---
    def _evaluate_accuracy(self, query_obj: Dict[str, Any]) -> float:
        # Simulation: Checks if results match keywords in the question crudely
        question = query_obj["question"].lower()
        results = query_obj["results"]
        score = 0.5
        if not results or "error" in results[0]: return 0.1 # Penalize errors or no results
        if len(results) > 5: score -= 0.2 # Penalize too many results

        match_count = 0
        expected_matches = 0
        if "premium" in question:
            expected_matches += 1
            if any(r.get("segment") == "premium" for r in results): match_count +=1
        if "alice" in question:
            expected_matches += 1
            if any(r.get("name") == "Alice" for r in results): match_count += 1
        if "gadget" in question:
             expected_matches += 1
             if any(r.get("name") == "Gadget" for r in results): match_count += 1

        if expected_matches > 0:
             score = 0.2 + 0.8 * (match_count / expected_matches)
        elif len(results) < 3 and len(results)>0: # Assume few results might be accurate if no keywords match
            score = 0.7

        # Artificial boost by version
        score += min(0.2, (query_obj["version"] - 1) * 0.1)
        return max(0.0, min(1.0, score))

    def _evaluate_relevance(self, query_obj: Dict[str, Any]) -> float:
        # Simulation: Checks if query structure matches question intent crudely
        question = query_obj["question"].lower()
        query = query_obj["query"].lower()
        results = query_obj["results"]
        score = 0.5
        if not results or "error" in results[0]: return 0.1

        # Score based on query complexity and keyword matching
        if "where" in query: score += 0.1
        if "order by" in query: score += 0.05
        if "join" in query: score += 0.1 # JOIN not simulated well, but check anyway

        relevance_terms = ["customer", "product", "sales", "premium", "gadget", "name", "price"]
        question_terms = [term for term in relevance_terms if term in question]
        query_terms = [term for term in relevance_terms if term in query]

        common_terms = len(set(question_terms) & set(query_terms))
        total_terms = len(set(question_terms) | set(query_terms))

        if total_terms > 0:
             term_score = 0.3 * (common_terms / total_terms)
             score += term_score
        elif len(results) > 0: # If results exist but no term match, give some points
             score += 0.1

        # Artificial boost by version
        score += min(0.2, (query_obj["version"] - 1) * 0.05)
        return max(0.0, min(1.0, score))

    def _evaluate_completeness(self, query_obj: Dict[str, Any]) -> float:
        # Simulation: Are we getting *some* results? Penalize '*' if specific fields likely needed.
        results = query_obj["results"]
        query = query_obj["query"].lower()
        question = query_obj["question"].lower()

        if not results or "error" in results[0]: return 0.0
        score = 0.6 # Base score for getting results

        if "select *" in query and ("name" in question or "price" in question or "segment" in question):
            score -= 0.2 # Penalize '*' when specific fields seem relevant

        # Did we get a reasonable number of results? (Very rough check)
        expected_count = 1 # Default
        if "all" in question: expected_count = 3
        if len(results) == 0: score = 0.1
        elif len(results) < expected_count: score *= 0.8 # Penalize fewer results than expected
        # No penalty for more results in this basic sim

        # Artificial boost by version
        score += min(0.2, (query_obj["version"] - 1) * 0.05)
        return max(0.0, min(1.0, score))


# --- Example 3: Creative Content Improvement ---

class CreativeContentExample(BaseEvaluatorOptimizerExample):
    """Example 3: Creative content improvement with feedback."""

    def __init__(self):
        criteria = [
            EvaluationCriterion(
                metric=EvaluationMetric.COHERENCE, weight=0.8, threshold=0.7,
                description="Measures logical flow and consistency."
            ),
            EvaluationCriterion(
                metric=EvaluationMetric.CREATIVITY, weight=1.0, threshold=0.75,
                description="Evaluates originality and novelty."
            ),
             EvaluationCriterion( # Using helpfulness as a proxy for topic adherence
                metric=EvaluationMetric.HELPFULNESS, weight=0.7, threshold=0.7,
                description="Assesses how well the content meets the user's topic/style request."
            )
        ]
        config = BaseExampleConfig(
            metrics_path=os.path.join(METRICS_BASE_PATH, "creative_content"),
            criteria=criteria
        )
        super().__init__(
            config=config,
            executor=self._generate_initial_content,
            evaluator=self._evaluate_content,
            optimizer=self._optimize_content
        )

    def _generate_initial_content(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generates initial creative content (simulation)."""
        content_type = request.get("type", "story")
        topic = request.get("topic", "adventure")
        style = request.get("style", "casual")
        logger.info(f"Generating initial {content_type} about {topic} ({style} style)")

        # Basic templates
        content = f"A {style} {content_type} about {topic}.\n\nThe beginning."
        if content_type == "poem":
            content = f"{topic.capitalize()} ({style.capitalize()})\n\nA line,\nAnother line."
        if topic == "adventure":
             content += "\nThere was danger. And excitement."
        if style == "dramatic":
             content = f"*{topic.upper()}*\n\nIt was a dark and stormy night... a tale of {topic} began."

        return {
            "id": f"creative_{content_type}_{topic}_{int(time.time())}",
            "type": content_type,
            "topic": topic,
            "style": style,
            "content": content,
            "version": 1,
            "metadata": {"word_count": len(content.split()), "line_count": content.count("\n") + 1}
        }

    def _evaluate_content(self, content_obj: Dict[str, Any], criteria: List[EvaluationCriterion]) -> EvaluationResult:
        """Evaluates creative content."""
        return self._base_evaluate(content_obj, criteria)

    def _optimize_content(self, content_obj: Dict[str, Any], evaluation: EvaluationResult) -> OptimizationResult:
        """Optimizes creative content based on feedback (simulation)."""
        logger.info(f"Optimizing {content_obj['type']} (version {content_obj['version']})")
        current_content = content_obj["content"]
        version = content_obj["version"]
        feedback = evaluation.feedback

        # Simulate improvement - LLM would rewrite/expand here
        improved_content = current_content
        improvements = []

        if "coherence" in feedback:
            improved_content += "\n\nHowever, things took an unexpected turn." # Add transition
            improvements.append("Improved coherence")
        if "creativity" in feedback:
             # Add slightly more descriptive language
             if "story" in content_obj["type"]:
                 improved_content += "\nThe setting was vividly imagined."
             elif "poem" in content_obj["type"]:
                 improved_content += "\nLike a metaphor."
             improvements.append("Enhanced creativity")
        if "helpfulness" in feedback: # Interpreted as topic/style adherence
            improved_content += f" (This relates directly to {content_obj['topic']} in a {content_obj['style']} way)."
            improvements.append("Improved topic/style focus")

        # Fallback
        if not improvements and version < self.config.max_iterations:
             improved_content += "\nMore details were added."
             improvements.append("General expansion")

        improved_content_obj = {
            **content_obj,
            "content": improved_content,
            "version": version + 1,
            "metadata": {
                 "word_count": len(improved_content.split()),
                 "line_count": improved_content.count("\n") + 1
             }
        }
        return OptimizationResult(
            original_output_id=content_obj["id"],
            optimized_output=improved_content_obj,
            improvement_summary=", ".join(improvements) or "Minor creative adjustments",
            optimization_method="simulated_creative_refinement"
        )

    # --- Specific Evaluation Methods for Creative Content ---
    def _evaluate_coherence(self, content_obj: Dict[str, Any]) -> float:
        # Use shared basic coherence evaluation
        # Adjust based on type - poems can be less strictly coherent
        base_score = evaluate_coherence_basic(content_obj["content"], content_obj["version"])
        if content_obj["type"] == "poem":
            base_score = (base_score + 1.0) / 2 # Allow lower coherence for poems
        return base_score

    def _evaluate_creativity(self, content_obj: Dict[str, Any]) -> float:
        # Simulation: Score based on length, version, and presence of "creative" words
        content = content_obj["content"].lower()
        score = 0.4 + min(0.2, (content_obj["version"] - 1) * 0.1) # Version boost
        score += min(0.2, len(content) / 500) # Length boost

        creative_words = ["vividly", "imagine", "unexpected", "metaphor", "essence"]
        if any(word in content for word in creative_words):
            score += 0.2
        return max(0.0, min(1.0, score))

    def _evaluate_helpfulness(self, content_obj: Dict[str, Any]) -> float:
        # Simulation: Does it mention the topic and align somewhat with style?
        content = content_obj["content"].lower()
        topic = content_obj["topic"].lower()
        style = content_obj["style"].lower()
        score = 0.5

        if topic in content:
             score += 0.2
        else: score -= 0.1

        # Crude style check
        if style == "dramatic" and ("dark and stormy" in content or "suddenly" in content):
            score += 0.1
        elif style == "casual" and content.count('.') < 5: # Very casual might use fewer sentences
             score += 0.05
        elif style == "formal" and ("however" in content or "therefore" in content):
            score += 0.1

        score += min(0.1, (content_obj["version"] - 1) * 0.05) # Version boost
        return max(0.0, min(1.0, score))


# --- Plotting Utility ---

def plot_results(results: List[Dict[str, Any]], title: str, save_path: str):
    """Generates and saves a plot of scores over iterations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("Matplotlib not installed. Skipping plot generation.")
        return

    plt.figure(figsize=(10, 6))
    iterations = range(len(results) + 1) # 0 to N iterations
    # Extract overall scores, assuming 0.0 before first iteration
    scores = [0.0] + [e["overall_score"] for e in results]

    plt.plot(iterations, scores, marker='o', linestyle='-', linewidth=2)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Overall Score")
    plt.xticks(iterations)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ensure directory exists and save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        plt.savefig(save_path)
        logger.info(f"Plot saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {save_path}: {e}")
    finally:
        plt.close() # Close the plot to free memory


# --- Main Execution Logic ---

def run_example(example_class, initial_input, name: str):
    """Runs a single example instance."""
    logger.info(f"\n--- Running {name} Example ---")
    try:
        example = example_class()
        result = example.run(initial_input)

        # Display summary
        logger.info(f"\n{name} Summary:")
        logger.info(f"  Initial Input: {result['initial_input']}")
        logger.info(f"  Iterations: {result['iterations']}")
        logger.info(f"  Improvement Summary: {result['improvement_summary']}")

        # Display initial/final content (truncated if long)
        initial_content = str(result.get('initial_output', {}).get('content', result.get('initial_output', {}).get('query', 'N/A')))
        final_content = str(result.get('final_output', {}).get('content', result.get('final_output', {}).get('query', 'N/A')))
        logger.info(f"  Initial Output: {initial_content[:100]}{'...' if len(initial_content)>100 else ''}")
        logger.info(f"  Final Output:   {final_content[:100]}{'...' if len(final_content)>100 else ''}")

        # Plot results
        plot_title = f"{name} Improvement Over Iterations"
        plot_filename = f"plot_{name.lower().replace(' ', '_')}.png"
        plot_save_path = os.path.join(example.config.metrics_path, plot_filename)
        plot_results(result["evaluation_history"], plot_title, plot_save_path)

    except Exception as e:
        logger.error(f"Error running {name} example: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluator-Optimizer pattern examples")
    parser.add_argument(
        "--example", type=str, choices=["text", "query", "creative", "all"], default="all",
        help="Which example to run (default: all)"
    )
    parser.add_argument(
        "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
        help="Set logging level (default: INFO)"
    )
    args = parser.parse_args()

    # Update root logger level
    logging.getLogger().setLevel(args.log_level.upper())

    if args.example == "text" or args.example == "all":
        run_example(TextRefinementExample, "artificial intelligence", "Text Refinement (AI)")

    if args.example == "query" or args.example == "all":
        run_example(QueryOptimizationExample, {"question": "Find premium customers"}, "Query Optimization")

    if args.example == "creative" or args.example == "all":
        run_example(CreativeContentExample, {"type": "story", "topic": "mystery", "style": "dramatic"}, "Creative Content (Story)")

    logger.info("\n--- All selected examples finished ---")