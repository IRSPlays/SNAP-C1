"""
SNAP-C1 Self-Evaluation Module
================================
Allows SNAP-C1 to evaluate its own responses and generate
improvement data for the self-improvement loop.

This module:
1. Scores responses on multiple dimensions
2. Identifies specific failure modes
3. Generates corrected training examples from failures
4. Logs evaluation data for retraining triggers
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
GENERATED_DATA_DIR = PROJECT_ROOT / "data" / "generated"


@dataclass
class EvaluationResult:
    """Result of a self-evaluation."""
    query: str
    response: str
    scores: dict          # Dimension → score (0.0 to 1.0)
    overall_score: float  # Weighted average
    issues: list[str]     # Identified problems
    suggested_fix: str    # How the response should be improved
    timestamp: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class SelfEvaluator:
    """Evaluates SNAP-C1's own responses."""
    
    DIMENSIONS = {
        "correctness": {"weight": 0.3, "description": "Is the answer factually/logically correct?"},
        "completeness": {"weight": 0.2, "description": "Does it fully address the query?"},
        "clarity": {"weight": 0.15, "description": "Is the response clear and well-structured?"},
        "reasoning_depth": {"weight": 0.15, "description": "Does it show deep multi-perspective reasoning?"},
        "self_awareness": {"weight": 0.1, "description": "Does it acknowledge uncertainty appropriately?"},
        "actionability": {"weight": 0.1, "description": "Are the suggestions practical and actionable?"},
    }
    
    def __init__(self, eval_model_fn=None):
        """
        Args:
            eval_model_fn: Function that takes a prompt and returns a model response.
                          Used for model-based self-evaluation.
                          If None, uses heuristic evaluation.
        """
        self.eval_model_fn = eval_model_fn
        self.eval_log: list[EvaluationResult] = []
        
        GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, query: str, response: str) -> EvaluationResult:
        """Evaluate a query-response pair.
        
        Uses heuristic checks + optional model-based evaluation.
        """
        scores = {}
        issues = []
        
        # Heuristic checks
        scores["correctness"] = self._check_correctness(response, issues)
        scores["completeness"] = self._check_completeness(query, response, issues)
        scores["clarity"] = self._check_clarity(response, issues)
        scores["reasoning_depth"] = self._check_reasoning_depth(response, issues)
        scores["self_awareness"] = self._check_self_awareness(response, issues)
        scores["actionability"] = self._check_actionability(query, response, issues)
        
        # Weighted overall score
        overall = sum(
            scores[dim] * info["weight"]
            for dim, info in self.DIMENSIONS.items()
        )
        
        # Generate suggested fix
        suggested_fix = self._generate_fix_suggestion(query, response, issues)
        
        result = EvaluationResult(
            query=query,
            response=response,
            scores=scores,
            overall_score=overall,
            issues=issues,
            suggested_fix=suggested_fix,
            timestamp=time.time(),
        )
        
        self.eval_log.append(result)
        
        # Log if score is low
        if overall < 0.6:
            logger.warning(f"Low score ({overall:.2f}): {query[:80]}...")
            self._save_for_retraining(result)
        
        return result
    
    def _check_correctness(self, response: str, issues: list) -> float:
        """Heuristic correctness checks."""
        score = 0.7  # Base assumption: probably correct
        
        # Red flags
        hedging_phrases = ["I think", "I believe", "probably", "maybe", "I'm not sure"]
        hedging_count = sum(1 for p in hedging_phrases if p.lower() in response.lower())
        
        if hedging_count > 3:
            score -= 0.2
            issues.append("Excessive hedging suggests uncertainty about correctness")
        
        # Very short responses are suspicious
        if len(response) < 50:
            score -= 0.3
            issues.append("Response is very short — may be incomplete or superficial")
        
        # Check for self-contradictions (simple heuristic)
        sentences = response.split(".")
        if any("however" in s.lower() and "but" in s.lower() for s in sentences):
            # Not necessarily wrong, but worth flagging
            pass
        
        return max(0.0, min(1.0, score))
    
    def _check_completeness(self, query: str, response: str, issues: list) -> float:
        """Check if the response fully addresses the query."""
        score = 0.7
        
        # Check if response length is proportional to query complexity
        query_words = len(query.split())
        response_words = len(response.split())
        
        if query_words > 30 and response_words < 50:
            score -= 0.3
            issues.append("Complex query got a short response")
        
        # Check for question marks in query (multiple questions?)
        questions = query.count("?")
        if questions > 1 and response_words < questions * 30:
            score -= 0.2
            issues.append(f"Query has {questions} questions but response seems to address fewer")
        
        return max(0.0, min(1.0, score))
    
    def _check_clarity(self, response: str, issues: list) -> float:
        """Check response clarity and structure."""
        score = 0.7
        
        # Structural indicators
        has_headers = "#" in response or "**" in response
        has_lists = "- " in response or "1." in response
        has_code = "```" in response
        has_paragraphs = "\n\n" in response
        
        structure_score = sum([has_headers, has_lists, has_paragraphs]) / 3
        score = 0.5 + (structure_score * 0.5)
        
        # Very long unbroken text is hard to read
        lines = response.split("\n")
        long_lines = sum(1 for l in lines if len(l) > 200)
        if long_lines > 3:
            score -= 0.2
            issues.append("Contains very long unbroken paragraphs")
        
        return max(0.0, min(1.0, score))
    
    def _check_reasoning_depth(self, response: str, issues: list) -> float:
        """Check for multi-perspective reasoning."""
        score = 0.3  # Base: no special reasoning detected
        
        # Check for team thinking markers
        perspectives = ["[Architect]", "[Critic]", "[Researcher]", "[Implementer]", "[Synthesizer]"]
        found = sum(1 for p in perspectives if p in response)
        
        if found >= 3:
            score = 0.9
        elif found >= 1:
            score = 0.6
        
        # Check for thinking tags
        if "<think>" in response:
            score = max(score, 0.7)
        
        # Check for tradeoff analysis
        tradeoff_words = ["tradeoff", "trade-off", "however", "on the other hand", "alternatively"]
        if sum(1 for w in tradeoff_words if w.lower() in response.lower()) >= 2:
            score = max(score, 0.6)
        
        if score < 0.5:
            issues.append("Response lacks multi-perspective reasoning")
        
        return score
    
    def _check_self_awareness(self, response: str, issues: list) -> float:
        """Check for appropriate uncertainty acknowledgment."""
        score = 0.6
        
        # Good: acknowledges limitations
        awareness_markers = [
            "I'm not certain",
            "this depends on",
            "there are cases where",
            "one limitation",
            "caveat",
            "assuming",
        ]
        
        found = sum(1 for m in awareness_markers if m.lower() in response.lower())
        if found >= 1:
            score = 0.8
        
        return score
    
    def _check_actionability(self, query: str, response: str, issues: list) -> float:
        """Check if the response provides actionable guidance."""
        score = 0.5
        
        # Check for code examples
        if "```" in response:
            score += 0.2
        
        # Check for step-by-step instructions
        if any(f"{i}." in response for i in range(1, 6)):
            score += 0.2
        
        # Check for concrete recommendations
        action_words = ["use", "implement", "start with", "install", "run", "create", "add"]
        if sum(1 for w in action_words if w.lower() in response.lower()) >= 2:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_fix_suggestion(self, query: str, response: str, issues: list) -> str:
        """Generate a suggestion for how to improve the response."""
        if not issues:
            return "Response looks good. No major issues detected."
        
        suggestions = ["Suggested improvements:"]
        for issue in issues:
            if "short" in issue.lower():
                suggestions.append("- Expand the response with more detail and examples")
            if "reasoning" in issue.lower():
                suggestions.append("- Add multi-perspective analysis (Architect, Critic, Researcher views)")
            if "hedging" in issue.lower():
                suggestions.append("- Be more decisive; state facts confidently, hedge only on genuinely uncertain claims")
            if "long" in issue.lower() or "unbroken" in issue.lower():
                suggestions.append("- Break up long paragraphs; use headers, lists, or code blocks")
            if "questions" in issue.lower():
                suggestions.append("- Address each question in the query explicitly")
        
        return "\n".join(suggestions)
    
    def _save_for_retraining(self, result: EvaluationResult):
        """Save a low-scoring example for potential retraining."""
        output_path = GENERATED_DATA_DIR / "self_improvement_candidates.jsonl"
        
        entry = {
            "instruction": result.query,
            "initial_response": result.response,
            "issues": result.issues,
            "suggested_fix": result.suggested_fix,
            "scores": result.scores,
            "overall_score": result.overall_score,
            "timestamp": result.timestamp,
        }
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved low-score example for retraining: {output_path}")
    
    def get_stats(self) -> dict:
        """Get evaluation statistics."""
        if not self.eval_log:
            return {"total_evaluations": 0}
        
        scores = [r.overall_score for r in self.eval_log]
        
        return {
            "total_evaluations": len(self.eval_log),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "low_score_count": sum(1 for s in scores if s < 0.6),
            "high_score_count": sum(1 for s in scores if s >= 0.8),
        }
    
    def should_retrain(self, threshold: int = 100) -> bool:
        """Check if enough examples have accumulated to trigger retraining."""
        candidates_path = GENERATED_DATA_DIR / "self_improvement_candidates.jsonl"
        
        if not candidates_path.exists():
            return False
        
        with open(candidates_path, "r") as f:
            count = sum(1 for _ in f)
        
        return count >= threshold
