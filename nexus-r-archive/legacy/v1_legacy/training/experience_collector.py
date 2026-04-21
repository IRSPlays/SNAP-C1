"""
SNAP-C1 v2 Experience Collector: The "Dojo" Logger
==================================================
Logs structured experiences (Prompt, Thought, Action, Result) for self-evolution.
These logs are used to fine-tune the model on its own successful reasoning.

Data Schema:
{
    "id": "exp_12345",
    "timestamp": 1715000000,
    "instruction": "Fix bug in...",
    "thought_trace": "<think>...</think><research>...</research>...",
    "final_answer": "...",
    "feedback": {
        "code_execution_success": true,
        "user_accepted": true,
        "latency_ms": 4500
    },
    "score": 0.95
}
"""

import json
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "self_improving"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BUFFER_FILE = DATA_DIR / "buffer.jsonl"


@dataclass
class Experience:
    instruction: str
    thought_trace: str
    final_answer: str
    feedback: dict
    timestamp: float = 0.0
    id: str = ""
    score: float = 0.5

    def __post_init__(self):
        if not self.id:
            self.id = f"exp_{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = time.time()


class ExperienceCollector:
    """Collects and manages self-improvement experiences."""
    
    def __init__(self):
        self.buffer_path = BUFFER_FILE
        self._ensure_buffer()

    def _ensure_buffer(self):
        if not self.buffer_path.exists():
            with open(self.buffer_path, "w") as f:
                pass  # Create empty file

    def log(self, instruction: str, thought_trace: str, final_answer: str, feedback: dict):
        """Log a new experience."""
        
        # calculate basic score
        score = 0.5
        if feedback.get("code_execution_success"):
            score += 0.3
        if feedback.get("user_accepted"):
            score += 0.2
            
        # Uncertainty-Triggered Online Learning: Boost score for hard-won successes
        if feedback.get("code_execution_success") and feedback.get("execution_failures", 0) > 0:
            score += 0.2  # Bonus for recovering from a failure (Agentic Resilience)
            logger.info(f"Self-Evolution: Boosted score for recovery (failures: {feedback.get('execution_failures')})")
            
        if feedback.get("code_execution_success") and feedback.get("duplicate_actions", 0) > 0:
            score -= 0.1  # Slight penalty for looping to avoid reinforcing bad habits
        
        # Filter low-quality logs
        if not thought_trace.strip():
            logger.warning("Skipping experience log: empty thought trace")
            return

        exp = Experience(
            instruction=instruction,
            thought_trace=thought_trace,
            final_answer=final_answer,
            feedback=feedback,
            score=score
        )
        
        try:
            with open(self.buffer_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(exp)) + "\n")
            logger.info(f"Logged experience {exp.id} (score={score:.2f})")
        except Exception as e:
            logger.error(f"Failed to log experience: {e}")

    def get_high_quality_experiences(self, min_score: float = 0.8) -> list[dict]:
        """Retrieve high-quality experiences for training."""
        experiences = []
        if not self.buffer_path.exists():
            return []
            
        with open(self.buffer_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("score", 0) >= min_score:
                        experiences.append(data)
                except json.JSONDecodeError:
                    continue
        return experiences
