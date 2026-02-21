import json
import logging
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dpo_buffer"

class DPOCollector:
    """
    Parses execution trajectories from FlowController and extracts 
    (Rejected, Chosen) pairs for Failure-Contrastive DPO.
    """
    
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.buffer_file = DATA_DIR / "realtime_trajectory.jsonl"
        
    def collect_from_trajectory(self, trajectory: dict) -> bool:
        """
        Takes a raw trajectory from FlowController, finds the moment 
        where a failure was corrected, and saves it for real-time DPO.
        
        Returns True if a valid (chosen, rejected) pair was found.
        """
        issue = trajectory.get("issue", "")
        patch_log = trajectory.get("patch_log", [])
        
        if not trajectory.get("success", False):
            logger.info("Trajectory was not successful (no final fix). Discarding for DPO.")
            return False
            
        if len(patch_log) < 2:
            logger.info("Trajectory succeeded on the first try. No negative/rejected example for contrastive learning.")
            return False
            
        # The classic Failure-Contrastive pattern:
        # User -> Model attempts fix (Mistake / Rejected) -> Executor says "Test Failed" -> Model self-corrects (Chosen) -> Executor says "Success"
        
        # In a real run, the FlowController's internal history would hold the exact prompts.
        # For our architecture, we take the LAST successful response as Chosen,
        # and the preceding FAILED response as Rejected.
        
        rejected_response = patch_log[-2]
        chosen_response = patch_log[-1]
        
        # We need the prompt context right before the decision was made.
        prompt_context = (
            f"ISSUE DESCRIPTION:\n{issue}\n\n"
            "You are in PHASE 3: FIX AND VERIFY.\n"
            "The reproduction script `reproduce.py` has been written and confirms the bug exists.\n"
            "What is your patch?"
        )
        
        dpo_pair = {
            "prompt": prompt_context,
            "chosen": chosen_response,
            "rejected": rejected_response
        }
        
        self._append_to_buffer(dpo_pair)
        logger.info("Harvested valid DPO pair from trajectory!")
        return True
        
    def _append_to_buffer(self, dpo_pair: dict):
        with open(self.buffer_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(dpo_pair) + "\n")
            
    def get_latest_pair(self) -> dict | None:
        """Reads the last pair from the buffer for real-time 1-step training."""
        if not self.buffer_file.exists():
            return None
            
        with open(self.buffer_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                return json.loads(lines[-1].strip())
        return None
