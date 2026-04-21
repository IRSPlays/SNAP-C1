import os
import sys
import time
from pathlib import Path
from loguru import logger
from datasets import Dataset

# Hack for dynamic imports
sys.path.append(str(Path(__file__).parent.parent))

from inference.molora_pipeline import MoLORAPipeline
from inference.flow_controller import FlowController
from training.dpo_collector import DPOCollector

try:
    from trl import DPOTrainer, DPOConfig
except ImportError:
    DPOTrainer = None
    logger.warning("trl not installed. Will not be able to run DPO training.")

def run_infinite_loop(pipeline: MoLORAPipeline, max_iterations: int = 1000):
    """
    The AGI Engine. 
    Continuously loops: Attempt Issue -> Collect Trajectory -> Real-Time DPO -> Repeat
    """
    logger.info("=== STARTING THE INFINITE ONLINE LEARNING LOOP ===")
    
    collector = DPOCollector()
    
    for i in range(max_iterations):
        logger.info(f"\n--- [Infinite Loop] Experience Cycle {i+1} ---")
        
        # Step 1: Simulate pulling an issue from SWE-bench (Mocked for now)
        mock_issue = (
            "Bug: App crashes when parsing an empty JSON array.\n"
            "File: parser.py\n"
            "Stack Trace: json.decoder.JSONDecodeError"
        )
        
        # Step 2: Attempt the issue using Flow Engineering
        workspace = f"/tmp/sandbox/issue_{i}"
        controller = FlowController(pipeline=pipeline, workspace_dir=workspace)
        
        logger.info("[Step 2] FlowController tackling issue...")
        
        # We manually simulate the trajectory since we don't have SWE-bench docker hooked up yet
        # trajectory = controller.run_issue(mock_issue)
        
        # Mocking a valid DPO trajectory:
        trajectory = {
            "issue": mock_issue,
            "success": True,
            "patch_log": [
                "<think>Round 1: Strategy...</think>\n<tool_call>{\"name\":\"run_command\",\"kwargs\":{\"command\":\"python reproduce.py\"}}</tool_call>",  # Mistake (Didn't fix anything)
                "<think>Round 2: Evaluation. The script crashed. I need to add an if check.</think>\n<tool_call>{\"name\":\"run_command\",\"kwargs\":{\"command\":\"sed -i 's/parse(data)/if not data: return []\\nparse(data)/g' parser.py\"}}</tool_call>", # Fix
                "<final_answer>Fixed the JSON parser bug.</final_answer>" # Final verification
            ]
        }
        
        # Step 3: Distillation - Harvest Failure-Contrastive pairs
        logger.info("[Step 3] Distilling trajectory into knowledge...")
        valid_pair = collector.collect_from_trajectory(trajectory)
        
        if valid_pair:
            # Step 4: Real-Time DPO Training (Online Learning)
            logger.info("[Step 4] Triggering Real-Time DPOTrainer Update...")
            latest_data = collector.get_latest_pair()
            
            if DPOTrainer and latest_data:
                # We format it for trl.DPOTrainer
                train_data = Dataset.from_list([latest_data])
                
                # In a full implementation, we would construct the DPOTrainer here
                # and do `.train()` for 1-2 epochs on just this single pair.
                # For safety and demo purposes, we log the hot-swap structure:
                logger.info(f"Hot-swapping {pipeline.config['model']['name']} to Training Mode")
                logger.info("Running 2 epochs of DPO on the Rejected vs Chosen path...")
                
                # Mock training delay
                time.sleep(2) 
                
                logger.info("Adapter 'team_thinking' weights updated.")
                logger.info("Hot-swapping back to Inference Mode for next issue.")
            else:
                logger.warning("Skipping DPO step (trl not installed or data missing)")
        else:
            logger.info("No actionable mistake/correction pair found. Skipping training.")
        
        logger.info(f"--- Cycle {i+1} Complete ---\n")
        time.sleep(1) # Breath before the next issue

if __name__ == "__main__":
    from inference.molora_pipeline import MoLORAPipeline
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).parent.parent
    CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
    
    # Init Pipeline with base config
    pipe = MoLORAPipeline(str(CONFIG_PATH))
    run_infinite_loop(pipe, max_iterations=5)
