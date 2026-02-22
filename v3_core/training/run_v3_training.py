import sys
import os
import json
import torch
from pathlib import Path
from loguru import logger
import time

# Add project root explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from v3_core.training.v3_trainer import V3GenerativeTrainer
from v2_core.architecture.holographic_compressor import HolographicCompressor

def build_training_pipeline(epochs: int = 100):
    """
    The Entry Point for SNAP-C1 V3 Model Training.
    
    This script connects the mathematical sequence data (v3_seed_dataset.json) 
    to the Lightning-fast Offline Execution Engine (v3_trainer.py).
    """
    logger.info("Initializing SNAP-C1 V3 Generative Reasoning Pipeline...")
    
    dataset_path = Path(__file__).parent.parent / "data" / "v3_seed_dataset.json"
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}. Please run generate_dataset.py first.")
        sys.exit(1)
        
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    trainer = V3GenerativeTrainer()
    device = trainer.device
    
    from v3_core.data.ast_parser import ASTGraphParser
    parser = ASTGraphParser()
    
    logger.info(f"Loaded {len(dataset)} execution-trace mathematical sequences.")
    logger.info(f"Beginning continuous offline pre-training for {epochs} epochs on {device}...")
    
    # In V2, we needed Tiktoken logic here. In V3, the Holographic Compressor handles it directly.
    compressor = HolographicCompressor(d_model=1024).to(device)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for task in dataset:
            # 1. The original "Prompt" logic (e.g. def solve_math(): ...)
            raw_code = task["original_code"]
            
            # Dynamically parse the original code into a mathematical integer target sequence!
            graph = parser.parse_to_graph(raw_code)
            target_sequence = [n["type"] for n in graph["nodes"]]
            
            # Compress the human prompt into the 1024-dimension V2 Holo-Concept vector
            target_hologram = compressor.process_string(raw_code, device=device)
            # Pool to maintain 2D shape [batch, dim] required by V3 AST Decoder input
            target_hologram = target_hologram.mean(dim=1) 
            
            # 2. Fire the instantaneous Offline Math Loss Engine with real CrossEntropy 
            # This triggers the forward pass entirely inside the trainer natively
            loss_value, branch_loss, bi_directional_loss = trainer.continuous_offline_epoch(target_hologram, target_sequence)
            
            # The Critical Missing Step: Actually Backpropagate the Error down the V3 Tree!
            trainer.optimizer.zero_grad()
            total_loss = loss_value + branch_loss + bi_directional_loss
            total_loss.backward()
            
            # Physically clip the gradients to prevent NaN explosion in the ODE solver
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
            
            trainer.optimizer.step()
            
            epoch_loss += total_loss.item() 
            
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Epoch [{epoch}/{epochs}] | Offline Dual-Loss: {epoch_loss/len(dataset):.4f} | Time Elapsed: {elapsed:.2f}s")
            
    # Save the brand new Generative Weights
    save_path = Path(project_root) / "v3_core" / "snapshot_v3_generative_LTC_core.pt"
    torch.save(trainer.model.state_dict(), save_path)
    logger.success(f"\nTraining Complete. V3 weights physically saved to {save_path}")

if __name__ == "__main__":
    # Boot the pipeline for 100 rapid VRAM epochs
    build_training_pipeline(epochs=100)
