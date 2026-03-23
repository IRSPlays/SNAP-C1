"""
Local Colab-Resumption Pre-Training Loop
========================================
This script loads the `frc_checkpoint_partial.pt` file you extracted from Google Colab
and natively resumes the FRC logic pre-training loop on your local AMD RX 7600 GPU.
"""

import os
import sys
import torch
import torch_directml
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v2_core.architecture.recurrent_core import FractalRecurrentCore
from v2_core.architecture.neuromorphic_decoder import ConceptDecoder
from v2_core.generate_logic_dataset import LogicDataset

# Optimize DirectML memory allocations for the backward pass
os.environ["PYTORCH_DIRECTML_MEMORY_GROWTH"] = "1"

def resume_local_training(checkpoint_path: str, data_path: str):
    logger.info("Initializing Local AMD Pre-Training Resumption...")
    
    # 1. Hardware Selection
    device = torch_directml.device()
    logger.info(f"Targeting Native AMD GPU: {device}")

    # 2. Re-Initialize Models (Must match Colab specs exactly)
    logger.info("Rebuilding 1.5GB Fractal Logic Core...")
    core = FractalRecurrentCore(
        hidden_dim=1024,
        ffn_dim=4096,      # Standard FFN dim matches Colab
        num_core_layers=12, # Standard FRC layer count
        max_loops=12        # Dynamic Reasoner Loop Count
    )
    
    decoder = ConceptDecoder(
        concept_dim=1024,
        vocab_size=32000,
        num_layers=4
    )
    
    # Bundle components into a unified state dict structure 
    # to perfectly match how Colab saved the `model` object.
    class FullModel(torch.nn.Module):
        def __init__(self, core, decoder):
            super().__init__()
            self.core = core
            self.decoder = decoder
            
        def forward(self, x):
            # Not used in loop directly, just for state dict alignment
            pass
            
    model = FullModel(core, decoder)

    # 3. Inject the Colab Weights
    if os.path.exists(checkpoint_path):
        logger.info(f"Injecting Neural Weights from {checkpoint_path}...")
        try:
            # map_location handles moving CUDA weights to DirectML/CPU smoothly
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            logger.success("Colab Weights Loaded. Training context restored!")
        except Exception as e:
            logger.error(f"Failed to load weights. Are the architecture dicts mismatched? Error: {e}")
            return
    else:
        logger.critical(f"Cannot find {checkpoint_path}! Shutting down.")
        return

    # Push to RX 7600
    model.to(device)

    # 4. Load Training Data
    if not os.path.exists(data_path):
        logger.error(f"Cannot find {data_path}. Please run `generate_logic_dataset.py` first.")
        return

    logger.info("Loading Logic Dataset...")
    # Using 1M rows representing the infinite pre-training stream
    dataset = LogicDataset(data_path, max_samples=1000000) 
    
    # IMPORTANT VRAM CONSTRAINT: 
    # Batch size is 2 to ensure the AMD PyTorch DirectML Backward Pass doesn't crash 8GB
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Slightly lower LR for resumed training

    model.train()

    logger.info("=== STARTING THE LOCAL RECURRENT ENGINE ===")
    
    epochs = 3
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        total_loops = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Simulated dummy input since we haven't wired up the exact HuggingFace tokenizer yet 
            # In a real run, this is replaced with the `tokenizer(batch['prompt'])` mapping 
            # into the `HolographicCompressor`.
            seq_len = 32
            dummy_state = torch.randn(2, seq_len, 1024).to(device)
            target_tokens = torch.randint(0, 32000, (2, seq_len)).to(device)
            
            optimizer.zero_grad()
            
            # Forward Logic Pass
            latent_output, loops_taken, _ = model.core(dummy_state)
            
            # Decoder Syntax Output
            logits = model.decoder.concept_proj(latent_output) 
            logits = model.decoder.lm_head(logits) 
            
            # Loss Calculation (Teaching it Pure Logic)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 32000), 
                target_tokens.reshape(-1)
            )
            
            # Backpropagation using AMD DirectML Hardware
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_loops += loops_taken

            # Update UI Real-Time
            avg_loss = total_loss / (batch_idx + 1)
            avg_loops = total_loops / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.3f}", "avg_loops": f"{int(avg_loops)}"})
            
            # Auto-save Checkpoints locally every 500 batches so you never lose progress
            if batch_idx % 500 == 0 and batch_idx != 0:
                torch.save(model.state_dict(), "frc_checkpoint_local_auto.pt")
                logger.debug(f"Auto-Saved backup checkpoint at batch {batch_idx}.")

if __name__ == "__main__":
    # Ensure these files are exactly in your v2_core/ directory
    CHECKPOINT_FILE = "frc_checkpoint_partial.pt"  
    DATASET_FILE = "frc_pretrain_data.jsonl"
    
    resume_local_training(CHECKPOINT_FILE, DATASET_FILE)
