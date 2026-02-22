import sys
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import json
import time
import argparse

# Add project root explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from v4_core.architecture.v4_assembly import V4HyperAssembly
from v4_core.utils.device import get_device

# ============================================================
#  PARALLELIZATION 1: PyTorch Dataset + DataLoader
# ============================================================
class V4TrainingDataset(Dataset):
    """Wraps the JSON dataset into a proper PyTorch Dataset for batched loading."""
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        logger.info(f"V4 Dataset loaded: {len(self.data)} chunks from {json_path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item.get("v4_trace_input", ""),
            "target_nodes": len(item.get("v4_geometric_target", {}).get("nodes", [])),
        }

def collate_fn(batch):
    """Custom collate to handle string prompts in batches."""
    return {
        "prompts": [item["prompt"] for item in batch],
        "target_nodes": torch.tensor([item["target_nodes"] for item in batch], dtype=torch.float32),
    }


class V4DistributedTrainer:
    """
    SNAP-C1 V4: Parallelized Multi-GPU Training Pipeline
    
    TRUE BATCHED INFERENCE:
      All chunks in a batch are processed as a single [B, 1, 1024] tensor
      through the entire neural pipeline in one GPU kernel launch.
    
    OPTIMIZATIONS:
      1. BATCH PARALLELISM   — [B, 1, 1024] tensors saturate GPU compute
      2. AMP MIXED PRECISION — FP16 matmuls with FP32 accumulation
      3. TORCH.COMPILE       — JIT fuses kernels on pure-tensor submodules
      4. DATALOADER           — Multi-worker async prefetching
    """
    def __init__(self, d_model: int = 1024, max_loops: int = 50):
        self.device = get_device()
        self.d_model = d_model
        self.is_cuda = torch.cuda.is_available()
        
        # Determine Distributed Rank
        self.is_distributed = torch.distributed.is_initialized()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            logger.info(f"V4 DDP Worker Node {self.local_rank}/{self.world_size} online.")
        else:
            logger.info(f"V4 Trainer running in Standalone mode on: {self.device}")

        # Initialize the Master Matrix
        self.model = V4HyperAssembly(d_model=d_model, max_loops=max_loops)
        
        # ============================================================
        #  PARALLELIZATION 3: torch.compile on pure-tensor submodules
        #  (Skips untraceable ops like ChromaDB and loguru)
        # ============================================================
        if self.is_cuda and hasattr(torch, 'compile'):
            try:
                self.model.logic_core = torch.compile(self.model.logic_core, mode="reduce-overhead")
                self.model.compressor = torch.compile(self.model.compressor, mode="reduce-overhead")
                logger.info("torch.compile() ENABLED on logic_core + compressor")
            except Exception as e:
                logger.warning(f"torch.compile() skipped: {e}")
        
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # ============================================================
        #  PARALLELIZATION 2: Automatic Mixed Precision (AMP)
        # ============================================================
        self.scaler = torch.amp.GradScaler('cuda') if self.is_cuda else None
        self.amp_enabled = self.is_cuda
        
        if self.amp_enabled:
            logger.info("AMP Mixed Precision ENABLED — FP16 Tensor Core acceleration active")

    def train_step(self, batch: dict) -> float:
        """
        TRUE BATCHED training step.
        The entire batch is processed as one [B, 1, 1024] tensor.
        Loss is computed through a differentiable projection head.
        """
        self.optimizer.zero_grad(set_to_none=True)
        
        prompts = batch["prompts"]
        target_nodes = batch["target_nodes"].to(self.device)  # [B]
        B = len(prompts)
        
        # ===== Forward: entire batch in one GPU launch =====
        if self.amp_enabled:
            with torch.amp.autocast('cuda'):
                output = self.model(prompts, batch_size=B)
                # Differentiable loss from the loss_head logits
                loss_logits = output["loss_logits"].squeeze(-1)  # [B]
                # MSE between predicted quality scores and target node counts
                loss = nn.functional.mse_loss(loss_logits, target_nodes)
        else:
            output = self.model(prompts, batch_size=B)
            loss_logits = output["loss_logits"].squeeze(-1)
            loss = nn.functional.mse_loss(loss_logits, target_nodes)
        
        # ===== Backward with AMP scaling =====
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return loss.item()


def setup_distributed():
    """Boots the PyTorch Distributed Backend."""
    backend = 'gloo' if os.name == 'nt' else 'nccl'
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend)
        return True
    return False


def run_training_loop(epochs: int = 100, batch_size: int = 16, num_workers: int = 2):
    is_clustered = setup_distributed()
    
    if not is_clustered:
        print("\n==================================================================")
        print("  SNAP-C1 V4 Parallelized Training Pipeline                       ")
        print("==================================================================\n")
        
    trainer = V4DistributedTrainer()
    
    # ============================================================
    #  PARALLELIZATION 4: DataLoader with async prefetching
    # ============================================================
    dataset_path = os.path.join(project_root, "v4_core", "data", "v4_test_dataset.json")
    dataset = V4TrainingDataset(dataset_path)
    
    effective_workers = num_workers if torch.cuda.is_available() else 0
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if effective_workers > 0 else None,
        persistent_workers=True if effective_workers > 0 else False,
    )
    
    logger.info(f"DataLoader: batch_size={batch_size}, workers={effective_workers}, "
                f"batches_per_epoch={len(dataloader)}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            loss_val = trainer.train_step(batch)
            epoch_loss += loss_val
            num_batches += 1
            
        avg_loss = epoch_loss / max(1, num_batches)
        
        # Log progress
        if epoch < 10 or epoch % 10 == 0:
            elapsed = time.time() - start_time
            epochs_per_sec = (epoch + 1) / elapsed
            eta_remaining = (epochs - epoch - 1) / max(epochs_per_sec, 0.001)
            logger.info(
                f"Epoch [{epoch}/{epochs}] | Loss: {avg_loss:.4f} | "
                f"Speed: {epochs_per_sec:.2f} epochs/s | "
                f"Elapsed: {elapsed:.0f}s | ETA: {eta_remaining:.0f}s"
            )
            
    # Save Model Weights
    total_time = time.time() - start_time
    save_path = os.path.join(project_root, "v4_core", "snapshot_v4_hyper_router.pt")
    if trainer.local_rank == 0:
        torch.save(trainer.model.state_dict(), save_path)
        logger.success(f"\nTraining Complete in {total_time:.0f}s ({total_time/60:.1f} min)")
        logger.success(f"V4 Checkpoint saved to: {save_path}")

    if is_clustered:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNAP-C1 V4 Parallelized Trainer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Chunks per batch")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader prefetch workers")
    args = parser.parse_args()
    
    run_training_loop(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.workers)
