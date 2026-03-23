import gc
import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from loguru import logger
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from v4_core.architecture.v4_assembly import V4HyperAssembly
from v4_core.data.bpe_tokenizer import HybridTokenDecoder
from v4_core.utils.device import get_device

class InstructionDataset(Dataset):
    def __init__(self, json_path: str, tokenizer: HybridTokenDecoder, max_target_len: int = 512):  # 512 to fit CoT <think> blocks
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = "Generate a unified diff to solve this issue:\n" + item["prompt"]
        target_code = item["target_code"]
        
        # Tokenize target string into integers using BPE
        encoded_tokens = self.tokenizer.bpe.encode(target_code)
        
        # Truncate or pad with 0s to max_target_len
        if len(encoded_tokens) > self.max_target_len:
            encoded_tokens = encoded_tokens[:self.max_target_len]
        else:
            encoded_tokens.extend([0] * (self.max_target_len - len(encoded_tokens)))
            
        return {
            "prompt": prompt,
            "target_tokens": torch.tensor(encoded_tokens, dtype=torch.long)
        }

def run_instruction_tuning():
    parser = argparse.ArgumentParser(description="V4 SNAP-C1 Instruction Fine-Tuning")
    parser.add_argument("--dataset", type=str, default="v4_core/data/v4_instruction_dataset.json")
    parser.add_argument("--weights", type=str, required=True, help="Path to base V4 weights (the Physics Engine)")
    parser.add_argument("--output", type=str, default="v4_core/snapshot_v4_instruct.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an interim AST-head checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Epoch to start from (use with --resume to skip completed epochs)")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Booting V4 Instruction Trainer on: {device}")
    
    # 1. Load Pre-trained V4 Base Model
    model = V4HyperAssembly()
    
    logger.info(f"Loading base structural weights from {args.weights}")
    state_dict = torch.load(args.weights, map_location='cpu', weights_only=True)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('._orig_mod.', '.')
        # Discard old untrained AST geometry decoder weights (512 vs 1024-dim mismatch)
        # Discard loss_head: upgraded from scalar [1] to 4-class [4] — let it train fresh
        if "ast_geometry_decoder" not in clean_key and "loss_head" not in clean_key:
            clean_state_dict[clean_key] = v
        
    model.load_state_dict(clean_state_dict, strict=False)

    # Resume from interim AST-head checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from interim checkpoint: {args.resume}")
        resume_sd = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(resume_sd, strict=False)
        logger.info(f"Loaded {len(resume_sd)} AST-head tensors from resume checkpoint")

    model.train()
    
    for name, param in model.named_parameters():
        # Freeze: ODE core, MoE router, AND any embedding weight tables.
        # nn.Embedding backward uses scatter_add_ which DirectML rejects.
        # Embeddings don't change meaning during instruction fine-tuning anyway;
        # all learnable signal is in the SSM matrices, attention, and decoder heads.
        if "logic_core" in name or "ssd_micro_moe" in name or "embedding.weight" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    logger.info("Froze ODE core + MoE router + all embedding tables. Unfroze Compressor SSM, RAG attention, and AST Decoder heads.")
            
    # 2. Setup Data
    tokenizer = HybridTokenDecoder()
    dataset = InstructionDataset(args.dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. Setup Optimizer + LR schedule: 10% linear warmup then cosine decay (fix #4)
    if len(dataloader) == 0:
        logger.error("Training dataset is empty — cannot initialise OneCycleLR (steps_per_epoch=0). Aborting.")
        return

    from torch.optim import AdamW
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    remaining_epochs = args.epochs - args.start_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader),
        epochs=remaining_epochs,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 4. Training Loop
    logger.info(f"Starting Instruction Fine-Tuning -> {args.epochs} Epochs (from epoch {args.start_epoch})")

    for epoch in range(args.start_epoch, args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            prompts = batch["prompt"]
            target_tokens = batch["target_tokens"]  # moved to GPU inside forward_train() via .to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward Pass: Compute Generation Loss via Teacher Forcing
            output = model(prompts, training_mode=True, target_tokens=target_tokens)
            loss = output["generation_loss"]
            
            # Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            
            # Explicitly free memory to avoid OOM on 8GB VRAM
            del loss
            del output
            gc.collect()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] - NLL Loss: {loss_val:.4f}")
                
            if batch_idx > 0 and batch_idx % 15 == 0:
                interim_path = args.output.replace('.pt', f'_ep{epoch}_batch{batch_idx}.pt')
                
                # Extract ONLY the AST Decoder head to avoid System RAM MemoryError
                # The rest of the ODE core is frozen anyway, so we don't need to save it
                ast_head_weights = {k: v.cpu() for k, v in model.state_dict().items() if 'ast_geometry_decoder' in k}
                torch.save(ast_head_weights, interim_path)
                
                logger.success(f"Interim instruction-head checkpoint saved: {interim_path}")
                
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        logger.info(f"=== Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s ===")
        
    # 5. Save the Instruction-Tuned Weights
    torch.save(model.state_dict(), args.output)
    logger.success(f"Final fine-tuned weights saved to {args.output}")

if __name__ == "__main__":
    run_instruction_tuning()
