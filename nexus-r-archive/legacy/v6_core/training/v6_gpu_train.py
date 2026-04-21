"""
V6 WHORMHOLE - GPU Training Script
===================================
Optimized for RTX 6000 Pro (24GB VRAM)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import GPT2Tokenizer
import time
import os
import sys

def setup_gpu():
    """Setup GPU and mixed precision."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device

def create_model(vocab_size, device):
    """Create V6 model for GPU training."""
    from v6_core.architecture.v6_assembly import V6ResonanceModel
    
    # Full-size model for RTX 6000 Pro (24GB VRAM)
    # d_model=1024, n_blocks=8 = ~138M params
    # With fp16 + gradient checkpointing: ~2GB activation memory
    model = V6ResonanceModel(
        d_model=1024,
        n_blocks=8,
        n_heads=8,
        window_size=128,
        max_seq_len=2048,
        vocab_size=vocab_size,
        K_hash=8,
        d_hash=128,
        use_skip=True,
    ).to(device)
    
    # Enable gradient checkpointing to save memory
    if hasattr(model.resonance, 'enable_gradient_checkpointing'):
        model.resonance.enable_gradient_checkpointing()
    
    return model

def create_optimizer(model, learning_rate=1e-4, weight_decay=0.1):
    """Create AdamW optimizer with separate learning rates."""
    no_decay = ['bias', 'LayerNorm.weight', 'norm']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

def save_checkpoint(model, optimizer, scheduler, step, val_loss, path):
    """Save training checkpoint."""
    checkpoint = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def evaluate(model, val_loader, device, vocab_size, max_batches=50):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            
            token_ids = batch['token_ids'].to(device)
            type_ids = batch['type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            result = model.forward_pretrain(token_ids, type_ids, labels)
            loss = result['loss']
            
            total_loss += loss.item() * token_ids.numel()
            total_tokens += token_ids.numel()
    
    model.train()
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    vocab_size,
    num_steps=10000,
    eval_every=500,
    save_every=1000,
    checkpoint_dir='./checkpoints_gpu',
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
):
    """Main training loop."""
    print(f"\n{'='*60}")
    print("V6 WHORMHOLE Training")
    print(f"{'='*60}")
    print(f"Steps: {num_steps}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch: {train_loader.batch_size * gradient_accumulation_steps}")
    print(f"Eval every: {eval_every}")
    print(f"{'='*60}\n")
    
    model.train()
    scaler = GradScaler()
    
    step = 0
    best_val_loss = float('inf')
    train_losses = []
    start_time = time.time()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        while step < num_steps:
            for batch in train_loader:
                # Move to GPU
                token_ids = batch['token_ids'].to(device)
                type_ids = batch['type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward with mixed precision
                with autocast():
                    result = model.forward_pretrain(token_ids, type_ids, labels)
                    loss = result['loss'] / gradient_accumulation_steps
                
                # Backward with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Unscale gradients and clip
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                step += 1
                train_losses.append(loss.item() * gradient_accumulation_steps)
                
                if len(train_losses) > 100:
                    train_losses.pop(0)
                
                # Logging
                if step % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_train = sum(train_losses) / len(train_losses)
                    lr = scheduler.get_last_lr()[0]
                    rate = step / elapsed
                    
                    print(f"Step {step:6d} | "
                          f"Loss: {avg_train:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Rate: {rate:.1f} step/s | "
                          f"Elapsed: {elapsed/60:.1f}min")
                
                # Evaluation
                if step % eval_every == 0:
                    val_loss, val_ppl = evaluate(model, val_loader, device, vocab_size)
                    print(f"\n*** EVAL at step {step} ***")
                    print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, scheduler, step, val_loss,
                                      f"{checkpoint_dir}/v6_best.pt")
                        print(f"*** New best model! ***")
                
                # Checkpointing
                if step % save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, step, val_loss,
                                  f"{checkpoint_dir}/v6_step{step}.pt")
                
                if step >= num_steps:
                    break
                    
    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at step {step}")
    
    # Final save
    print(f"\nTraining complete! {step} steps")
    print(f"Best val loss: {best_val_loss:.4f}")
    save_checkpoint(model, optimizer, scheduler, step, best_val_loss,
                   f"{checkpoint_dir}/v6_final.pt")
    
    return best_val_loss

def main():
    """Main entry point."""
    # Setup
    device = setup_gpu()
    os.makedirs('./checkpoints_gpu', exist_ok=True)
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # Dataset - WikiText-2
    print("\nLoading WikiText-2...")
    train_ds = load_dataset('wikitext', 'wikitext-2-v1', split='train')
    val_ds = load_dataset('wikitext', 'wikitext-2-v1', split='validation')
    test_ds = load_dataset('wikitext', 'wikitext-2-v1', split='test')
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Config
    MAX_SEQ_LEN = 512  # Longer context for better learning
    BATCH_SIZE = 8     # 8 * 512 * 1024 * 4 bytes ~ 16MB per sample in fp16
    
    # Tokenize
    def tokenize(examples):
        texts = [t if t.strip() else ' . ' for t in examples['text']]
        return tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding='max_length')
    
    print("\nTokenizing...")
    train_tokenized = train_ds.map(tokenize, batched=True, remove_columns=['text'])
    val_tokenized = val_ds.map(tokenize, batched=True, remove_columns=['text'])
    test_tokenized = test_ds.map(tokenize, batched=True, remove_columns=['text'])
    
    # Dataset class
    class WikiDataset:
        def __init__(self, tokenized):
            self.ids = tokenized['input_ids']
        
        def __len__(self):
            return len(self.ids)
        
        def __getitem__(self, idx):
            ids = self.ids[idx]
            return {
                'token_ids': torch.tensor(ids, dtype=torch.long),
                'type_ids': torch.zeros(MAX_SEQ_LEN, dtype=torch.long),
                'labels': torch.tensor(ids, dtype=torch.long)
            }
    
    # DataLoaders
    train_loader = DataLoader(WikiDataset(train_tokenized), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(WikiDataset(val_tokenized), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader = DataLoader(WikiDataset(test_tokenized), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    
    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    # Model
    print("\nCreating V6 model...")
    model = create_model(vocab_size, device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {params:.1f}M")
    
    # Optimizer
    optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=0.1)
    
    # Scheduler - Cosine with warmup
    total_steps = 10000
    warmup_steps = 500
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
    )
    
    # Train
    best_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        vocab_size=vocab_size,
        num_steps=10000,
        eval_every=500,
        save_every=1000,
        gradient_accumulation_steps=4,
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load('./checkpoints_gpu/v6_best.pt')
    model.load_state_dict(checkpoint['model_state'])
    
    test_loss, test_ppl = evaluate(model, test_loader, device, vocab_size, max_batches=100)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_ppl:.1f}")
    
    # Save test results
    with open('./checkpoints_gpu/test_results.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Perplexity: {test_ppl:.1f}\n")
        f.write(f"Best Val Loss: {best_loss:.4f}\n")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
