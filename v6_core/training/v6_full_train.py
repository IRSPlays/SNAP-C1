"""
V6 WHORMHOLE - Full GPU Training (RTX 6000 Ada 48GB)
====================================================
Optimized for maximum model size and performance.

48GB VRAM allows:
- d_model=2048, n_blocks=16 → ~800M params
- batch=16, seq=2048
- fp16, gradient checkpointing
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
    """Setup GPU."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device

def create_model(vocab_size, device, d_model=2048, n_blocks=16):
    """Create large V6 model."""
    from v6_core.architecture.v6_assembly import V6ResonanceModel
    
    model = V6ResonanceModel(
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=16,  # 16 heads for d_model=2048
        window_size=256,
        max_seq_len=2048,
        vocab_size=vocab_size,
        K_hash=8,
        d_hash=256,
        use_skip=True,
    ).to(device)
    
    # Enable gradient checkpointing
    if hasattr(model.resonance, 'enable_gradient_checkpointing'):
        model.resonance.enable_gradient_checkpointing()
    
    return model

def create_optimizer(model, learning_rate=5e-5, weight_decay=0.1):
    """AdamW with separate LR for different param groups."""
    no_decay = ['bias', 'norm', 'ln_']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

def evaluate(model, val_loader, device, vocab_size, max_batches=50):
    """Evaluate perplexity."""
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
            total_loss += result['loss'].item() * token_ids.numel()
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
    num_steps=50000,
    eval_every=1000,
    save_every=5000,
    gradient_accumulation_steps=8,
    max_grad_norm=1.0,
    project_name='v6_whormhole',
):
    """Full training loop."""
    print(f"\n{'='*60}")
    print("V6 WHORMHOLE - Full Training")
    print(f"{'='*60}")
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    print(f"Batch: {train_loader.batch_size} x {gradient_accumulation_steps} = {train_loader.batch_size * gradient_accumulation_steps}")
    print(f"Steps: {num_steps:,}")
    print(f"{'='*60}\n")
    
    model.train()
    scaler = GradScaler()
    
    step = 0
    best_val_loss = float('inf')
    train_losses = []
    start_time = time.time()
    
    checkpoint_dir = f'./checkpoints/{project_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        while step < num_steps:
            for batch in train_loader:
                token_ids = batch['token_ids'].to(device)
                type_ids = batch['type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass with mixed precision
                with autocast():
                    result = model.forward_pretrain(token_ids, type_ids, labels)
                    loss = result['loss'] / gradient_accumulation_steps
                
                # Backward
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
                    eta = (num_steps - step) / rate / 60
                    
                    print(f"Step {step:7d} | "
                          f"Loss: {avg_train:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Rate: {rate:.1f}/s | "
                          f"ETA: {eta:.1f}min")
                
                # Evaluation
                if step % eval_every == 0:
                    val_loss, val_ppl = evaluate(model, val_loader, device, vocab_size)
                    
                    print(f"\n{'='*40}")
                    print(f"EVAL at step {step}")
                    print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'step': step,
                            'model_state': model.state_dict(),
                            'val_loss': val_loss,
                        }, f'{checkpoint_dir}/best.pt')
                        print(f"*** New best model! ***")
                    print(f"{'='*40}\n")
                
                # Checkpoint
                if step % save_every == 0:
                    torch.save({
                        'step': step,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'val_loss': val_loss if 'val_loss' in dir() else None,
                    }, f'{checkpoint_dir}/step{step}.pt')
                    print(f"Checkpoint saved at step {step}")
                
                if step >= num_steps:
                    break
                    
    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at step {step}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete! {step} steps in {elapsed/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    # Save final
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
    }, f'{checkpoint_dir}/final.pt')
    
    return best_val_loss

def main():
    device = setup_gpu()
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # Dataset - Use larger OpenWebText for more diverse data
    print("\nLoading dataset...")
    
    # Try OpenWebText first, fall back to WikiText
    try:
        train_ds = load_dataset('openwebtext', split='train[:100000]')
        print(f"OpenWebText train: {len(train_ds)}")
    except:
        print("Falling back to WikiText-2...")
        train_ds = load_dataset('wikitext', 'wikitext-2-v1', split='train')
        val_ds = load_dataset('wikitext', 'wikitext-2-v1', split='validation')
        test_ds = load_dataset('wikitext', 'wikitext-2-v1', split='test')
    
    # Config for 48GB VRAM
    MAX_SEQ_LEN = 2048
    BATCH_SIZE = 8
    GRAD_ACCUM = 8  # Effective batch = 64
    
    # Tokenize
    print("\nTokenizing...")
    def tokenize(examples):
        texts = [t if t.strip() else ' . ' for t in examples['text']]
        return tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding='max_length')
    
    train_tokenized = train_ds.map(tokenize, batched=True, remove_columns=['text'])
    
    if 'val_ds' not in dir():
        val_ds = load_dataset('wikitext', 'wikitext-2-v1', split='validation')
    
    val_tokenized = val_ds.map(tokenize, batched=True, remove_columns=['text'])
    
    # Dataset class
    class TextDataset:
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
    
    train_loader = DataLoader(
        TextDataset(train_tokenized), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        TextDataset(val_tokenized), 
        batch_size=BATCH_SIZE, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    
    # Model - larger for 48GB
    print("\nCreating V6 model...")
    model = create_model(vocab_size, device, d_model=2048, n_blocks=16)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {params:.0f}M")
    
    # Check VRAM
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM: {allocated:.1f}GB / {reserved:.1f}GB reserved")
    
    # Optimizer & scheduler
    optimizer = create_optimizer(model, learning_rate=5e-5, weight_decay=0.1)
    
    total_steps = 50000
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,
        total_steps=total_steps,
        pct_start=0.1,
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
        num_steps=total_steps,
        eval_every=1000,
        save_every=5000,
        gradient_accumulation_steps=GRAD_ACCUM,
    )
    
    print(f"\nFinal best val loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()
