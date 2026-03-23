"""
V6 WHORMHOLE - Resume Training from Checkpoint
===============================================
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import GPT2Tokenizer
import time
import os

def setup_gpu():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        exit(1)
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device

def create_model(vocab_size, device):
    from v6_core.architecture.v6_assembly import V6ResonanceModel
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
    return model

def main():
    device = setup_gpu()
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    # Dataset
    print("Loading WikiText-2...")
    train_ds = load_dataset('wikitext', 'wikitext-2-v1', split='train')
    val_ds = load_dataset('wikitext', 'wikitext-2-v1', split='validation')
    
    MAX_SEQ_LEN = 512
    BATCH_SIZE = 8
    
    def tokenize(examples):
        texts = [t if t.strip() else ' . ' for t in examples['text']]
        return tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding='max_length')
    
    train_tokenized = train_ds.map(tokenize, batched=True, remove_columns=['text'])
    val_tokenized = val_ds.map(tokenize, batched=True, remove_columns=['text'])
    
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
    
    train_loader = DataLoader(WikiDataset(train_tokenized), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(WikiDataset(val_tokenized), batch_size=BATCH_SIZE, num_workers=4)
    
    # Model
    print("Creating model...")
    model = create_model(vocab_size, device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {params:.1f}M params")
    
    # Load checkpoint
    checkpoint_path = './checkpoints_gpu/v6_best.pt'
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    start_step = checkpoint['step']
    best_val_loss = checkpoint['val_loss']
    print(f"Resuming from step {start_step}, best val loss: {best_val_loss:.4f}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    # Continue with lower LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = 5e-5  # Lower LR for fine-tuning
    
    total_steps = 20000
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,
        total_steps=total_steps,
        pct_start=0.1,
    )
    
    print(f"\nContinuing training to step {total_steps}...")
    
    # Training loop (simplified)
    model.train()
    scaler = GradScaler()
    step = start_step
    train_losses = []
    
    while step < total_steps:
        for batch in train_loader:
            token_ids = batch['token_ids'].to(device)
            type_ids = batch['type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                result = model.forward_pretrain(token_ids, type_ids, labels)
                loss = result['loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            step += 1
            train_losses.append(loss.item())
            if len(train_losses) > 100: train_losses.pop(0)
            
            if step % 50 == 0:
                avg = sum(train_losses) / len(train_losses)
                print(f"Step {step:6d} | Loss: {avg:.4f}")
            
            if step % 1000 == 0:
                # Save checkpoint
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, f'./checkpoints_gpu/v6_continued_{step}.pt')
                print(f"Checkpoint saved at step {step}")
            
            if step >= total_steps:
                break
    
    print(f"\nTraining complete! Final step: {step}")

if __name__ == '__main__':
    main()
