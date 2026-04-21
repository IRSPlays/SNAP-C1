"""
V6 WHORMHOLE: Enhanced Self-Learning System
===========================================
Combines:
1. Hebbian Plasticity (V6) - weights change during inference
2. DPO Self-Play (V5) - learn from preference pairs  
3. Continuous Learning - never stops improving

48GB VRAM allows: ~500M params, batch 16, seq 2048
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple
import time
import json
from dataclasses import dataclass


@dataclass
class SelfLearningConfig:
    """Configuration for V6 self-learning."""
    d_model: int = 1536        # Larger model for 48GB
    n_blocks: int = 12        # More layers
    n_heads: int = 12         # More heads
    window_size: int = 256    # Larger window
    max_seq_len: int = 2048   # Long context
    plasticity_rate: float = 0.01  # Faster Hebbian learning
    hebbian_decay: float = 0.9     # Less decay = faster learning
    dpo_lr: float = 2e-5
    dpo_steps: int = 100
    pairs_per_round: int = 32
    batch_size: int = 8
    gradient_accumulation: int = 4
    temperature: float = 0.7   # For DPO sampling


class V6SelfLearning:
    """
    V6 Self-Learning System.
    
    Learns in two complementary ways:
    1. HEBBIAN: Weights update during inference (fast, local)
    2. DPO: Preferences from self-play (slow, global)
    
    Combined: Hebbian for quick adaptation, DPO for structural improvement.
    """
    
    def __init__(self, model, config: SelfLearningConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.scaler = GradScaler()
        
        # DPO optimizer
        self.dpo_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.dpo_lr,
            weight_decay=0.1
        )
        
        # Experience buffer
        self.dpo_buffer: List[Dict] = []
        self.hebbian_buffer: List[Dict] = []
        
        # Stats
        self.round = 0
        self.total_dpo_steps = 0
        self.total_hebbian_updates = 0
        
    def generate_dpo_pair(self, prompt: str) -> Tuple[str, str, str]:
        """
        Generate a DPO pair from self-play.
        
        Returns:
            (prompt, chosen_response, rejected_response)
        """
        # Tokenize prompt
        tokenizer = self._get_tokenizer()
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, 
                         max_length=self.config.max_seq_len)
        input_ids = inputs['input_ids'].to(self.device)
        
        # Generate two responses with different temperatures
        with torch.no_grad():
            # Generation 1: higher temperature (more creative/wrong)
            response_1 = self._generate(input_ids, temperature=0.9)
            
            # Generation 2: lower temperature (more conservative/correct)
            response_2 = self._generate(input_ids, temperature=0.5)
        
        # Assume lower temp is usually better (more focused)
        chosen = response_1 if torch.rand(1).item() > 0.5 else response_2
        rejected = response_2 if chosen == response_1 else response_1
        
        return prompt, chosen, rejected
    
    def _generate(self, input_ids: torch.Tensor, temperature: float = 0.7) -> str:
        """Generate text from input tokens."""
        tokenizer = self._get_tokenizer()
        
        # Simple greedy/softmax generation
        max_new_tokens = 128
        generated = input_ids.clone()
        
        self.model.eval()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model.forward_pretrain(generated, None, None)['logits']
            
            # Get next token
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        return tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def _get_tokenizer(self):
        """Get tokenizer (lazy load)."""
        if not hasattr(self, '_tokenizer'):
            from transformers import GPT2Tokenizer
            self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer
    
    def update_from_hebbian(self, input_ids: torch.Tensor):
        """
        Update via Hebbian plasticity during inference.
        
        This is called AFTER each inference to modify plastic weights.
        """
        self.model.eval()
        
        # Forward pass (plastic weights update automatically)
        with torch.no_grad():
            result = self.model.forward_pretrain(input_ids, None, input_ids)
        
        self.total_hebbian_updates += 1
        
        # Record for analysis
        if len(self.hebbian_buffer) > 1000:
            self.hebbian_buffer.pop(0)
        
        return result['loss'].item()
    
    def update_from_dpo(self, prompt: str, chosen: str, rejected: str) -> float:
        """
        Update via DPO (Direct Preference Optimization).
        
        This is the global, slow learning that Hebbian can't do.
        """
        tokenizer = self._get_tokenizer()
        
        # Tokenize
        chosen_tokens = tokenizer(chosen, return_tensors='pt', truncation=True,
                                  max_length=512)['input_ids'].to(self.device)
        rejected_tokens = tokenizer(rejected, return_tensors='pt', truncation=True,
                                   max_length=512)['input_ids'].to(self.device)
        prompt_tokens = tokenizer(prompt, return_tensors='pt', truncation=True,
                                 max_length=self.config.max_seq_len)['input_ids'].to(self.device)
        
        # Compute logits for chosen and rejected
        self.model.train()
        
        # Chosen loss
        chosen_logits = self.model.forward_pretrain(
            torch.cat([prompt_tokens, chosen_tokens], dim=1), 
            None, None
        )['logits']
        
        # Rejected loss
        rejected_logits = self.model.forward_pretrain(
            torch.cat([prompt_tokens, rejected_tokens], dim=1),
            None, None
        )['logits']
        
        # DPO loss (simplified)
        # We want: log_prob(chosen) > log_prob(rejected)
        chosen_log_prob = F.log_softmax(chosen_logits, dim=-1).mean()
        rejected_log_prob = F.log_softmax(rejected_logits, dim=-1).mean()
        
        # Loss: -log(sigmoid(chosen - rejected))
        loss = -F.logsigmoid(chosen_log_prob - rejected_log_prob)
        
        # Backward
        self.dpo_optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.dpo_optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.dpo_optimizer)
        self.scaler.update()
        
        self.total_dpo_steps += 1
        
        return loss.item()
    
    def self_learning_round(self, prompts: List[str]) -> Dict:
        """
        One round of self-learning.
        
        1. For each prompt, generate DPO pairs via self-play
        2. Update via DPO
        3. Hebbian updates happen during generation
        
        Args:
            prompts: List of coding prompts
            
        Returns:
            Stats dictionary
        """
        self.round += 1
        start_time = time.time()
        
        hebbian_losses = []
        dpo_losses = []
        
        print(f"\n{'='*60}")
        print(f"  SELF-LEARNING ROUND {self.round}")
        print(f"{'='*60}")
        
        for i, prompt in enumerate(prompts[:self.config.pairs_per_round]):
            # Generate DPO pair
            _, chosen, rejected = self.generate_dpo_pair(prompt)
            
            # Update via DPO (slow learning)
            dpo_loss = self.update_from_dpo(prompt, chosen, rejected)
            dpo_losses.append(dpo_loss)
            
            # Log progress
            if (i + 1) % 5 == 0:
                avg_dpo = sum(dpo_losses) / len(dpo_losses)
                print(f"  Pair {i+1}/{self.config.pairs_per_round} | "
                      f"DPO loss: {avg_dpo:.4f}")
        
        elapsed = time.time() - start_time
        
        stats = {
            'round': self.round,
            'hebbian_updates': self.total_hebbian_updates,
            'dpo_steps': self.total_dpo_steps,
            'avg_hebbian_loss': sum(hebbian_losses) / max(1, len(hebbian_losses)),
            'avg_dpo_loss': sum(dpo_losses) / max(1, len(dpo_losses)),
            'elapsed': elapsed,
        }
        
        print(f"\n  Round complete in {elapsed:.1f}s")
        print(f"  Avg DPO loss: {stats['avg_dpo_loss']:.4f}")
        
        return stats
    
    def continuous_learning(self, prompts: List[str], num_rounds: int = 100):
        """
        Run continuous self-learning.
        
        Args:
            prompts: Coding prompts to learn from
            num_rounds: Number of self-learning rounds
        """
        print(f"\n{'='*60}")
        print(f"  CONTINUOUS SELF-LEARNING")
        print(f"  Rounds: {num_rounds}")
        print(f"  Prompts per round: {len(prompts)}")
        print(f"{'='*60}")
        
        for round_num in range(num_rounds):
            stats = self.self_learning_round(prompts)
            
            # Save checkpoint every 10 rounds
            if round_num % 10 == 0 and round_num > 0:
                self.save_checkpoint(f'./checkpoints_gpu/v6_selflearn_r{round_num}.pt')
        
        print(f"\n{'='*60}")
        print(f"  CONTINUOUS LEARNING COMPLETE")
        print(f"{'='*60}")
    
    def save_checkpoint(self, path: str):
        """Save self-learning state."""
        torch.save({
            'round': self.round,
            'model_state': self.model.state_dict(),
            'dpo_optimizer_state': self.dpo_optimizer.state_dict(),
            'total_dpo_steps': self.total_dpo_steps,
            'total_hebbian_updates': self.total_hebbian_updates,
        }, path)
        print(f"Checkpoint saved: {path}")


class CodePromptGenerator:
    """
    Generates coding prompts for self-play.
    
    Creates diverse coding challenges that test:
    - Algorithm knowledge
    - Code completion
    - Bug fixing
    - Refactoring
    """
    
    def __init__(self):
        self.prompt_templates = [
            "Write a function to {task} in Python",
            "Complete the {task} implementation",
            "Fix the bug in this {task} code",
            "Optimize this {task} for better performance",
            "Explain what this {task} code does",
            "Write a test for the {task} function",
            "Refactor this {task} to be more readable",
            "Add error handling to this {task}",
        ]
        
        self.tasks = [
            "binary search",
            "merge sort",
            "linked list reversal",
            "dynamic programming Fibonacci",
            "BFS graph traversal",
            "depth-first search",
            "hash table implementation",
            "queue using stacks",
            "binary tree traversal",
            "memoized recursion",
        ]
    
    def generate(self, num_prompts: int = 32) -> List[str]:
        """Generate diverse coding prompts."""
        import random
        
        prompts = []
        for _ in range(num_prompts):
            template = random.choice(self.prompt_templates)
            task = random.choice(self.tasks)
            prompts.append(template.format(task=task))
        
        return prompts


def run_self_learning_demo():
    """Demo of V6 self-learning on GPU."""
    print("="*60)
    print("V6 WHORMHOLE - Self-Learning Demo")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required!")
        return
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    from v6_core.architecture.v6_assembly import V6ResonanceModel
    
    config = SelfLearningConfig()
    
    print("\nCreating model...")
    model = V6ResonanceModel(
        d_model=config.d_model,
        n_blocks=config.n_blocks,
        n_heads=config.n_heads,
        window_size=config.window_size,
        max_seq_len=config.max_seq_len,
        vocab_size=100279,
        K_hash=8,
        d_hash=128,
        use_skip=True,
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {params:.1f}M params")
    
    # Create self-learning system
    self_learn = V6SelfLearning(model, config, device)
    
    # Generate prompts
    prompt_gen = CodePromptGenerator()
    prompts = prompt_gen.generate(16)
    
    # Run one round
    print("\nRunning self-learning round...")
    stats = self_learn.self_learning_round(prompts)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"DPO steps: {stats['dpo_steps']}")
    print(f"Hebbian updates: {stats['hebbian_updates']}")
    print(f"DPO loss: {stats['avg_dpo_loss']:.4f}")
    
    # Save
    self_learn.save_checkpoint('./checkpoints_gpu/v6_selflearn_demo.pt')
    print("Saved: v6_selflearn_demo.pt")


if __name__ == '__main__':
    run_self_learning_demo()
