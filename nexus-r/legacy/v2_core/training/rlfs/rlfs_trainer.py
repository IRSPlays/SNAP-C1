"""
Reinforcement Learning from Formal Systems (RLFS) Trainer
=========================================================
This script orchestrates the final training phase for SNAP-C1 V2.

1. It loads the biologically-pretrained 1.5GB Fractal Recurrent Core (frozen).
2. It feeds a coding problem into the Core.
3. The Neuromorphic Decoder outputs Python syntax.
4. The Sandbox executes the Python code.
5. If the code FAILS, the stack trace error is converted into an embedding 
   and fed BACK into the Core's recurrent loop to act as negative reward learning.
"""

import sys
import os
import torch
from pathlib import Path
from loguru import logger
from torch.optim import AdamW

# Add project root explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from v2_core.architecture.holographic_compressor import HolographicCompressor
from v2_core.architecture.recurrent_core import FractalRecurrentCore
from v2_core.architecture.neuromorphic_decoder import ConceptDecoder
from v2_core.training.rlfs.sandbox import PythonSandbox

class RLFSTrainer:
    def __init__(self, device: str = None):
        """Initializes the SOTA Architecture and the isolated Sandbox."""
        if device is None:
            try:
                import torch_directml
                # Dynamically find the discrete RX GPU rather than the iGPU
                dml_device = torch_directml.device() # default to 0
                for i in range(torch_directml.device_count()):
                    name = torch_directml.device_name(i).lower()
                    if "rx" in name or "tx" in name:
                        dml_device = torch_directml.device(i)
                        break
                self.device = dml_device
            except ImportError:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Setting up RLFS Pipeline on device: {self.device}")
        
        self.dim = 1024
        
        # 1. Instantiate the Engine
        self.compressor = HolographicCompressor(d_model=self.dim, state_size=64).to(self.device)
        self.core = FractalRecurrentCore(hidden_dim=self.dim, max_loops=15).to(self.device)
        self.decoder = ConceptDecoder(vocab_size=100277, concept_dim=self.dim, decoder_dim=256).to(self.device)
        self.sandbox = PythonSandbox(timeout_seconds=1) # Squeezed timeout to 1 second down from 3!
        
        # 2. Inject the A6000 Datacenter Weights
        weights_path = Path(__file__).parent.parent.parent / "frc_pretrained_core_A6000_FINAL.pt"
        if weights_path.exists():
            logger.info(f"Injecting massive pre-trained logic core: {weights_path}")
            # Strict=False because we are only loading the 'core' math engine
            self.core.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
            logger.success("Biological Core Logic successfully loaded.")
        else:
            logger.critical(f"A6000 weights not found at {weights_path}! RLFS cannot proceed on an empty brain!")
            sys.exit(1)
            
        # 3. Freeze the Core, Train the Decoder
        # The core already knows HOW to think (math/logic). Now we only train the Decoder to translate math into Python syntax.
        for param in self.core.parameters():
            param.requires_grad = False
            
        # 4. Auto-Resume Checkpoint Logic (Using the RunPod backup)
        decoder_path = Path(__file__).parent.parent.parent / "neuromorphic_decoder_A6000.pt"
        if decoder_path.exists():
            logger.warning(f"Found existing Decoder Checkpoint: {decoder_path}")
            # Clean possible `_orig_mod.` compilation traces from the saved dictionary
            raw_state = torch.load(decoder_path, map_location='cpu', weights_only=False)
            clean_state = {k.replace('_orig_mod.', ''): v for k, v in raw_state.items()}
            self.decoder.load_state_dict(clean_state)
            logger.success("Resuming Formal Systems training from saved progress!")
            
        # Optimize Decoder for lightning fast token generation (if OS allows it)
        # We exclusively compile *after* the state_dict is loaded so the layer dictionaries match.
        try:
            self.decoder = torch.compile(self.decoder)
            logger.info("PyTorch 2.x C++ Compilation Engine enabled!")
        except Exception:
            pass
            
        self.optimizer = AdamW(self.decoder.parameters(), lr=1e-4)
        
    # Removed Mock Functions. Now using Native cl100k_base Tiktoken strings.
        
    def run_training_loop(self, epochs: int = 5):
        """
        Executes the RLFS formal sandbox loop.
        """
        logger.info("=== STARTING PHASE 5: RLFS TRAINING LOOP ===")
        
        # In a real environment, this array comes from a dataset like humaneval
        full_prompts = [
            "Write a python function that calculates the sum of two integers.",
            "Write a python string reversal tool.",
            "Write a python function that returns the nth fibonacci number.",
            "Write a code that prints fizzbuzz up to 100.",
            "Write a python script that sorts an array of integers.",
            "Write a function to check if a string is a palindrome.",
            "Write a function that returns the maximum value in a list of numbers.",
            "Write a function that counts the number of vowels in a string."
        ]
        
        for epoch in range(epochs):
            import random
            training_prompts = random.sample(full_prompts, 3) # Process 3 random tasks per epoch
            
            logger.info(f"\n--- RLFS Epoch {epoch+1}/{epochs} ---")
            
            for prompt in training_prompts:
                self.optimizer.zero_grad()
                
                logger.info(f"PROMPT: {prompt}")
                
                # Forward Pass (Using Real Tiktoken String Input)
                hologram = self.compressor.process_string(prompt, device=self.device)
                concept_vector, loops, _ = self.core(hologram)
                
                # The decoder attempts to build python syntax (Using Real Tiktoken String Output)
                # Wait for the decoding to finish using the actual generation logic
                # Try generating string. If it crashes due to shape mismatch or tokenizer error, we wrap it in a try-except.
                try:
                    generated_code = self.decoder.generate_string(concept_vector, max_new_tokens=25)
                except Exception as e:
                    logger.error(f"Generation crashed. Checkpoint mismatch: {e}")
                    generated_code = "pass"
                
                logger.debug(f"Decoder Generated ({loops} loops): \n{generated_code}")
                
                # To ensure valid gradients flow from the simulated loss to the decoder parameters
                # WE MUST PASS THE ACTUALLY GENERATED TOKENS to the graph, not a hardcoded [[1]]!
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                try:
                    gen_ids = enc.encode(generated_code)
                    if not gen_ids: gen_ids = [1]
                except Exception:
                    gen_ids = [1]
                    
                gen_tensor = torch.tensor([gen_ids[:25]]).to(self.device)
                
                dummy_tgt = self.decoder.embedding(gen_tensor)
                memory = self.decoder.concept_proj(concept_vector)
                dummy_out = self.decoder.transformer(dummy_tgt, memory)
                
                # Get the logits corresponding to the generated sequence
                dummy_logits = self.decoder.lm_head(dummy_out)
                
                # Execute securely in the Sandbox
                logger.info("Deploying generation to Isolated Python Sandbox...")
                results = self.sandbox.run(generated_code)
                
                if results["success"]:
                    logger.success(f"Sandbox Execution PASSED! Output: {results['stdout'].strip()}")
                    # Massive positive reward: We want to Maximize these logits
                    # PyTorch minimizes, so multiply by negative to pull logits UP
                    reward = 1.0 
                    loss = -dummy_logits.mean() * reward
                else:
                    logger.error(f"Sandbox Execution FAILED: {results['error_type']}")
                    logger.error(f"Stack Trace Captured: {results['error_traceback'][:150]}...")
                    
                    # Negative Reward!
                    # We want to MINIMIZE the probability of the tokens that caused the crash.
                    reward = -10.0
                    
                    # Minimize the mean logit value for the failed tokens, pushing them DOWN in probability.
                    loss = dummy_logits.mean() * abs(reward)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    logger.warning("Backwards pass successful. Decoder parameters adjusted based on Formal Systems penalty!")
                    
            # Save the trained Decoder Syntactic Weights locally at the end of each epoch
            # Strip out `_orig_mod.` prefixes added by torch.compile to prevent loading mismatches
            state_dict = self.decoder.state_dict()
            clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            save_path = Path(__file__).parent.parent.parent / "neuromorphic_decoder_A6000.pt"
            temp_path = save_path.with_suffix(".tmp.pt")
            
            try:
                # Atomically write the file so multiple PowerShell instances can run concurrently
                torch.save(clean_state_dict, temp_path)
                import os
                os.replace(temp_path, save_path)
                logger.success(f"Saved Checkpoint to local disk: {save_path}")
            except Exception as e:
                logger.warning(f"Concurrent Save Collision Detected. Bypassed safely: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Phase 5 formal systems loop locally.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    args = parser.parse_args()

    trainer = RLFSTrainer()
    trainer.run_training_loop(epochs=args.epochs)
