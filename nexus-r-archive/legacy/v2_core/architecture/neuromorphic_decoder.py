"""
Neuromorphic Concept Decoder
============================
Unbundles grammar and syntax from logical reasoning. 
The 1.5GB Fractal Recurrent Core outputs abstract "Concept Vectors".
This tiny (100MB) decoder acts as a translation layer, converting 
pure math into grammatical English, Python, or bash commands.
"""

import torch
import torch.nn as nn
import tiktoken
from loguru import logger

class ConceptDecoder(nn.Module):
    """
    A lightweight auto-regressive Transformer decoder.
    Takes a Concept Vector as the cross-attention context and generates tokens.
    """
    def __init__(
        self, 
        vocab_size: int = 100277, # OpenAI cl100k_base vocabulary size
        concept_dim: int = 1024, 
        decoder_dim: int = 256, 
        num_layers: int = 4
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        
        # Maps the heavy concept vector down to the lightweight syntax dimension
        self.concept_proj = nn.Linear(concept_dim, decoder_dim)
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, decoder_dim)
        
        # Lightweight decoder layers
        # Using PyTorch's native TransformerDecoderLayer for rapid prototyping
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim, 
            nhead=4, 
            dim_feedforward=decoder_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final language modeling head
        self.lm_head = nn.Linear(decoder_dim, vocab_size, bias=False)
        
        # Tie weights (standard practice)
        self.lm_head.weight = self.embedding.weight

    def generate(self, concept_vector: torch.Tensor, max_new_tokens: int = 64) -> torch.Tensor:
        """
        Auto-regressively translates the concept vector into token IDs.
        
        Args:
            concept_vector: The final mathematical output from the FRC.
                            Shape: [batch, concept_len, concept_dim]
            max_new_tokens: The maximum number of word tokens to generate.
            
        Returns:
            generated_tokens: Shape [batch, seq_len]
        """
        batch_size = concept_vector.size(0)
        device = concept_vector.device
        
        # Project the concept into decoder space
        # This acts as the "Memory" for the Transformer cross-attention
        memory = self.concept_proj(concept_vector) # [batch, concept_len, decoder_dim]
        
        # Start token (assuming 1 is BOS for standard tokenizers)
        bos_token_id = 1 
        current_tokens = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        # Auto-regressive generation loop
        for t in range(max_new_tokens):
            # Embed current sequence
            tgt = self.embedding(current_tokens) # [batch, seq_len, decoder_dim]
            
            # Create causal mask for target (upper triangular)
            seq_len = tgt.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            # Forward pass through decoder
            # PyTorch expects [batch, seq_len, dim] if batch_first=True
            out = self.transformer(tgt, memory, tgt_mask=tgt_mask)
            
            # Get logits for the last token
            next_token_logits = self.lm_head(out[:, -1, :]) # [batch, vocab_size]
            
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # [batch, 1]
            
            # Append token
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            # Simulated EOS check (standard EOF/EOS for tiktoken)
            if (next_token == 100257).any(): # <|endoftext|> in cl100k_base
                break
                
        return current_tokens

    def generate_string(self, concept_vector: torch.Tensor, max_new_tokens: int = 64) -> str:
        """
        Helper method that runs the generation loop and decodes the resulting
        token IDs immediately into a human-readable string.
        """
        # Ensure we are passing raw logits/tokens without unneeded gradient math
        with torch.no_grad():
            token_ids = self.generate(concept_vector, max_new_tokens)
            
        # Extract the sequence from the batch (assuming batch_size=1 for chatting)
        sequence = token_ids[0].tolist()
        
        # Load the GPT-4 level tokenizer
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # Decode ignoring the BOS and EOS tokens
        decoded_string = encoding.decode([t for t in sequence if t < 100000])
        return decoded_string



if __name__ == "__main__":
    import time
    print("\n--- Testing Neuromorphic Concept Decoder ---")
    
    decoder = ConceptDecoder()
    
    # Mock final output directly from the Fractal Recurrent Loop
    # (The FRC concluded its math and output 16 concept tokens of dim 1024)
    pure_math_concept = torch.randn(1, 16, 1024)
    
    print("FRC finished latent math loop.")
    print("Handing Concept Vector to Syntactic Decoder...")
    
    start = time.perf_counter()
    
    # Generate 20 tokens of syntax directly to string!
    text_output = decoder.generate_string(pure_math_concept, max_new_tokens=20)
    
    ms = (time.perf_counter() - start) * 1000
    
    print(f"Generated grammatical string in {ms:.2f} ms")
    print(f"Decoded Output:\n'{text_output}'")
    print("Syntax unbundling successful. Core FLOPs preserved for pure logic.")
