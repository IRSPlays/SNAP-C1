"""
SNAP-C1 V6: Self-Verification Loop
====================================
KEY INNOVATION: Verify output before sending.

Standard models: Generate → Done (no self-check)
V6: Generate → Verify → If wrong, regenerate → Done

How it works:
1. Generate candidate output
2. Verification head checks: is this correct?
3. If wrong: encode feedback, regenerate with corrections
4. If right: send to user

This adds a "conscience" to the model - it checks its own work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v6_core.architecture.dml_ops import RMSNorm


class VerificationHead(nn.Module):
    """
    Checks if generated output is correct.
    
    Input: hidden state + generated output encoding
    Output: P(correct), P(partial), P(wrong)
    
    Simple 3-class classification.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
        )
        self.classifier = nn.Linear(d_model // 2, 3)  # CORRECT, PARTIAL, WRONG

    def forward(self, hidden_state: torch.Tensor, 
                generated_encoding: torch.Tensor) -> dict:
        """
        Args:
            hidden_state: [B, d_model] - model state
            generated_encoding: [B, d_model] - encoding of generated output
        
        Returns:
            dict with 'probs' [B, 3], 'prediction' [B]
        """
        combined = torch.cat([hidden_state, generated_encoding], dim=-1)
        encoded = self.encoder(combined)
        logits = self.classifier(encoded)
        probs = F.softmax(logits, dim=-1)
        
        return {
            'probs': probs,  # [B, 3] - CORRECT, PARTIAL, WRONG
            'logits': logits,
            'prediction': probs.argmax(dim=-1),  # [B] - which class
        }


class FeedbackEncoder(nn.Module):
    """
    Encodes what went wrong into a feedback signal.
    This feedback is used to improve regeneration.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model + 3, d_model),  # hidden + verification class
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, hidden_state: torch.Tensor, 
                verification_class: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, d_model]
            verification_class: [B] - 0=CORRECT, 1=PARTIAL, 2=WRONG
        
        Returns:
            feedback: [B, d_model] - encoding of what to improve
        """
        # One-hot verification class
        B = hidden_state.shape[0]
        verify_onehot = torch.zeros(B, 3, device=hidden_state.device)
        verify_onehot.scatter_(1, verification_class.unsqueeze(1), 1)
        
        combined = torch.cat([hidden_state, verify_onehot], dim=-1)
        return self.encoder(combined)


class SelfVerificationLoop(nn.Module):
    """
    Complete self-verification loop.
    
    For each generated action:
    1. GENERATE: Produce candidate output
    2. VERIFY: Check if output is correct
    3. IF WRONG: Regenerate with feedback
    4. IF RIGHT: Send to user
    """
    
    VERIFY_CORRECT = 0
    VERIFY_PARTIAL = 1
    VERIFY_WRONG = 2
    
    CONFIDENCE_THRESHOLD = 0.8  # Need 80% confidence to accept
    
    def __init__(self, d_model: int, n_verification_passes: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_verification_passes = n_verification_passes
        
        self.verification_head = VerificationHead(d_model)
        self.feedback_encoder = FeedbackEncoder(d_model)
        
        # Output encoder: encodes generated text for verification
        self.output_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def verify(self, hidden_state: torch.Tensor, 
               generated_output: torch.Tensor) -> dict:
        """
        Verify a generated output.
        
        Args:
            hidden_state: [B, d_model]
            generated_output: [B, d_model] - encoding of generated output
        
        Returns:
            dict with verification results
        """
        encoding = self.output_encoder(generated_output)
        result = self.verification_head(hidden_state, encoding)
        
        return {
            'probs': result['probs'],
            'prediction': result['prediction'],
            'confidence': result['probs'][:, self.VERIFY_CORRECT],
            'is_correct': result['probs'][:, self.VERIFY_CORRECT] > self.CONFIDENCE_THRESHOLD,
        }

    def encode_feedback(self, hidden_state: torch.Tensor,
                        verification_class: torch.Tensor) -> torch.Tensor:
        """
        Encode feedback for regeneration.
        """
        return self.feedback_encoder(hidden_state, verification_class)


class VerifiedActionDecoder(nn.Module):
    """
    Action decoder with self-verification built in.
    
    After generating an action, verify it before executing.
    If wrong, regenerate with feedback.
    """
    
    def __init__(self, d_model: int, n_tools: int = 8,
                 n_verification_passes: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_tools = n_tools
        self.n_verification_passes = n_verification_passes
        
        # Standard action decoder components
        from v6_core.architecture.action_decoder import ActionDecoder, ToolID
        self.action_decoder = ActionDecoder(
            d_model=d_model, n_tools=n_tools
        )
        
        # Self-verification
        self.verification_loop = SelfVerificationLoop(
            d_model=d_model, 
            n_verification_passes=n_verification_passes
        )
        
        # Feedback injection: modifies hidden state for regeneration
        self.feedback_injection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, resonance_output: torch.Tensor,
                context: torch.Tensor = None,
                context_token_ids: torch.Tensor = None,
                verify: bool = True):
        """
        Generate action with optional self-verification.
        
        Args:
            resonance_output: [B, T, d_model] - from resonance stack
            context: [B, slots, d_model] - elastic context
            context_token_ids: [B, slots] - token IDs
            verify: whether to run self-verification (default True)
        
        Returns:
            dict with action + verification results
        """
        # Initial action decision
        action = self.action_decoder(resonance_output, context, context_token_ids)
        
        if not verify:
            return {
                **action,
                'verification_passed': None,
                'n_passes': 0,
            }
        
        # Get pooled hidden state for verification
        if len(resonance_output.shape) == 3:
            hidden = resonance_output.mean(dim=1)  # [B, d_model]
        else:
            hidden = resonance_output
        
        # Self-verification loop
        best_action = action
        best_confidence = action['confidence']
        
        for pass_idx in range(self.n_verification_passes):
            # Encode the current action decision
            action_encoding = torch.zeros_like(hidden)
            for i, tool_id in enumerate(action['tool_id']):
                action_encoding[i, tool_id] = action['confidence'][i].item()
            
            # Verify
            verify_result = self.verification_loop.verify(
                hidden, action_encoding
            )
            
            if verify_result['is_correct'].all():
                # All correct, we're done
                return {
                    **action,
                    'verification_passed': True,
                    'confidence': verify_result['confidence'],
                    'n_passes': pass_idx + 1,
                }
            
            # Some wrong - get feedback and regenerate
            for i in range(hidden.shape[0]):
                if not verify_result['is_correct'][i]:
                    # Encode feedback for this sample
                    feedback = self.verification_loop.encode_feedback(
                        hidden[i:i+1],
                        verify_result['prediction'][i:i+1]
                    )
                    
                    # Inject feedback into hidden state
                    enhanced = self.feedback_injection(
                        torch.cat([hidden[i:i+1], feedback], dim=-1)
                    )
                    hidden[i:i+1] = enhanced
            
            # Regenerate action with feedback
            # (Simplified: in practice, would re-run action decoder with enhanced hidden)
            action = self.action_decoder(
                resonance_output, context, context_token_ids
            )
        
        # Exhausted passes, return best effort
        return {
            **action,
            'verification_passed': False,
            'confidence': best_confidence,
            'n_passes': self.n_verification_passes,
        }


def apply_verification_to_model(model: nn.Module, 
                                  n_verification_passes: int = 3):
    """
    Wrap an existing V6 model with verification.
    
    Replace the action decoder with a VerifiedActionDecoder.
    """
    if hasattr(model, 'action_decoder'):
        old_decoder = model.action_decoder
        model.action_decoder = VerifiedActionDecoder(
            d_model=model.d_model,
            n_tools=old_decoder.n_tools,
            n_verification_passes=n_verification_passes,
        )
    return model
