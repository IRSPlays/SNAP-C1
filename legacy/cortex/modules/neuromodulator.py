"""Deterministic Neuromodulator

Computes 4 global signals from prediction error and memory confidence.
No learnable parameters in V1 — purely functional.

Signals:
    delta (dopamine)       — write gate for hippocampus (surprise/novelty)
    nu (norepinephrine)    — cortex effort / LTC dt modulation (uncertainty)
    sigma (serotonin)      — memory-vs-cortex blend weight (confidence)
    alpha (acetylcholine)  — reserved for V2

Dopamine uses two-component trigger (verified in simulation):
    absolute: clamp((eps - 0.3) / 0.7, 0, 1)   — catches truly wrong predictions
    relative: clamp(z_score, 0, 1)               — catches contextual surprises
    delta = max(absolute, relative)
"""

import torch
import torch.nn as nn


class Neuromodulator(nn.Module):

    def __init__(self, ema_decay: float = 0.99, abs_threshold: float = 0.3):
        super().__init__()
        self.ema_decay = ema_decay
        self.abs_threshold = abs_threshold

        # Running statistics for adaptive thresholding (not learnable)
        self.register_buffer('mu_eps', torch.tensor(0.5))
        self.register_buffer('sigma_eps', torch.tensor(0.1))
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def _update_stats(self, eps_mean: torch.Tensor):
        """Update running mean and std of prediction error."""
        if not self.initialized:
            self.mu_eps.fill_(eps_mean.item())
            self.initialized.fill_(True)
            return

        d = self.ema_decay
        self.mu_eps = d * self.mu_eps + (1 - d) * eps_mean
        # Running std via exponential moving variance
        diff = eps_mean - self.mu_eps
        self.sigma_eps = torch.sqrt(
            d * self.sigma_eps ** 2 + (1 - d) * diff ** 2
        ).clamp(min=1e-6)

    def forward(
        self,
        prediction_error: torch.Tensor,
        memory_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute neuromodulatory signals.

        Args:
            prediction_error: [B, T] or [B] — mean squared PC error
            memory_confidence: [B, T] or [B] — max attention weight from memory read

        Returns:
            dict with keys 'delta', 'nu', 'sigma', 'alpha' — same shape as input
        """
        eps = prediction_error

        # Update running stats with batch mean
        self._update_stats(eps.mean())

        # --- DOPAMINE: surprise → write gate ---
        # Absolute component: error above fixed threshold
        delta_abs = ((eps - self.abs_threshold) / (1.0 - self.abs_threshold + 1e-6)).clamp(0, 1)
        # Relative component: z-score surprise
        z_score = (eps - self.mu_eps) / self.sigma_eps.clamp(min=1e-6)
        delta_rel = z_score.clamp(0, 1)
        # Take the max — either trigger is sufficient
        delta = torch.max(delta_abs, delta_rel)

        # --- NOREPINEPHRINE: uncertainty → cortex effort ---
        nu = (eps * (1.0 - memory_confidence) * 2.0).clamp(0, 1)

        # --- SEROTONIN: confidence → trust memory ---
        # High confidence → high sigma → trust memory
        # sigmoid(10*(0.9-0.5)) ≈ 0.98, sigmoid(10*(0.1-0.5)) ≈ 0.02
        sigma = torch.sigmoid(10.0 * (memory_confidence - 0.5))

        # --- ACETYLCHOLINE: reserved ---
        alpha = torch.zeros_like(eps)

        return {
            'delta': delta,
            'nu': nu,
            'sigma': sigma,
            'alpha': alpha,
        }
