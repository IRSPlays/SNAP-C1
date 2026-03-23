"""
SNAP-C1 Embedding Model Wrapper
=================================
Lightweight embedding model for memory retrieval.
Uses sentence-transformers with a small, fast model.

Default: all-MiniLM-L6-v2 (22M params, 384 dimensions)
Fast enough to run on CPU alongside the main model on GPU.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Using ChromaDB default embeddings.")


class EmbeddingModel:
    """Wrapper for the embedding model used in memory retrieval."""
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        """Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
            device: Device to run on (default: cpu, to leave GPU for main model)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.model = None
        
        if ST_AVAILABLE:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model ready. Dimension: {self.dimension}")
        else:
            logger.warning("sentence-transformers not available. Embedding model disabled.")
            self.dimension = 384  # Default dimension
    
    def encode(self, texts: list[str] | str, normalize: bool = True) -> list[list[float]]:
        """Encode text(s) into embedding vectors.
        
        Args:
            texts: Single string or list of strings
            normalize: L2-normalize embeddings (recommended for cosine similarity)
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            logger.warning("Embedding model not available. Returning empty embeddings.")
            if isinstance(texts, str):
                return [[0.0] * self.dimension]
            return [[0.0] * self.dimension] * len(texts)
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        
        return embeddings.tolist()
    
    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.
        
        Returns:
            Similarity score between -1.0 and 1.0
        """
        if self.model is None:
            return 0.0
        
        embeddings = self.encode([text_a, text_b])
        
        # Cosine similarity (already normalized)
        import numpy as np
        a = np.array(embeddings[0])
        b = np.array(embeddings[1])
        return float(np.dot(a, b))
