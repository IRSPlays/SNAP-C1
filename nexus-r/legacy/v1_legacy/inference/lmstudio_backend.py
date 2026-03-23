"""
LM Studio Backend for SNAP-C1
=============================
Provides an OpenAI-compatible interface to communicate with
LM Studio's local server (default: http://localhost:1234/v1).
Since we are using Vulkan for AMD GPU acceleration via llama.cpp,
this replaces the direct HuggingFace model loading.
"""

from openai import OpenAI
from loguru import logger


class LMStudioBackend:
    """Wrapper for LM Studio's local OpenAI-compatible API."""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "local-model"):
        """
        Initialize the LM Studio backend.
        
        Args:
            base_url: The URL of the LM Studio local server.
            model: The generic model identifier (can be anything as LM Studio uses the currently loaded model).
        """
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model
        
        logger.info(f"Initialized LM Studio Backend at {base_url}")
        
    def check_connection(self) -> bool:
        """Ping the local server to verify it's running."""
        try:
            # List available models
            models = self.client.models.list()
            logger.info(f"Connected to LM Studio. Available models: {[m.id for m in models.data]}")
            
            if models.data:
                # Use the exact ID of the loaded model
                self.model = models.data[0].id
                
            return True
        except Exception as e:
            logger.error(f"Failed to connect to LM Studio at {self.client.base_url}: {e}")
            logger.warning("Make sure LM Studio is open, a model is loaded, and the Local Server is running!")
            return False

    def generate(self, messages: list[dict], **kwargs) -> str:
        """
        Generate a response using the LM Studio API.
        
        Args:
            messages: List of message dicts (e.g., [{"role": "user", "content": "..."}])
            kwargs: Generation configs (temperature, max_tokens, etc.)
            
        Returns:
            The generated response string.
        """
        # Map our internal kwargs (like max_new_tokens) to OpenAI format
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        
        # Mapping inference parameters
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        if "max_new_tokens" in kwargs:
            api_kwargs["max_tokens"] = kwargs["max_new_tokens"]
        if "top_p" in kwargs:
            api_kwargs["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            api_kwargs["stop"] = kwargs["stop"]
            
        # Call LM Studio API
        try:
            response = self.client.chat.completions.create(**api_kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LM Studio API generation failed: {e}")
            # Ensure the user knows they need the server running
            if "Connection" in str(e) or "connect" in str(e).lower():
                logger.error("Is LM Studio running and the Local Server started?")
            return "Error generating response from local server."
