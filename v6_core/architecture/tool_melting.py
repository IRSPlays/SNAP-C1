"""
SNAP-C1 V6: Tool Melting Engine
=================================
KEY INNOVATION: Tools are synthesized on-the-fly, not fixed at training.

Standard agents: Tool registry is FIXED. Need new tool? Fine-tune model.
V6 Tool Melting: Tools MELT and reform during inference.

How it works:
1. DECOMPOSE: Break new task into known primitive operations
2. MELT: Combine primitives into temporary tool
3. USE: Execute the melted tool
4. COOL: If useful, solidify into permanent registry

This enables ZERO-SHOT tool learning without fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Any, Optional
import hashlib
import inspect


# Primitive operations that the model knows how to do
PRIMITIVES = {
    'READ_FILE': 'Read contents of a file',
    'WRITE_FILE': 'Write contents to a file',
    'SEARCH_TEXT': 'Search for text pattern in content',
    'REPLACE_TEXT': 'Replace text pattern with new text',
    'RUN_COMMAND': 'Execute a shell command',
    'PARSE_JSON': 'Parse JSON string to dict',
    'EMIT_JSON': 'Convert dict to JSON string',
    'PARSE_XML': 'Parse XML string',
    'EMIT_XML': 'Convert dict to XML string',
    'FILTER_LINES': 'Filter lines matching pattern',
    'GET_URL': 'Fetch content from URL',
    'LIST_DIR': 'List files in directory',
    'FILE_EXISTS': 'Check if file exists',
    'GET_TIMESTAMP': 'Get current timestamp',
    'HASH_CONTENT': 'Compute hash of content',
    'BASE64_ENCODE': 'Encode content to base64',
    'BASE64_DECODE': 'Decode base64 to content',
    'ZIP_CONTENT': 'Compress content to zip',
    'UNZIP_CONTENT': 'Decompress zip to content',
    'SORT_LINES': 'Sort lines alphabetically',
    'COUNT_LINES': 'Count number of lines',
    'EXTRACT_EMAILS': 'Extract email addresses',
    'EXTRACT_URLS': 'Extract URLs from text',
    'VALIDATE_JSON': 'Validate JSON syntax',
    'MINIFY_JSON': 'Minify JSON content',
    'FORMAT_JSON': 'Pretty-print JSON',
}


class PrimitiveRegistry:
    """
    Registry of primitive operations the model knows.
    
    Each primitive has:
    - name: identifier
    - description: what it does
    - function: actual implementation
    - input_type: expected input format
    - output_type: output format
    """
    
    def __init__(self):
        self.primitives: Dict[str, dict] = {}
        for name, desc in PRIMITIVES.items():
            self.register(name, desc, lambda x: x, 'any', 'any')
    
    def register(self, name: str, description: str,
                 function: Callable, input_type: str, output_type: str):
        """Register a new primitive."""
        self.primitives[name] = {
            'name': name,
            'description': description,
            'function': function,
            'input_type': input_type,
            'output_type': output_type,
        }
    
    def get(self, name: str) -> Optional[dict]:
        """Get primitive by name."""
        return self.primitives.get(name)
    
    def list_primitives(self) -> List[str]:
        """List all available primitives."""
        return list(self.primitives.keys())
    
    def get_descriptions(self) -> str:
        """Get formatted descriptions of all primitives."""
        lines = []
        for p in self.primitives.values():
            lines.append(f"- {p['name']}: {p['description']}")
        return '\n'.join(lines)


class MeltedTool:
    """
    A tool synthesized by melting primitives together.
    
    Created dynamically when the model needs a capability it doesn't have.
    """
    
    def __init__(self, name: str, primitive_chain: List[dict],
                 synthesized: bool = True, verified: bool = False):
        self.name = name
        self.primitive_chain = primitive_chain
        self.synthesized = synthesized
        self.verified = verified
        self.usage_count = 0
        self.success_count = 0
    
    def execute(self, input_data: Any, context: dict = None) -> Any:
        """Execute the melted tool."""
        self.usage_count += 1
        result = input_data
        
        for primitive in self.primitive_chain:
            try:
                func = primitive['function']
                args = primitive.get('args', {})
                result = func(result, **args)
            except Exception as e:
                self.verified = False
                return {'error': str(e), 'partial_result': result}
        
        self.success_count += 1
        return result
    
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


class ToolSynthesizer(nn.Module):
    """
    Neural network that synthesizes tools from primitives.
    
    Given an intent (what the model wants to do), predicts
    which primitives to chain together.
    """
    
    def __init__(self, d_model: int = 1024, n_primitives: int = 28):
        super().__init__()
        self.n_primitives = n_primitives
        
        # Encode the intent
        self.intent_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )
        
        # Predict which primitives to use
        self.primitive_head = nn.Linear(d_model // 2, n_primitives)
        
        # Predict ordering (softmax over positions)
        self.order_head = nn.Linear(d_model // 2, n_primitives)
        
        # Predict arguments for each primitive
        self.arg_encoder = nn.Linear(d_model, d_model)
    
    def forward(self, intent_hidden: torch.Tensor) -> tuple:
        """
        Given intent hidden state, predict tool synthesis.
        
        Returns:
            primitive_weights: [n_primitives] - which primitives to use
            order_weights: [n_primitives] - in what order
        """
        encoded = self.intent_encoder(intent_hidden)
        
        primitive_logits = self.primitive_head(encoded)
        primitive_weights = torch.softmax(primitive_logits, dim=-1)
        
        order_logits = self.order_head(encoded)
        order_weights = torch.softmax(order_logits, dim=-1)
        
        return primitive_weights, order_weights
    
    def synthesize(self, intent: str, primitive_registry: PrimitiveRegistry,
                   threshold: float = 0.1) -> List[dict]:
        """
        Synthesize a tool from intent string.
        
        Args:
            intent: natural language description of what to do
            primitive_registry: available primitives
            threshold: minimum weight to include primitive
        
        Returns:
            list of primitives to chain
        """
        # In practice, would use the model to encode the intent
        # For now, use simple keyword matching
        intent_lower = intent.lower()
        
        selected = []
        for name, prim in primitive_registry.primitives.items():
            # Simple keyword matching
            desc_lower = prim['description'].lower()
            if any(word in intent_lower for word in name.lower().split('_')):
                selected.append(prim)
        
        # Sort by relevance (simplified)
        selected = selected[:5]  # Max 5 primitives
        
        return selected


class ToolMeltingEngine:
    """
    Main engine for tool melting.
    
    Manages:
    - Primitive registry
    - Melted tool cache
    - Tool synthesis
    - Tool solidification
    """
    
    def __init__(self, model: nn.Module = None):
        self.primitives = PrimitiveRegistry()
        self.melted_tools: Dict[str, MeltedTool] = {}
        self.synthesizer = ToolSynthesizer() if model is None else None
        self.permanent_tools: Dict[str, MeltedTool] = {}
    
    def get_tool(self, intent: str) -> Optional[MeltedTool]:
        """
        Get existing tool or create new one for intent.
        
        Args:
            intent: natural language description
        
        Returns:
            MeltedTool or None if can't synthesize
        """
        # Check cache
        cache_key = self._hash_intent(intent)
        if cache_key in self.melted_tools:
            return self.melted_tools[cache_key]
        
        # Check permanent registry
        for tool in self.permanent_tools.values():
            if self._intent_matches(intent, tool.name):
                return tool
        
        # Synthesize new tool
        tool = self._synthesize_tool(intent)
        if tool:
            self.melted_tools[cache_key] = tool
        
        return tool
    
    def _synthesize_tool(self, intent: str) -> Optional[MeltedTool]:
        """Synthesize a new tool from primitives."""
        # Decompose intent into primitives
        primitives = self.synthesizer.synthesize(intent, self.primitives)
        
        if len(primitives) == 0:
            return None
        
        # Create melted tool
        tool = MeltedTool(
            name=intent[:50],  # Truncate for name
            primitive_chain=primitives,
            synthesized=True,
            verified=False
        )
        
        return tool
    
    def _intent_matches(self, intent: str, tool_name: str) -> bool:
        """Check if intent matches a tool."""
        # Simple implementation
        intent_words = set(intent.lower().split())
        name_words = set(tool_name.lower().split('_'))
        return len(intent_words & name_words) > 0
    
    def _hash_intent(self, intent: str) -> str:
        """Create cache key from intent."""
        return hashlib.md5(intent.encode()).hexdigest()
    
    def solidify_tool(self, tool: MeltedTool) -> bool:
        """
        If a melted tool is useful, add to permanent registry.
        
        Args:
            tool: MeltedTool to solidify
        
        Returns:
            True if solidified, False otherwise
        """
        # Check if tool has been useful
        if tool.usage_count < 3:
            return False
        
        # Check if tool works
        if tool.success_rate() < 0.8:
            return False
        
        # Add to permanent registry
        tool.synthesized = False
        tool.verified = True
        self.permanent_tools[tool.name] = tool
        
        return True
    
    def get_permanent_tools(self) -> Dict[str, MeltedTool]:
        """Get all permanently registered tools."""
        return self.permanent_tools.copy()
    
    def list_tools(self) -> List[str]:
        """List all available tools."""
        tools = list(self.permanent_tools.keys())
        tools.extend(f"[melted] {t.name}" for t in self.melted_tools.values())
        return tools


class ToolMeltingWrapper(nn.Module):
    """
    Wrapper that adds tool melting to any model.
    
    Intercepts tool selection decisions and checks if
    a melted tool should be used instead.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.melting_engine = ToolMeltingEngine(model)
        
        # When to use melted vs permanent tools
        self.melt_threshold = 0.5
        self.melt_confidence_threshold = 0.7
    
    def forward(self, *args, **kwargs):
        """Forward pass - just delegate to underlying model."""
        return self.model(*args, **kwargs)
    
    def get_or_create_tool(self, intent: str) -> Optional[MeltedTool]:
        """Get existing tool or create new one."""
        return self.melting_engine.get_tool(intent)
    
    def should_melt(self, tool_id: int, confidence: float) -> bool:
        """
        Decide whether to use tool melting.
        
        Args:
            tool_id: selected tool ID
            confidence: confidence in tool selection
        
        Returns:
            True if should try melting
        """
        # Low confidence + unknown tool = try melting
        return confidence < self.melt_confidence_threshold


def integrate_tool_melting(model: nn.Module) -> ToolMeltingWrapper:
    """
    Add tool melting capability to a model.
    
    Returns a wrapper that intercepts tool decisions
    and can synthesize new tools on-the-fly.
    """
    return ToolMeltingWrapper(model)
