"""
SNAP-C1 MoLoRA Router (Mixture of LoRA Experts)
==================================================
Lightweight query classifier that routes incoming queries to the optimal
LoRA adapter(s) at inference time — avoiding catastrophic interference
from weight merging.

Architecture:
    1. Keyword-based fast path (zero overhead, catches obvious cases)
    2. TF-IDF + cosine similarity fallback (lightweight, no GPU needed)
    3. Multi-adapter blending for hybrid queries

This is a key SNAP-C1 innovation: instead of merging LoRA adapters into
base weights (which causes destructive interference), we dynamically
route each query to the right expert adapter.
"""

import re
import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    """Result of routing a query to adapter(s)."""
    primary: str                          # Primary adapter name
    scores: dict[str, float]              # Confidence scores per adapter
    reasoning: str                        # Why this routing was chosen
    multi_adapter: bool = False           # Whether to blend multiple adapters
    secondary: str | None = None          # Secondary adapter (for blending)
    blend_weights: dict[str, float] | None = None  # Weights for multi-adapter


class MoLoRARouter:
    """Routes queries to the optimal LoRA adapter(s).
    
    Uses a tiered approach:
        Tier 1: Keyword matching (instant, deterministic)
        Tier 2: TF-IDF similarity against capability profiles (fast, no GPU)
    
    Adapters:
        - team_thinking: Multi-perspective analysis, design decisions, trade-offs
        - self_correction: Code review, bug fixing, error correction
        - tool_use: File operations, code execution, search, shell commands
        - base: No adapter (general queries)
    """
    
    # ---- Tier 1: Keyword patterns ----
    # Strong signals that definitively indicate a capability
    KEYWORD_RULES: dict[str, list[re.Pattern]] = {
        "team_thinking": [
            re.compile(r"\b(architect|critic|researcher|implementer|synthesizer)\b", re.I),
            re.compile(r"\bperspective[s]?\b", re.I),
            re.compile(r"\b(pros?\s+and\s+cons?|trade[\s-]?offs?|tradeoffs?|weigh\s+(the\s+)?options?)\b", re.I),
            re.compile(r"\b(should\s+i|which\s+(is\s+)?better|compare|versus|vs\.?)\b", re.I),
            re.compile(r"\b(design\s+decision|architecture|approach)\b", re.I),
            re.compile(r"\b(analyze|evaluate|assess|debate)\b", re.I),
            re.compile(r"\b(think\s+through|reason\s+about|consider)\b", re.I),
            re.compile(r"\b(monolith|microservice|strategy|methodology)\b", re.I),
        ],
        "self_correction": [
            re.compile(r"\b(review|correct|fix|debug|bug|error|mistake)\b", re.I),
            re.compile(r"\b(previous\s+attempt|wrong|incorrect|broken)\b", re.I),
            re.compile(r"\b(improve|refactor|optimize)\s+(this|the|my)\b", re.I),
            re.compile(r"\b(what['\u2019]?s\s+wrong|find\s+(the\s+)?(bug|error|issue))\b", re.I),
            re.compile(r"\b(self[\s-]?correct|validate|verify)\b", re.I),
            re.compile(r"<review>|<fix>|<validate>", re.I),
            re.compile(r"\bprevious\s+attempt\b", re.I),
            re.compile(r"\b(code\s+review|pull\s+request\s+review)\b", re.I),
        ],
        "tool_use": [
            re.compile(r"\b(read|show|open|cat|display)\s+(the\s+)?(file|contents?)\b", re.I),
            re.compile(r"\b(write|create|save|edit)\s+(a\s+)?(file|script)\b", re.I),
            re.compile(r"\b(run|execute|shell|command|terminal|bash)\b", re.I),
            re.compile(r"\b(search|find|grep|look\s+for)\s+(in\s+|for\s+|all\s+)?(files?|code|project)\b", re.I),
            re.compile(r"\bfiles?\s+(containing|with|that\s+(have|contain))\b", re.I),
            re.compile(r"\b(list|ls|dir)\s+(files?|directory|folder)\b", re.I),
            re.compile(r"<tool_call>", re.I),
            re.compile(r"\.(py|js|ts|yaml|json|txt|md|sh|cfg|ini|toml)\b"),
            re.compile(r"\b(install|pip|npm|git)\s", re.I),
        ],
    }
    
    # Negative patterns — if these appear, reduce confidence for the adapter
    NEGATIVE_RULES: dict[str, list[re.Pattern]] = {
        "team_thinking": [
            re.compile(r"\b(just|simply|quickly)\b", re.I),  # Simple queries don't need deep analysis
        ],
        "self_correction": [
            re.compile(r"\bwrite\s+(a|an|the)\s+new\b", re.I),  # Writing new code, not fixing
        ],
        "tool_use": [
            re.compile(r"\bexplain\b", re.I),  # Explaining, not executing
        ],
    }
    
    # ---- Tier 2: TF-IDF capability profiles ----
    # Representative terms for each capability (manually curated from training data)
    CAPABILITY_PROFILES: dict[str, list[str]] = {
        "team_thinking": [
            "architecture", "design", "tradeoff", "perspective", "analyze",
            "approach", "strategy", "decision", "compare", "evaluate",
            "consider", "monolithic", "microservices", "scalability", "pattern",
            "framework", "methodology", "requirements", "stakeholder", "consensus",
            "debate", "system", "infrastructure", "deployment", "migration",
            "database", "api", "frontend", "backend", "cloud",
            "performance", "security", "maintainability", "complexity", "coupling",
        ],
        "self_correction": [
            "bug", "fix", "error", "review", "correct", "debug",
            "wrong", "broken", "mistake", "refactor", "improve",
            "optimize", "validate", "verify", "test", "edge",
            "case", "exception", "crash", "undefined", "null",
            "overflow", "infinite", "loop", "recursion", "memory",
            "leak", "race", "condition", "deadlock", "timeout",
            "assertion", "failure", "stack", "trace", "previous",
            "attempt", "incorrect", "issue", "problem", "solution",
        ],
        "tool_use": [
            "file", "read", "write", "create", "edit", "search",
            "find", "run", "execute", "shell", "command", "terminal",
            "directory", "folder", "path", "install", "package",
            "config", "yaml", "json", "python", "script",
            "output", "result", "contents", "list", "show",
            "grep", "todo", "project", "code", "module",
            "import", "save", "open", "delete", "move",
        ],
    }
    
    # Confidence thresholds
    KEYWORD_CONFIDENCE_STRONG = 0.85   # Strong keyword match
    KEYWORD_CONFIDENCE_WEAK = 0.60     # Weak keyword match (1 pattern)
    TFIDF_CONFIDENCE_MIN = 0.30        # Minimum TF-IDF score to consider
    BLEND_THRESHOLD = 0.15             # Max diff between top-2 to trigger blending
    BASE_FALLBACK_THRESHOLD = 0.25     # Below this, use base model (no adapter)
    
    def __init__(self):
        """Initialize the router."""
        # Pre-compute IDF-like weights for capability terms
        # (terms that appear in fewer profiles are more discriminative)
        all_terms = set()
        for terms in self.CAPABILITY_PROFILES.values():
            all_terms.update(terms)
        
        self._idf: dict[str, float] = {}
        n_caps = len(self.CAPABILITY_PROFILES)
        for term in all_terms:
            doc_freq = sum(1 for terms in self.CAPABILITY_PROFILES.values() if term in terms)
            self._idf[term] = math.log(n_caps / doc_freq) + 1.0
    
    def route(self, query: str) -> RoutingDecision:
        """Route a query to the optimal adapter(s).
        
        Args:
            query: The user's input query
            
        Returns:
            RoutingDecision with the selected adapter(s) and confidence scores
        """
        # Tier 1: Keyword matching
        keyword_scores = self._keyword_score(query)
        
        # Check for strong keyword match (fast path)
        max_kw_adapter = max(keyword_scores, key=keyword_scores.get)
        max_kw_score = keyword_scores[max_kw_adapter]
        
        if max_kw_score >= self.KEYWORD_CONFIDENCE_STRONG:
            return RoutingDecision(
                primary=max_kw_adapter,
                scores=keyword_scores,
                reasoning=f"Strong keyword match for {max_kw_adapter} ({max_kw_score:.2f})",
            )
        
        # Tier 2: TF-IDF similarity
        tfidf_scores = self._tfidf_score(query)
        
        # Combine keyword and TF-IDF scores (keyword has priority)
        combined = {}
        for adapter in self.CAPABILITY_PROFILES:
            kw = keyword_scores.get(adapter, 0.0)
            tf = tfidf_scores.get(adapter, 0.0)
            # Weighted combination: keywords matter more but TF-IDF breaks ties
            combined[adapter] = 0.65 * kw + 0.35 * tf
        
        # Sort by combined score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        top_adapter, top_score = ranked[0]
        second_adapter, second_score = ranked[1]
        
        # Check if we should fall back to base model
        if top_score < self.BASE_FALLBACK_THRESHOLD:
            return RoutingDecision(
                primary="base",
                scores=combined,
                reasoning=f"Low confidence across all adapters (max={top_score:.2f}), using base model",
            )
        
        # Check if we should blend two adapters
        score_diff = top_score - second_score
        if score_diff < self.BLEND_THRESHOLD and second_score >= self.TFIDF_CONFIDENCE_MIN:
            # Normalize weights so they sum to 1
            total = top_score + second_score
            weights = {
                top_adapter: top_score / total,
                second_adapter: second_score / total,
            }
            return RoutingDecision(
                primary=top_adapter,
                scores=combined,
                reasoning=f"Hybrid query: {top_adapter} ({top_score:.2f}) + {second_adapter} ({second_score:.2f})",
                multi_adapter=True,
                secondary=second_adapter,
                blend_weights=weights,
            )
        
        return RoutingDecision(
            primary=top_adapter,
            scores=combined,
            reasoning=f"Routed to {top_adapter} (combined={top_score:.2f}, kw={keyword_scores.get(top_adapter, 0):.2f}, tf={tfidf_scores.get(top_adapter, 0):.2f})",
        )
    
    def _keyword_score(self, query: str) -> dict[str, float]:
        """Score query against keyword patterns for each adapter."""
        scores = {}
        
        for adapter, patterns in self.KEYWORD_RULES.items():
            matches = sum(1 for p in patterns if p.search(query))
            total = len(patterns)
            
            if matches == 0:
                scores[adapter] = 0.0
            elif matches == 1:
                scores[adapter] = self.KEYWORD_CONFIDENCE_WEAK
            else:
                # Scale from WEAK to STRONG based on number of matches
                ratio = min(matches / (total * 0.5), 1.0)  # Cap at 50% match rate
                scores[adapter] = self.KEYWORD_CONFIDENCE_WEAK + ratio * (
                    self.KEYWORD_CONFIDENCE_STRONG - self.KEYWORD_CONFIDENCE_WEAK
                )
            
            # Apply negative rules
            neg_patterns = self.NEGATIVE_RULES.get(adapter, [])
            neg_matches = sum(1 for p in neg_patterns if p.search(query))
            if neg_matches > 0:
                scores[adapter] *= 0.7  # Penalize by 30%
        
        return scores
    
    def _tfidf_score(self, query: str) -> dict[str, float]:
        """Score query against capability profiles using TF-IDF cosine similarity."""
        # Tokenize query
        query_terms = re.findall(r'\b[a-z]+\b', query.lower())
        query_tf = Counter(query_terms)
        
        # Compute query vector (TF * IDF)
        query_vec: dict[str, float] = {}
        for term, count in query_tf.items():
            if term in self._idf:
                query_vec[term] = count * self._idf[term]
        
        if not query_vec:
            return {adapter: 0.0 for adapter in self.CAPABILITY_PROFILES}
        
        # Compute cosine similarity with each capability profile
        query_norm = math.sqrt(sum(v ** 2 for v in query_vec.values()))
        
        scores = {}
        for adapter, profile_terms in self.CAPABILITY_PROFILES.items():
            # Profile vector: each term has TF=1, weighted by IDF
            dot_product = 0.0
            profile_norm_sq = 0.0
            for term in profile_terms:
                idf = self._idf.get(term, 1.0)
                profile_norm_sq += idf ** 2
                if term in query_vec:
                    dot_product += query_vec[term] * idf
            
            profile_norm = math.sqrt(profile_norm_sq)
            
            if query_norm > 0 and profile_norm > 0:
                scores[adapter] = dot_product / (query_norm * profile_norm)
            else:
                scores[adapter] = 0.0
        
        return scores
    
    def explain(self, query: str) -> str:
        """Get a detailed explanation of the routing decision."""
        decision = self.route(query)
        kw_scores = self._keyword_score(query)
        tf_scores = self._tfidf_score(query)
        
        lines = [
            f"Query: {query[:80]}{'...' if len(query) > 80 else ''}",
            f"",
            f"Keyword scores:  {', '.join(f'{k}={v:.2f}' for k, v in kw_scores.items())}",
            f"TF-IDF scores:   {', '.join(f'{k}={v:.2f}' for k, v in tf_scores.items())}",
            f"Combined scores: {', '.join(f'{k}={v:.2f}' for k, v in decision.scores.items())}",
            f"",
            f"Decision: {decision.reasoning}",
            f"Primary adapter: {decision.primary}",
        ]
        
        if decision.multi_adapter:
            lines.append(f"Secondary adapter: {decision.secondary}")
            lines.append(f"Blend weights: {decision.blend_weights}")
        
        return "\n".join(lines)
