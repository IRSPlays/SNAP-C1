"""
SNAP-C1 Memory Manager
========================
Persistent memory system using ChromaDB for:
- Episodic memory (conversation history, interactions)
- Semantic memory (facts, knowledge, learned concepts)
- Skill memory (successful approaches, patterns)
- User preferences (communication style, project context)

The memory system allows SNAP-C1 to learn across conversations
and build a persistent knowledge base.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed. Memory system disabled.")

try:
    from .embeddings import EmbeddingModel
except ImportError:
    EmbeddingModel = None

PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory" / "chromadb_store"


@dataclass
class Memory:
    """A single memory entry."""
    content: str
    memory_type: str  # episodic, semantic, skill, preference
    source: str       # conversation, self_reflection, tool_output, user
    timestamp: float
    metadata: dict
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        return cls(**data)


class MemoryManager:
    """Manages SNAP-C1's persistent memory system."""
    
    COLLECTIONS = {
        "episodic": "Conversation history and interaction memories",
        "semantic": "Facts, knowledge, and learned concepts",
        "skills": "Successful approaches, patterns, and techniques",
        "preferences": "User preferences and project context",
    }
    
    def __init__(self, persist_dir: str | Path | None = None):
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB is required for the memory system.")
            logger.error("Install it: pip install chromadb")
            self.client = None
            return
        
        persist_dir = Path(persist_dir) if persist_dir else MEMORY_DIR
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Initialize collections
        self.collections = {}
        for name, description in self.COLLECTIONS.items():
            self.collections[name] = self.client.get_or_create_collection(
                name=f"snap_c1_{name}",
                metadata={"description": description},
            )
        
        logger.info(f"Memory system initialized at: {persist_dir}")
        self._log_stats()
    
    def _log_stats(self):
        """Log memory statistics."""
        for name, collection in self.collections.items():
            count = collection.count()
            logger.info(f"  {name}: {count} memories")
    
    def _generate_id(self, content: str, memory_type: str) -> str:
        """Generate a unique ID for a memory entry."""
        hash_input = f"{content}:{memory_type}:{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def store(
        self,
        content: str,
        memory_type: str,
        source: str = "conversation",
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> str:
        """Store a new memory.
        
        Args:
            content: The memory content (text)
            memory_type: One of: episodic, semantic, skills, preferences
            source: Where this memory came from
            importance: How important this memory is (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            The memory ID
        """
        if self.client is None:
            logger.warning("Memory system not available. Skipping store.")
            return ""
        
        if memory_type not in self.COLLECTIONS:
            logger.error(f"Invalid memory type: {memory_type}. Must be one of: {list(self.COLLECTIONS.keys())}")
            return ""
        
        memory_id = self._generate_id(content, memory_type)
        now = time.time()
        
        meta = {
            "source": source,
            "importance": importance,
            "timestamp": now,
            "access_count": 0,
            "last_accessed": now,
        }
        if metadata:
            meta.update(metadata)
        
        # Store in appropriate collection
        self.collections[memory_type].add(
            ids=[memory_id],
            documents=[content],
            metadatas=[meta],
        )
        
        logger.debug(f"Stored {memory_type} memory: {memory_id} ({len(content)} chars)")
        return memory_id
    
    def recall(
        self,
        query: str,
        memory_type: str | None = None,
        n_results: int = 5,
        min_importance: float = 0.0,
    ) -> list[dict]:
        """Recall relevant memories based on a query.
        
        Args:
            query: Search query (semantic similarity)
            memory_type: Specific collection to search (None = search all)
            n_results: Maximum number of results
            min_importance: Minimum importance threshold
            
        Returns:
            List of memory dicts with content, metadata, and relevance score
        """
        if self.client is None:
            return []
        
        results = []
        collections_to_search = (
            [self.collections[memory_type]] if memory_type 
            else list(self.collections.values())
        )
        
        for collection in collections_to_search:
            if collection.count() == 0:
                continue
            
            # Query with semantic similarity
            query_results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count()),
            )
            
            if not query_results["documents"] or not query_results["documents"][0]:
                continue
            
            for i, (doc, meta, dist) in enumerate(zip(
                query_results["documents"][0],
                query_results["metadatas"][0],
                query_results["distances"][0],
            )):
                # Filter by importance
                if meta.get("importance", 0) < min_importance:
                    continue
                
                # Update access count
                memory_id = query_results["ids"][0][i]
                meta["access_count"] = meta.get("access_count", 0) + 1
                meta["last_accessed"] = time.time()
                
                collection.update(
                    ids=[memory_id],
                    metadatas=[meta],
                )
                
                results.append({
                    "id": memory_id,
                    "content": doc,
                    "metadata": meta,
                    "relevance": 1.0 - dist,  # Convert distance to similarity
                    "collection": collection.name,
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:n_results]
    
    def forget(self, memory_id: str, memory_type: str) -> bool:
        """Remove a specific memory."""
        if self.client is None:
            return False
        
        if memory_type not in self.collections:
            return False
        
        try:
            self.collections[memory_type].delete(ids=[memory_id])
            logger.debug(f"Forgot memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to forget memory {memory_id}: {e}")
            return False
    
    def consolidate(self, memory_type: str = "episodic", max_age_days: int = 30):
        """Consolidate old memories.
        
        Low-importance, rarely-accessed memories older than max_age_days
        get removed. High-importance memories are preserved.
        """
        if self.client is None:
            return
        
        collection = self.collections.get(memory_type)
        if not collection or collection.count() == 0:
            return
        
        cutoff = time.time() - (max_age_days * 86400)
        
        # Get all memories
        all_data = collection.get(include=["metadatas"])
        
        to_delete = []
        for memory_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            timestamp = meta.get("timestamp", 0)
            importance = meta.get("importance", 0.5)
            access_count = meta.get("access_count", 0)
            
            # Keep if: recent, important, or frequently accessed
            if timestamp > cutoff:
                continue
            if importance >= 0.7:
                continue
            if access_count >= 5:
                continue
            
            to_delete.append(memory_id)
        
        if to_delete:
            collection.delete(ids=to_delete)
            logger.info(f"Consolidated {len(to_delete)} old {memory_type} memories")
    
    def get_context_injection(self, query: str, max_tokens: int = 1024) -> str:
        """Get formatted memory context to inject into the model's prompt.
        
        Returns a string of relevant memories formatted for injection
        into the system prompt or context window.
        """
        memories = self.recall(query, n_results=10)
        
        if not memories:
            return ""
        
        lines = ["[Relevant memories from previous interactions:]"]
        total_chars = 0
        char_limit = max_tokens * 4  # Rough token-to-char estimate
        
        for mem in memories:
            entry = f"- [{mem['metadata'].get('source', 'unknown')}] {mem['content']}"
            if total_chars + len(entry) > char_limit:
                break
            lines.append(entry)
            total_chars += len(entry)
        
        return "\n".join(lines)
    
    def store_conversation(self, user_msg: str, assistant_msg: str, quality_score: float = 0.5):
        """Store a conversation exchange as episodic memory.
        
        Args:
            user_msg: The user's message
            assistant_msg: The assistant's response
            quality_score: How good the interaction was (0.0-1.0)
        """
        content = f"User asked: {user_msg[:200]}\nI responded: {assistant_msg[:500]}"
        self.store(
            content=content,
            memory_type="episodic",
            source="conversation",
            importance=quality_score,
            metadata={
                "user_msg_length": len(user_msg),
                "assistant_msg_length": len(assistant_msg),
            },
        )
    
    def store_skill(self, skill_description: str, example: str, domain: str = "general"):
        """Store a learned skill or successful approach."""
        content = f"Skill: {skill_description}\nExample: {example}"
        self.store(
            content=content,
            memory_type="skills",
            source="self_reflection",
            importance=0.7,
            metadata={"domain": domain},
        )
    
    def store_fact(self, fact: str, source: str = "learned", confidence: float = 0.8):
        """Store a learned fact or piece of knowledge."""
        self.store(
            content=fact,
            memory_type="semantic",
            source=source,
            importance=confidence,
            metadata={"confidence": confidence},
        )
    
    def stats(self) -> dict:
        """Get memory system statistics."""
        if self.client is None:
            return {"status": "unavailable"}
        
        stats = {"status": "active", "collections": {}}
        for name, collection in self.collections.items():
            stats["collections"][name] = {
                "count": collection.count(),
            }
        stats["total_memories"] = sum(
            c["count"] for c in stats["collections"].values()
        )
        return stats
