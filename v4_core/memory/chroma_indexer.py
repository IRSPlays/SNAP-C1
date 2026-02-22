import os
import sys
from loguru import logger
from typing import List, Dict

try:
    import chromadb
except ImportError:
    chromadb = None

# Add project root explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

class V4RepositoryIndexer:
    """
    SNAP-C1 V4: Local Vector Database & Context Retrieval
    
    Because the Holographic Compressor cannot natively compress a 10,000-file
    SWE-Bench repository into a 1024-d array without loss, we build a local
    RAG (Retrieval-Augmented Generation) pipeline using ChromaDB.
    
    This breaks massive Python repositories down into sub-chunks (Classes/Functions),
    embeds them offline using Fast BPE vectors, and allows the V4 Router to 
    instantly search for mathematically relevant functions during a bug fix.
    """
    def __init__(self, db_path: str = "./v4_core/chroma_db"):
        if chromadb is None:
            logger.warning("ChromaDB is not installed. To run V4 Retrieval, run `pip install chromadb`")
            self.collection = None
            return
            
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # We use a cosine-similarity embedding function by default
        self.collection = self.client.get_or_create_collection(
            name="v4_swe_bench_repo",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"V4 Repository Indexer Initialized at {db_path}")

    def chunk_and_index_file(self, filepath: str, chunk_size: int = 40):
        """
        Naive chunking for prototype V4. Reads a python file, breaks it into
        chunks of N lines, and shoves it into the Chroma Vector space.
        """
        if self.collection is None: return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            file_identifier = os.path.basename(filepath)
            
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i + chunk_size]
                chunk_text = "".join(chunk_lines).strip()
                
                if not chunk_text: continue
                
                chunk_id = f"{file_identifier}_chunk_{i}"
                
                # In production V4, we use our own Holographic Compressor embeddings.
                # Here, we let ChromaDB use its default fast embedding mapping.
                self.collection.upsert(
                    documents=[chunk_text],
                    metadatas=[{"file": file_identifier, "start_line": i}],
                    ids=[chunk_id]
                )
        except Exception as e:
            logger.error(f"Failed to index {filepath}: {e}")

    def query_context(self, prompt: str, top_k: int = 3) -> List[Dict]:
        """
        The Discovery Head Network query.
        Given a bug report (e.g. ValueError in Router), physically search the
        vector database for the matching Python chunks and retrieve them 
        into the VRAM Context window for assembly.
        """
        if self.collection is None: return []
        
        results = self.collection.query(
            query_texts=[prompt],
            n_results=top_k
        )
        
        retrieved_context = []
        for i in range(len(results['documents'][0])):
            retrieved_context.append({
                "id": results['ids'][0][i],
                "file": results['metadatas'][0][i]["file"],
                "text": results['documents'][0][i]
            })
            
        return retrieved_context

if __name__ == "__main__":
    print("\n=== Testing V4 Context Retrieval Engine ===")
    
    indexer = V4RepositoryIndexer()
    
    # We will simulate indexing the V3 Architecture codebase itself!
    v3_folder = os.path.join(project_root, "v3_core", "architecture")
    
    if indexer.collection is not None and os.path.exists(v3_folder):
        print(f"Indexing physical codebase: {v3_folder}")
        for filename in os.listdir(v3_folder):
            if filename.endswith(".py"):
                indexer.chunk_and_index_file(os.path.join(v3_folder, filename))
                
        print("\nRepository Embedded. Simulating SWE-Bench Query...")
        query = "Where is the ASTDecoder defined for structural inference?"
        print(f"User Prompt: '{query}'")
        
        retrieved_blocks = indexer.query_context(query, top_k=1)
        
        if retrieved_blocks:
            best_match = retrieved_blocks[0]
            print(f"\n[SUCCESS] Retrieved Mathematical Context from {best_match['file']} (Score Matrix Indexed):")
            print("--- Snippet ---")
            print(best_match['text'][:200] + "...\n")
            print("This context block will now be routed to the SSD Streamer.")
    else:
        print("\nChromaDB dependency missing, skipping physical database verification.")
