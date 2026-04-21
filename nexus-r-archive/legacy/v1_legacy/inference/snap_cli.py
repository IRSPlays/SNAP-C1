"""
SNAP-C1 v2 CLI Interface: God Mode
===================================
Interactive command-line interface for the Recursive Self-Improving Intelligence.

Usage:
    python snap_cli.py --query "Build a React app"
    python snap_cli.py --chat

Features:
- Streams 'System 2' thought process (Thinking -> Action -> Observation).
- Shows Memory Retrieval and Storage events.
- Handles autonomous tool execution.
"""

import argparse
import sys
import time
from pathlib import Path

# Fix Windows console encoding
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.molora_pipeline import MoLORAPipeline
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 v2 God Mode CLI")
    parser.add_argument("--query", type=str, help="Single query to execute recursively")
    parser.add_argument("--chat", action="store_true", help="Interactive session")
    parser.add_argument("--no-recursion", action="store_true", help="Disable System 2 loop (legacy mode)")
    args = parser.parse_args()

    print("=" * 70)
    print("SNAP-C1 v2: Recursive Self-Improving Intelligence")
    print("Architecture: MoLoRA + ThoughtController + Episodic Memory")
    print("=" * 70)
    
    # Initialize Pipeline (loads model + memory + tools)
    logger.info("Initializing Neural Core...")
    pipeline = MoLORAPipeline(verbose=True) 
    
    recursive_mode = not args.no_recursion
    logger.info(f"System 2 Loop: {'ENABLED' if recursive_mode else 'DISABLED'}")

    if args.query:
        print(f"\nUser: {args.query}")
        result = pipeline.run(args.query, recursive=recursive_mode)
        print("\n" + "="*30 + " FINAL ANSWER " + "="*30)
        print(result["response"])
        print("="*74)
        
    elif args.chat:
        print("\n[Interactive Mode] Type 'quit' to exit.\n")
        history = []
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            
            if not user_input or user_input.lower() in ["quit", "exit"]:
                break
            
            result = pipeline.run(user_input, conversation_history=history, recursive=recursive_mode)
            
            # Update history (for short-term context)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result["response"]})
            
            print("\n" + "="*30 + " FINAL ANSWER " + "="*30)
            print(result["response"])
            print("="*74)

if __name__ == "__main__":
    main()
