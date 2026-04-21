"""
SNAP-C1 GGUF Export Script
============================
Exports the merged SNAP-C1 model to GGUF format for use in LM Studio.

Prerequisites:
1. Merge adapters into base model first:
   python training/merge_adapters.py --adapters team_thinking self_correction tool_use --merge-into-base
2. Install llama.cpp: pip install llama-cpp-python
   Or clone https://github.com/ggerganov/llama.cpp and build convert tools

Usage:
    python export_gguf.py --input adapters/merged --quantization Q4_K_M
    python export_gguf.py --input adapters/merged --quantization Q5_K_M
"""

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent


def find_llama_cpp_convert():
    """Find the llama.cpp conversion script."""
    # Check common locations
    possible_paths = [
        Path("llama.cpp/convert_hf_to_gguf.py"),
        Path.home() / "llama.cpp/convert_hf_to_gguf.py",
        Path("/usr/local/bin/convert_hf_to_gguf.py"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # Try to find via pip-installed package
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import llama_cpp; print(llama_cpp.__file__)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("llama-cpp-python is installed, but convert script needs llama.cpp repo")
    except Exception:
        pass
    
    return None


def export_to_gguf(input_dir: str, quantization: str, output_path: str | None = None):
    """Export HuggingFace model to GGUF format."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        logger.error("Run merge_adapters.py --merge-into-base first")
        sys.exit(1)
    
    if output_path is None:
        output_path = str(input_path / f"snap-c1-{quantization.lower()}.gguf")
    
    convert_script = find_llama_cpp_convert()
    
    if convert_script is None:
        logger.error("Could not find llama.cpp convert script.")
        logger.error("")
        logger.error("To install llama.cpp for GGUF conversion:")
        logger.error("  1. git clone https://github.com/ggerganov/llama.cpp")
        logger.error("  2. pip install -r llama.cpp/requirements.txt")
        logger.error("")
        logger.error("Then re-run this script.")
        sys.exit(1)
    
    # Step 1: Convert to GGUF (F16)
    logger.info(f"Converting {input_path} to GGUF (F16)...")
    f16_path = str(input_path / "snap-c1-f16.gguf")
    
    cmd_convert = [
        sys.executable,
        convert_script,
        str(input_path),
        "--outfile", f16_path,
        "--outtype", "f16",
    ]
    
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        sys.exit(1)
    
    logger.info(f"F16 GGUF saved to: {f16_path}")
    
    # Step 2: Quantize
    if quantization.upper() != "F16":
        logger.info(f"Quantizing to {quantization}...")
        
        # Find quantize binary
        quantize_bin = Path(convert_script).parent / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = Path(convert_script).parent / "llama-quantize"
        
        if not quantize_bin.exists():
            logger.warning(f"llama-quantize binary not found at {quantize_bin}")
            logger.warning("You may need to build llama.cpp first: cd llama.cpp && make")
            logger.warning(f"F16 model is available at: {f16_path}")
            logger.warning(f"Quantize manually: llama-quantize {f16_path} {output_path} {quantization}")
            return f16_path
        
        cmd_quantize = [
            str(quantize_bin),
            f16_path,
            output_path,
            quantization,
        ]
        
        result = subprocess.run(cmd_quantize, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Quantization failed: {result.stderr}")
            sys.exit(1)
        
        logger.info(f"Quantized GGUF saved to: {output_path}")
        
        # Clean up F16 file (it's large)
        logger.info(f"Cleaning up F16 intermediate: {f16_path}")
        Path(f16_path).unlink(missing_ok=True)
    else:
        output_path = f16_path
    
    logger.info(f"Export complete! GGUF file: {output_path}")
    logger.info(f"Load in LM Studio: File → Open Model → select {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 GGUF Export")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "adapters" / "merged"),
        help="Path to merged model directory",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="Q4_K_M",
        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16"],
        help="GGUF quantization type (Q4_K_M recommended for 8GB VRAM)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GGUF file path",
    )
    args = parser.parse_args()
    
    export_to_gguf(args.input, args.quantization, args.output)


if __name__ == "__main__":
    main()
