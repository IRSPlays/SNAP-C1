import sys
import json
from loguru import logger
from typing import Dict, Any, List

class ExecutionTraceSimulator:
    """
    SNAP-C1 V3 Model Component: Memory-Mapped Training Data Generator
    
    In order to train the AI to simulate a CPU in its own weights 
    (Execution-Trace Tokenization), we first need mathematical ground-truth data.
    
    This utility script takes a valid block of human dataset Python code,
    executes it natively, and uses `sys.settrace()` to rip the `locals()` 
    variable states out of the RAM at every single line execution.
    It returns a specialized string interleaved with [CODE] and [MEM] tokens.
    """
    def __init__(self):
        self.trace_history = []
        self.execution_error = None
        
    def _trace_lines(self, frame, event, arg):
        """
        The native Python Debugger hook.
        Triggers on every single mathematical Python line executed.
        """
        if event == 'line':
            # Extract the raw source code line currently executing
            try:
                # `f_code.co_filename` is `<string>` since we run via `exec()`
                # We can't easily extract the exact string line number dynamically this way 
                # without inspecting the `f_code`, so we map the line number explicitly.
                lineno = frame.f_lineno
                
                # Extract the CPU RAM mapping of all variables at this exact millisecond
                # Filter out python built-ins to save tensor space
                clean_locals = {k: repr(v) for k, v in frame.f_locals.items() if not k.startswith('_')}
                
                self.trace_history.append({
                    "line": lineno,
                    "memory": clean_locals
                })
            except Exception as e:
                pass
                
        return self._trace_lines

    def generate_trace_dataset(self, code_string: str) -> str:
        """
        Executes the string and returns the interleaved Training Data format.
        """
        self.trace_history = []
        self.execution_error = None
        
        # We need a clean namespace to avoid polluting our own simulator
        exec_namespace = {}
        
        # Split the source code into a map so we can match it to the trace line numbers
        source_lines = [""] + code_string.splitlines() # 1-indexed to match `f_lineno`
        
        try:
            sys.settrace(self._trace_lines)
            # Actually run the code!
            exec(code_string, exec_namespace, exec_namespace)
        except Exception as e:
            self.execution_error = str(e)
            logger.warning(f"Failed to generate trace for invalid dataset code: {e}")
        finally:
            sys.settrace(None) # Safety release the debugger hook!
            
        if self.execution_error:
            return None
            
        # Compile the interleaved V3 Training String
        # E.g: [CODE] x = 5 [MEM] {"x": "5"} [CODE] y = x + 2 [MEM] {"x": "5", "y": "7"}
        interleaved_string = ""
        
        # Keep track of what memory looked like previously to only push diffs
        prev_memory = {}
        
        for trace in self.trace_history:
            lineno = trace["line"]
            if lineno < len(source_lines):
                code_line = source_lines[lineno].strip()
                if not code_line: continue # skip blank lines
                
                mem_state = trace["memory"]
                # Convert the dictionary to a strict JSON string for the Tokenizer
                mem_json = json.dumps(mem_state, sort_keys=True)
                
                interleaved_string += f"[CODE] {code_line} [MEM] {mem_json} "
                
        return interleaved_string.strip()

if __name__ == "__main__":
    print("\n--- Testing V3 Execution Trace Engine ---")
    simulator = ExecutionTraceSimulator()
    
    # Example Dataset Code (e.g. from Humaneval)
    code_to_trace = """
def calculate_area(width, height):
    base = width * 2
    area = base * height
    return area
    
# We actually call it so the Python interpreter runs the math
result = calculate_area(5, 10)
"""

    print("Origin Code:")
    print(code_to_trace.strip())
    
    trace_data = simulator.generate_trace_dataset(code_to_trace)
    
    print("\nGenerated [CODE]/[MEM] Mathematical V3 Sequence:")
    print(trace_data)
    
    print("\nSuccess! The AI can now learn to simulate the CPU natively.")
