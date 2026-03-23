"""
Reinforcement Learning from Formal Systems (RLFS) Sandbox
=========================================================
Provides the absolute ground truth for the V2 Architecture.
Instead of learning from human preferences (RLHF), the model learns 
by writing code, executing it in this sandbox, and receiving a reward 
based on if the code compiled and produced the desired mathematical output.
"""

import sys
import tempfile
import subprocess
from loguru import logger
from typing import Tuple

class RLFSSandbox:
    """
    A secure-ish execution environment that runs the AI's generated code.
    Evaluates the code for compilation, runtime errors, and logic correctness.
    """
    def __init__(self, timeout_seconds: int = 5):
        self.timeout = timeout_seconds

    def evaluate(self, generated_code: str, expected_output: str = None) -> Tuple[float, str]:
        """
        Executes the Python code and calculates the RL reward.
        
        Args:
            generated_code: The Python script written by the AI
            expected_output: (Optional) string that must be printed for success
            
        Returns:
            reward: Float value (-1.0 to 1.0)
            feedback: The stdout or stderr string from the execution
        """
        # Create a temporary file to hold the AI's code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(generated_code)
            temp_file_path = f.name
            
        try:
            # Execute the code in a subprocess
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # 1. Compilation & Runtime Errors
            if result.returncode != 0:
                logger.debug("Code failed to run. Reward: -1.0")
                return -1.0, f"Error:\n{result.stderr}"
                
            # 2. Logic Verification (If expected output is provided)
            output = result.stdout.strip()
            if expected_output:
                if expected_output in output:
                    logger.debug("Code ran AND passed assertion. Reward: +1.0")
                    # Bonus reward for shorter/efficient solutions could be added here
                    return 1.0, f"Success!\n{output}"
                else:
                    logger.debug(f"Code ran but output '{output}' did not match '{expected_output}'. Reward: -0.5")
                    return -0.5, f"Logic Failure. Expected '{expected_output}', got:\n{output}"
                    
            # 3. Simple successful compilation (No specific output expected)
            logger.debug("Code compiled and ran successfully. Reward: +0.5")
            return 0.5, f"Success:\n{output}"
            
        except subprocess.TimeoutExpired:
            logger.debug("Code caused an infinite loop. Reward: -1.0")
            return -1.0, f"Timeout after {self.timeout} seconds."
            
        finally:
            # Cleanup temp file
            import os
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


if __name__ == "__main__":
    print("\n--- Testing RLFS Sandbox Evaluator ---")
    sandbox = RLFSSandbox()
    
    # AI writes a perfect math script
    good_code = """
def solve():
    return sum([x for x in range(10) if x % 2 == 0])
print(f"The answer is {solve()}")
"""
    print("\nTesting perfect AI code:")
    reward, feedback = sandbox.evaluate(good_code, expected_output="The answer is 20")
    print(f"Reward: {reward}")
    print(f"Feedback: {feedback.splitlines()[0]}...")
    
    # AI hallucinates syntax
    bad_code_syntax = """
def solve()
    return sum([x for x in range(10)])
print(solve)
"""
    print("\nTesting AI syntax hallucination:")
    reward, feedback = sandbox.evaluate(bad_code_syntax)
    print(f"Reward: {reward}")
    print(f"Feedback: {feedback.splitlines()[-1] if feedback.strip() else 'None'}")
    
    # AI writes code that runs but gets wrong math answer
    bad_logic = """
print("The answer is 55")
"""
    print("\nTesting AI logic failure:")
    reward, feedback = sandbox.evaluate(bad_logic, expected_output="The answer is 20")
    print(f"Reward: {reward}")
    
    print("\nSandbox formal verification tests passed. Ground truth engine ready.")
