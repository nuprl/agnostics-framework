'''
A script to test C++ generations, either from offline evaluation or from an RL run.
Input is a JSONL file with the following fields in each line:
- prompt: A list with following entries under each:
  - role: "system" or "user"
  - content": string
- outputs: list of strings

Algorithm:
- Accept a command line argument for the path to the input file.
- Launch the cpp executor in a subprocess.
- Step through the input file, and for each line:
  - For each output in the outputs list:
  - Find the cpp code encapsulated in a fenced code block (```cpp)
  - If no fenced code block is found, print the output and wait to continue
  - If a fenced code block is found, run the code with empty test cases just to make sure it
  compiles
  - If the code does not compile, print the error message and wait to continue

"Wait to continue" means to print appropriate messages and wait for the user to press enter.
'''

import argparse
import json
import re
import subprocess
import sys
import atexit
from pathlib import Path
from typing import Optional, Tuple


def extract_cpp_code(text: str) -> Optional[str]:
  """Extract C++ code from fenced code blocks."""
  pattern = r"```cpp\n(.*?)```"
  matches = re.findall(pattern, text, re.DOTALL)
  return matches[0] if matches else None


class ContainerManager:
  def __init__(self):
    self.process = None
    self.start_container()
    atexit.register(self.cleanup)

  def start_container(self):
    """Start the container and set up pipes."""
    self.process = subprocess.Popen(
      ["docker", "run", "--rm", "--tmpfs", "/ramdisk:size=512m,exec",
       "-i", "agnostics-cpp-executor"],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      bufsize=1  # Line buffered
    )

  def cleanup(self):
    """Clean up the container process."""
    if self.process:
      self.process.terminate()
      try:
        self.process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        self.process.kill()

  def test_cpp_code(self, code: str) -> dict:
    """Test C++ code using the existing container."""
    input_data = {
      "code": code,
      "test_cases": [{"input": "", "output": ""}],  # Empty test case just for compilation
      "timeout_s": 50
    }
    
    try:
      # Send input data
      self.process.stdin.write(json.dumps(input_data) + "\n")
      self.process.stdin.flush()
      
      # Read response
      response = self.process.stdout.readline()
      return json.loads(response)
      
    except json.JSONDecodeError as e:
      print(f"JSON decode error:\n {str(e)}\nResponse: {response}")
      return {"result": "fail:JSONDecodeError", "stderr": str(e)}
    except Exception as e:
      return {"result": "fail:Exception", "stderr": str(e)}


def process_file(file_path: Path, container_manager: ContainerManager):
  """Process the input JSONL file."""
  with open(file_path) as f:
    for line_num, line in enumerate(f, 1):
      print(f"\nProcessing line {line_num}")
      try:
        data = json.loads(line)
        # If there is a "response" field, use that instead of the "outputs" field.
        if "response" in data:
          outputs = [data["response"]]
        else:
          outputs = data.get("outputs", [])
        
        for i, output in enumerate(outputs, 1):
          print(f"Output {i:4d}:", end=" ")
          
          cpp_code = extract_cpp_code(output)
          if not cpp_code:
            print("✗ No C++ code block found in output")
            continue
          
          result = container_manager.test_cpp_code(cpp_code)
          
          if result["result"] == "success":
            print("✓ Success")
          elif "Exception" in result["result"]:
            print(f"✗ Exception: {result['result']}")
            print(f"stderr: {result['stderr']}")
            sys.exit(1)
          elif "compile-error" in result["result"]:
            print(f"✗ Compile error: {result['result']}")
            print(f"stderr: {result['stderr']}")
            print(f"stdout: {result['stdout']}")
            print(f"exit_code: {result['exit_code']}")
            print(f"C++ code:")
            print("="*100)
            print(cpp_code)
            print("="*100)
            sys.exit(1)
          else:
            print(f"✗ Failure: {result['result']}", end=" ")
            if "details" in result:
              print(f" || Details: {result['details']['type']}")
            else:
              print()
          
      except json.JSONDecodeError as e:
        print(f"Error parsing JSON at line {line_num}: {e}", file=sys.stderr)
        print(f"Line: {line}")
        input("Press Enter to continue...")


def main():
  parser = argparse.ArgumentParser(description="Test C++ code generations from a JSONL file")
  parser.add_argument("input_file", type=Path, help="Path to the input JSONL file")
  args = parser.parse_args()
  
  if not args.input_file.exists():
    print(f"Error: File {args.input_file} does not exist", file=sys.stderr)
    sys.exit(1)
  
  container_manager = ContainerManager()
  process_file(args.input_file, container_manager)


if __name__ == "__main__":
  main()
