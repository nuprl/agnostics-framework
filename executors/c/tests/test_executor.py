import json
import subprocess
import unittest
from pathlib import Path


class TestCExecutor(unittest.TestCase):
    def setUp(self):
        self.executor_image = "agnostics-c-executor"
        self.test_dir = Path(__file__).parent

    def run_test(self, code: str, test_cases: list, timeout_s: int = 5) -> dict:
        input_data = {
            "code": code,
            "test_cases": test_cases,
            "timeout_s": timeout_s
        }
        
        try:
            process = subprocess.run(
                ["docker", "run", "--rm", "-i", self.executor_image],
                input=json.dumps(input_data),
                capture_output=True,
                text=True,
                timeout=timeout_s + 1  # Add 1 second buffer for Docker overhead
            )
            
            if process.returncode == -9:  # SIGKILL from timeout
                return {"result": "timeout", "stderr": "Process killed due to timeout"}
            elif process.returncode != 0:
                return {"result": "fail:error", "stderr": process.stderr}
                
            return json.loads(process.stdout)
        except subprocess.TimeoutExpired as e:
            # If we get here, it means the subprocess.run itself timed out
            return {"result": "timeout", "stderr": "Test harness timeout"}
        except Exception as e:
            return {"result": "fail:error", "stderr": str(e)}

    def test_successful_program(self):
        code = """
#include <stdio.h>

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    printf("%d\\n", a + b);
    return 0;
}
"""
        test_cases = [
            {"input": "5 3\n", "output": "8\n"},
            {"input": "10 20\n", "output": "30\n"}
        ]
        
        result = self.run_test(code, test_cases)
        self.assertEqual(result["result"], "success")

    def test_compilation_error(self):
        code = """
#include <stdio.h>

int main() {
    printf("Hello World\\n")
    return 0;
}
"""
        test_cases = [{"input": "", "output": ""}]
        
        result = self.run_test(code, test_cases)
        self.assertEqual(result["result"], "fail:error")
        self.assertIn("error", result["stderr"].lower())

    def test_runtime_error(self):
        code = """
#include <stdio.h>

int main() {
    int* ptr = NULL;
    *ptr = 42;  // Segmentation fault
    return 0;
}
"""
        test_cases = [{"input": "", "output": ""}]
        
        result = self.run_test(code, test_cases)
        self.assertEqual(result["result"], "fail:error")

    def test_wrong_output(self):
        code = """
#include <stdio.h>

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    printf("%d\\n", a - b);  // Should be addition
    return 0;
}
"""
        test_cases = [{"input": "5 3\n", "output": "8\n"}]
        
        result = self.run_test(code, test_cases)
        self.assertEqual(result["result"], "fail:wrong-output")
        self.assertEqual(result["expected"], "8\n")
        self.assertEqual(result["got"], "2\n")

    def test_timeout(self):
        code = """
#include <stdio.h>

int main() {
    while(1) {}  // Infinite loop
    return 0;
}
"""
        test_cases = [{"input": "", "output": ""}]
        
        result = self.run_test(code, test_cases, timeout_s=1)
        self.assertEqual(result["result"], "timeout")

    def test_segfault(self):
        code = """
#include <stdio.h>
#include <stdlib.h>

int main() {
    int* arr = 0;
    *arr = 42;
    return 0;
}
"""
        test_cases = [{"input": "", "output": ""}]
        
        result = self.run_test(code, test_cases)
        self.assertEqual(result["result"], "fail:error")

if __name__ == '__main__':
    unittest.main() 