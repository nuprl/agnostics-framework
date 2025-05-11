import json
import subprocess
from pathlib import Path
from container_protocol import *

cwd = Path.cwd()


def test_snippet(input_str: str, timeout_s: int):
    # Run the compiled Java class
    process = subprocess.run(
        ["java", str(cwd/"snippet.java")],
        input=input_str,
        text=True,
        capture_output=True,
        timeout=float(timeout_s),
    )
    return process.returncode, process.stdout, process.stderr

def main():
    while True:
        try:
            data = json.loads(input())
        except EOFError:
            break

        code = data["code"]
        test_cases = data["test_cases"]
        timeout_s = data["timeout_s"]

        (cwd/'snippet.java').write_text(code)

        res = None
        for test_case in test_cases:
            in_str = test_case["input"]
            expected_output = test_case["output"]

            try:
                exit_code, real_output, stderr = test_snippet(in_str, timeout_s)
            except subprocess.TimeoutExpired:
                res = res_fail_timeout()
                break
            except subprocess.CalledProcessError as e:
                res = res_fail_other(details={
                    "type": "subprocess.CalledProcessError",
                    "exception": str(e),
                })
                break
            except Exception as e:
                res = res_fail_other(details={
                    "type": "Unknown Exception",
                    "exception": str(e),
                })
                break

            if exit_code != 0:
                # Compile error or runtime error
                res = res_fail_error(exit_code=exit_code, stderr=stderr)
                break

            if real_output.rstrip() != expected_output.rstrip():
                res = res_fail_wrong_output(expected=expected_output, got=real_output, stderr=stderr)
                break

            res = res_success(stderr=stderr)

        print(json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
