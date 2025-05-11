import json
import subprocess
from pathlib import Path

from container_protocol import *


cwd = Path.cwd()
snippet_f = cwd/'snippet.py'


def test_snippet(input_str: str, timeout_s: int):
    process = subprocess.run(
        ['python3', str(snippet_f)],
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
        code = data['code']
        test_cases = data['test_cases']
        timeout_s = data['timeout_s']
        snippet_f.write_text(code)

        res = None
        for test_case in test_cases:
            in_str = test_case['input']
            expected_output = test_case['output']
            try:
                exit_code, real_output, stderr = test_snippet(in_str, timeout_s)
            except subprocess.TimeoutExpired:
                res = res_fail_timeout()
                break

            if exit_code != 0:
                res = res_fail_error(exit_code=exit_code, stderr=stderr)
                break

            if real_output.rstrip() != expected_output.rstrip():
                res = res_fail_wrong_output(expected=expected_output, got=real_output, stderr=stderr)
                break

            res = res_success(stderr=stderr)

        print(json.dumps(res))


if __name__ == '__main__':
    main()
