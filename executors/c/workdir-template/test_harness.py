import json
import subprocess
from pathlib import Path

from container_protocol import *


cwd = Path.cwd()
snippet_c = cwd/'snippet.c'
snippet_exe = cwd/'snippet'


def test_snippet(input_str: str, timeout_s: int):
    # First compile the code
    compile_process = subprocess.run(
        ['gcc', '-o', str(snippet_exe), str(snippet_c)],
        capture_output=True,
        text=True,
    )

    if compile_process.returncode != 0:
        return compile_process.returncode, '', compile_process.stderr

    # Then run the compiled executable
    process = subprocess.run(
        [str(snippet_exe)],
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
        snippet_c.write_text(code)

        res = None
        for test_case in test_cases:
            in_str = test_case['input']
            expected_output = test_case['output']
            try:
                exit_code, real_output, stderr = test_snippet(in_str, timeout_s)
            except subprocess.TimeoutExpired:
                res = res_fail_timeout()
                break
            except subprocess.CalledProcessError as e:
                res = res_fail_other(details={
                    'type': 'subprocess.CalledProcessError',
                    'exception': str(e),
                })
            except Exception as e:
                res = res_fail_other(details={
                    'type': 'Unkown Exception',
                    'exception': str(e),
                })
                break


            if exit_code == -9:  # SIGKILL from timeout
                res = res_fail_timeout(stdout=real_output, stderr=stderr)
                break
            elif exit_code != 0:
                res = res_fail_error(exit_code=exit_code, stderr=stderr)
                break

            if real_output.rstrip() != expected_output.rstrip():
                res = res_fail_wrong_output(expected=expected_output, got=real_output, stderr=stderr)
                break

            res = res_success(stderr=stderr)

        print(json.dumps(res))


if __name__ == '__main__':
    main()