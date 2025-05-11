import json
import tempfile
from bounded_subprocess.bounded_subprocess import run
from container_protocol import *

def test_snippet(filename: str, input_str: str, timeout_s: int):
    process = run(
        ["utop", "-require", "base", "-require", "stdio", filename],
        stdin_data=input_str,
        timeout_seconds=timeout_s,
    )
    return process.exit_code, process.stdout, process.stderr


def main():
    while True:
        try:
            data = json.loads(input())
        except EOFError:
            break
        code = data['code']
        test_cases = data['test_cases']
        timeout_s = data['timeout_s']
        # An advantage of using a temp file is that we can expect it to save to
        # a local file system on a Slurm cluster.

        with tempfile.NamedTemporaryFile(suffix='.ml', mode="wt") as f:
            f.write(code)
            f.flush()

            res = None
            for i, test_case in enumerate(test_cases):
                in_str = test_case['input']
                expected_output = test_case['output']
                try:
                    exit_code, real_output, stderr = test_snippet(f.name, in_str, timeout_s)
                except subprocess.TimeoutExpired:
                    res = res_fail_timeout()
                    break

                if exit_code != 0:
                    res = res_fail_error(
                        exit_code=exit_code,
                        stderr=stderr,
                    )
                    break

                if real_output.rstrip() != expected_output.rstrip():
                    res = res_fail_wrong_output(
                        expected=expected_output,
                        got=real_output,
                        stderr=stderr,
                    )
                    break

                res = res_success(stderr=stderr)

        print(json.dumps(res))


if __name__ == '__main__':
    main()
