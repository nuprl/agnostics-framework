import json
import tempfile
from bounded_subprocess.bounded_subprocess import run
from container_protocol import *
import os, errno


def test_snippet(filename: str, input_str: str, timeout_s: int):
    with tempfile.NamedTemporaryFile(suffix=".out", mode="rt", delete=False) as out_file:
        # We let the file be closed since otherwise it's busy and can't be written to.
        pass
    try:
        compile_result = run(["gfortran", "-o", out_file.name, filename], timeout_seconds=timeout_s)
        if compile_result.exit_code != 0:
            return compile_result.exit_code, compile_result.stdout, compile_result.stderr
        run_result = run([out_file.name], stdin_data=input_str, timeout_seconds=timeout_s)
        return run_result.exit_code, run_result.stdout, run_result.stderr
    finally:
        # Try to remove the file. Occasionally it may not exist (if compilation fails?).
        try:
            os.unlink(out_file.name)
        except OSError as e:
            # Swallow "file does not exit" errors.
            if e.errno != errno.ENOENT:
                raise

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
        with tempfile.NamedTemporaryFile(suffix=".f90", mode="wt") as f:
            f.write(code)
            f.flush()

            res = None
            for test_case in test_cases:
                in_str = test_case['input']
                expected_output = test_case['output']
                exit_code, real_output, stderr = test_snippet(f.name, in_str, timeout_s)
                if exit_code == -1:
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
