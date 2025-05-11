import json
import sys
import subprocess
from pathlib import Path

from container_protocol import *


cwd = Path.cwd()
snippet_cpp = Path('/ramdisk/snippet.cpp')
snippet_exe = Path('/ramdisk/snippet')
# snippet_cpp = cwd/'snippet.cpp'
# snippet_exe = cwd/'snippet'
compile_timeout_s = 60

def compile_snippet() -> tuple[bool, int, str, str]:
    """Compile a C++ code snippet.

    Returns:
        Tuple containing:
        - bool: True if compilation succeeded, False otherwise
        - int: Return code from compilation
        - str: stdout from compilation
        - str: stderr from compilation
    """
    compile_process = subprocess.run(
        ['clang++', '-O0', '-g0', '-pipe', '-std=c++17', '-o', str(snippet_exe), str(snippet_cpp)],
        capture_output=True,
        text=True,
        timeout=float(compile_timeout_s),
    )

    compile_success = compile_process.returncode == 0
    return compile_success, compile_process.returncode, compile_process.stdout, compile_process.stderr

def test_snippet(input_str: str, timeout_s: int) -> tuple[int, str, str]:
    """Test a C++ code snippet by running it. Assumes the code has already been compiled to `snippet_exe`.

    Args:
        input_str: Input to provide to the program's stdin
        timeout_s: Timeout in seconds for program execution

    Returns:
        Tuple containing:
        - int: Return code from compilation (if failed) or execution (if succeeded)
        - str: Program stdout (empty string if compilation failed)
        - str: Program stderr or compilation errors
    """

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
        if False:
            try:
                input()
                res = res_success()
                print(json.dumps(res))
                continue
            except EOFError:
                break

        try:
            input_str = input()
        except EOFError:
            break
        except Exception as e:
            res = res_fail_other(details={
                'type': 'Exception',
                'message': f'Exception: {str(e)}'
            })
            print(json.dumps(res))
            break

        try:
            data = json.loads(input_str)
        except Exception as e:
            res = res_fail_other(details={
                'type': 'JSON parse error',
                'message': (
                    f'Exception: {str(e)}\n'
                    f'JSON:\n{input_str}'
                )
            })
            print(json.dumps(res))
            continue

        try:
            code = data['code']
            test_cases = data['test_cases']
            timeout_s = data['timeout_s']
        except Exception as e:
            res = res_fail_other(details={
                'type': 'Error reading fields',
                'message': (
                    f'Exception: {str(e)}\n'
                    f'JSON:\n{input_str}'
                )
            })
            print(json.dumps(res))
            continue

        try:
            snippet_cpp.write_text(code)
        except Exception as e:
            res = res_fail_other(details={
                'type': 'Error writing snippet',
                'message': f'Exception: {str(e)}'
            })
            print(json.dumps(res))
            continue

        try:
            compile_success, compile_exit_code, compile_output, compile_stderr = compile_snippet()
        except Exception as e:
            res = res_fail_other(details={
                'type': 'Error compiling snippet',
                'message': (
                    f'Exception: {str(e)}\n'
                    f'Code:\n{code}'
                )
            })
            print(json.dumps(res))
            continue

        if not compile_success:
            res = res_fail_other(details={
                'type': 'Compilation failed',
                'message': (
                    f'Exit code: {compile_exit_code}\n'
                    f'Stdout:\n{compile_output}\n'
                    f'Stderr:\n{compile_stderr}'
                )
            })
            print(json.dumps(res))
            continue

        res = res_fail_other(details={
            'type': 'No test cases',
            'message': 'No test cases.'
        })

        for test_case in test_cases:
            # print(f"test_case: {test_case}")
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
                    'message': (
                        f'Exception: {str(e)}\n'
                        f'Input:\n{in_str}\n'
                        f'Output:\n{real_output}\n'
                        f'Stderr:\n{stderr}'
                    )
                })
                break
            except Exception as e:
                res = res_fail_other(details={
                    'type': 'Unkown Exception',
                    'message': (
                        f'Exception: {str(e)}\n'
                        f'Input:\n{in_str}\n'
                        f'Output:\n{real_output}\n'
                        f'Stderr:\n{stderr}'
                    )
                })
                break

            if exit_code == -9:  # SIGKILL from timeout
                res = res_fail_timeout(stdout=real_output, stderr=stderr)
                break
            elif exit_code != 0:
                res = res_fail_other(details={
                    'type': 'Non-zero exit code',
                    'message': (
                        f'Exit code: {exit_code}\n'
                        f'Input:\n{in_str}\n'
                        f'Output:\n{real_output}\n'
                        f'Stderr:\n{stderr}'
                    )
                })
                break

            if real_output.rstrip() != expected_output.rstrip():
                res = res_fail_wrong_output(
                    expected=expected_output,
                    got=real_output,
                    stderr=stderr,
                )
                break

            res = res_success(
                stderr=stderr,
            )

        print(json.dumps(res))


if __name__ == '__main__':
    main()
    sys.exit(0)