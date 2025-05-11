import json
import subprocess
from pathlib import Path

from container_protocol import *


cwd = Path.cwd()
snippet_rs = cwd / 'snippet.rs'
snippet_exe = cwd / 'snippet'

def normalize_output(s: str) -> list[str]:
    # drop blank lines & trailing spaces
    return [line.rstrip() for line in s.rstrip().splitlines() if line.strip()]

def test_snippet(input_str: str, timeout_s: float):
    # Write & compile the Rust code
    compile = subprocess.run(
        ['rustc', '--edition=2021', '-o', str(snippet_exe), str(snippet_rs)],
        capture_output=True, text=True
    )
    if compile.returncode != 0:
        return compile.returncode, '', compile.stderr

    # Run the binary
    run = subprocess.run(
        [str(snippet_exe)],
        input=input_str, text=True,
        capture_output=True, timeout=timeout_s
    )
    return run.returncode, run.stdout, run.stderr

def main():
    while True:
        try:
            data = json.loads(input())
        except EOFError:
            break

        # Unpack the protocol
        code        = data['code']
        test_cases  = data['test_cases']
        timeout_s   = data['timeout_s']

        # Write source file
        snippet_rs.write_text(code)

        result = None
        for tc in test_cases:
            inp  = tc['input']
            exp  = tc['output']
            try:
                ec, out, err = test_snippet(inp, timeout_s)
            except subprocess.TimeoutExpired:
                result = res_fail_timeout()
                break
            except Exception as e:
                result = res_fail_other(details=str(e))
                break

            # timeout via SIGKILL
            if ec == -9:
                result = res_fail_timeout(stdout=out, stderr=err)
                break
            # compile/runtime error
            if ec != 0:
                result = res_fail_error(exit_code=ec, stderr=err)
                break
            # wrong output
            if normalize_output(out) != normalize_output(exp):
                result = res_fail_wrong_output(expected=exp, got=out, stderr=err)
                break

            result = res_success(stderr=err)

        print(json.dumps(result))

if __name__=='__main__':
    main()
