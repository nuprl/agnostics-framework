from operator import xor
import shutil
from pathlib import Path
import typer
import yaml

app = typer.Typer(help="Generate executor and Codeforces prompt from YAML")


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PROTOCOL = REPO_ROOT / "src" / "agnostics" / "schema" / "container_protocol.py"


def generate_dockerfile(
    out_dir: Path,
    install_command: str | None,
    container_instructions: str | None,
    container_base_image: str | None,
    container_type: str | None,
) -> None:
    if container_base_image:
        assert container_type, "container_type is required if container_base_image is provided"
    assert not all([install_command, container_instructions]), (
        "at most one of install_command, container_instructions must be provided"
    )
    if container_type is not None:
        assert container_type == "debian", "the only supported container type is 'debian'"

    resolved_container_base_image = container_base_image or "ubuntu:22.04"
    resolved_install_instructions = ""
    if install_command:
        resolved_install_instructions = f"RUN {install_command.strip()}"
    elif container_instructions:
        resolved_install_instructions = f"RUN {container_instructions.strip()}"

    dockerfile = f"""\
FROM {resolved_container_base_image}
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install bounded_subprocess==2.3.1
{resolved_install_instructions}
WORKDIR /workdir
COPY ./workdir-template/* ./
ENTRYPOINT ["python3", "/workdir/test_harness.py"]
"""
    (out_dir / "Dockerfile").write_text(dockerfile)


def generate_build_sh(out_dir: Path, lang: str) -> None:
    build_sh = f"""\
#!/usr/bin/env bash
script_dir=$( dirname "$0" )
cd "$script_dir"

executor_image_tag=agnostics-{lang}-executor
"${{AGNOSTICS_CONTAINER_TOOL:-podman}}" build -t "$executor_image_tag" .
"""
    path = out_dir / "build.sh"
    path.write_text(build_sh)
    path.chmod(0o755)


def generate_test_harness(out_dir: Path, filename: str, execute_script: str, compile_script: str | None) -> None:
    harness = f"""\
import json
import subprocess
import shutil
import tempfile
from pathlib import Path

from container_protocol import *

FILENAME = {repr(filename)}
COMPILE_CMD = {repr(compile_script) if compile_script else 'None'}
EXECUTE_CMD = {repr(execute_script)}


def run_cmd(cmd: str, cwd: str, *, input_str: str | None = None, timeout_s: int = 1):
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        input=input_str,
        text=True,
        capture_output=True,
        timeout=float(timeout_s),
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_compile_cmd(
    cwd: str,
    timeout_s: int,
) -> dict | None:
    if not COMPILE_CMD:
        return None
    try:
        rc, out, err = run_cmd(COMPILE_CMD, cwd=cwd, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        return res_fail_timeout()
    if rc != 0:
        return res_fail_error(exit_code=rc, stdout=out, stderr=err)
    return None


def run_test_cases(
    test_cases: list[dict],
    cwd: str,
    timeout_s: int,
):
    res = None
    for tc in test_cases:
        try:
            rc, out, err = run_cmd(EXECUTE_CMD, cwd=cwd, input_str=tc['input'], timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            return res_fail_timeout()
        if rc != 0:
            return res_fail_error(exit_code=rc, stdout=out, stderr=err)
        if out.rstrip() != tc['output'].rstrip():
            return res_fail_wrong_output(expected=tc['output'], got=out, stderr=err)
        res = res_success(stderr=err)
    return res


def main():
    while True:
        try:
            data = json.loads(input())
        except EOFError:
            break
        code = data['code']
        test_cases = data['test_cases']
        timeout_s = data['timeout_s']
        with tempfile.TemporaryDirectory() as tmp:
            snippet_path = Path(tmp)/FILENAME
            snippet_path.write_text(code)
            res = run_compile_cmd(cwd=tmp, timeout_s=timeout_s)
            if res is None:
                res = run_test_cases(test_cases=test_cases, cwd=tmp, timeout_s=timeout_s)
            print(json.dumps(res))

if __name__ == '__main__':
    main()
"""
    workdir = out_dir / "workdir-template"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "test_harness.py").write_text(harness)
    shutil.copy(TEMPLATE_PROTOCOL, workdir / "container_protocol.py")


def generate_cli_module(lang: str, prompt_suffix: str) -> str:
    return f"""\
import typer

from .common import commands

app = typer.Typer()


PROMPT_SUFFIX = '''\\
{prompt_suffix.strip()}

Put your final answer in a single code block:
```
<your code here>
```
'''

executor_image_name = 'agnostics-{lang}-executor'
command_make_prompts = commands.CmdStandardMakePrompts(__file__, prompt_pl_suffix=PROMPT_SUFFIX)
command_generate = commands.CmdGenerate(__file__)
command_verify = commands.CmdVerifyAnswers(executor_image_name=executor_image_name)

for cmd in (command_make_prompts, command_generate, commands.CmdExtractAnswersInstance, command_verify):
    cmd.add_commands(app)


if __name__ == '__main__':
    app()
"""


@app.command()
def main(config: Path):
    cfg = yaml.safe_load(config.read_text())
    filename = cfg['filename']
    prompt = cfg['prompt']

    install_command = None
    container_instructions = None
    install_raw = cfg.get('install')
    if install_raw is None:
        pass
    elif type(install_raw) == str:
        install_command = install_raw.strip()
        install_nl_count = install_command.count('\n')
        assert install_nl_count == 0, (
            f"only single-line 'install' commands are supported; consider using the container-instructions form.")
    elif type(install_raw) == dict:
        assert list(install_raw.keys()) == ['container-instructions'], (
            f"unexpected keys in 'install' dict: {set(install_raw.keys()) - {'container-instructions'}}")
        container_instructions = install_raw['container-instructions']
    else:
        raise ValueError(f"unexpected type of 'install' field: {type(install_raw)}")

    container_base_image = None
    container_type = None
    container_raw = cfg.get('container')
    if container_raw is not None:
        assert set(container_raw.keys()) == {'base-image', 'type'}, (
            f"unexpected keys in 'container' dict: {set(container_raw.keys()) - {'base-image', 'type'}}")
        container_base_image = container_raw['base-image']
        container_type = container_raw['type']
        assert container_type == "debian", f"the only currently supported container type is 'debian', but got: {container_type}"

    execute = cfg['execute']
    compile_cmd = cfg.get('compile')
    lang = config.stem

    executor_dir = REPO_ROOT / 'executors' / lang
    executor_dir.mkdir(parents=True, exist_ok=True)
    generate_dockerfile(executor_dir, install_command, container_instructions, container_base_image, container_type)
    generate_build_sh(executor_dir, lang)
    generate_test_harness(executor_dir, filename, execute, compile_cmd)

    cli_module = generate_cli_module(lang, prompt)
    cli_path = REPO_ROOT / 'src' / 'agnostics' / 'cli' / 'codeforces_cots' / f'{lang}.py'
    cli_path.write_text(cli_module)

    print(f'Generated executor in {executor_dir.relative_to(REPO_ROOT)}')
    print(f'Generated CLI module at {cli_path.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    app()
