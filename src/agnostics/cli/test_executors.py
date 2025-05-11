from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Callable

from loguru import logger
import typer

from dockerinator.agnostics import resolve_container_tool_command
from . import cmd

app = typer.Typer()


def run_one_test(
    test_dir: Path,
    command: list[str],
    expected_result: str,
    repeats: int = 1
):
    test_dir_rel = cmd.cwd_rel(test_dir)
    snippet_candidates = list(test_dir.glob('snippet.*'))
    if len(snippet_candidates) != 1:
        logger.error('Expected exactly one file named "snippet" in {}, found {}', test_dir, snippet_candidates)
        return
    snippet_path = snippet_candidates[0]
    snippet = snippet_path.read_text()

    test_cases_path = test_dir / 'test-cases.json'
    test_cases = json.loads(test_cases_path.read_text())

    config_path = test_dir / 'config.json'
    config = json.loads(config_path.read_text())

    request_obj = {
        'code': snippet,
        'test_cases': test_cases,
        'timeout_s': config['timeout_s'],
    }

    proc = subprocess.run(
        command,
        input='\n'.join(json.dumps(request_obj) for _ in range(repeats)),
        text=True,
        capture_output=True,
        timeout=float(config['timeout_s'])*(repeats+1),
    )

    if proc.returncode != 0:
        logger.error(
            'In test {!r}, unexpected exit code: {}\n## STDOUT ##\n{}{}{}',
            test_dir_rel,
            proc.returncode,
            proc.stdout,
            '\n## STDERR ##\n' if proc.stderr else '',
            proc.stderr,
        )
        return

    try:
        responses = [json.loads(line) for line in proc.stdout.splitlines()]
    except:
        logger.error(
            'In test {!r}, malformed JSON on stdout: {}{}{}',
            test_dir_rel,
            proc.stdout,
            '\n## STDERR ##\n' if proc.stderr else '',
            proc.stderr,
        )
        return

    for resp in responses:
        if resp['result'] != expected_result:
            logger.error(
                'In test {!r}, unexpected response result: {}\n## STDOUT ##\n{}{}{}',
                test_dir_rel,
                resp['result'],
                proc.stdout,
                '\n## STDERR ##\n' if proc.stderr else '',
                proc.stderr,
            )
            return

    logger.success('Test passed: {}', test_dir_rel)


@dataclass
class ImageTestCase:
    test_dir: Path
    expected_result: str
    repeats: int


def run_tests(
    tests_dir: Path,
    run_one_test_fn: Callable[[ImageTestCase], None],
):
    image_test_cases = [
        ImageTestCase(
            test_dir=tests_dir/'fail-error',
            expected_result='fail:error',
            repeats=3,
        ),
        ImageTestCase(
            test_dir=tests_dir/'fail-wrong-output',
            expected_result='fail:wrong-output',
            repeats=3,
        ),
        ImageTestCase(
            test_dir=tests_dir/'timeout',
            expected_result='fail:timeout',
            repeats=3,
        ),
        ImageTestCase(
            test_dir=tests_dir/'success',
            expected_result='success',
            repeats=3,
        ),
    ]

    for test_case in image_test_cases:
        if not test_case.test_dir.exists():
            logger.warning('Skipping absent dir: {}', cmd.cwd_rel(test_case.test_dir))
            continue

        run_one_test_fn(test_case)


@app.command()
def test_harness(
    executor_definition_dir: Path,
    harness_command: list[str],
):
    tests_dir = executor_definition_dir / 'tests'
    assert tests_dir.exists()

    def _run_test(test_case: ImageTestCase):
        run_one_test(test_case.test_dir, harness_command, test_case.expected_result, test_case.repeats)

    run_tests(tests_dir, _run_test)


@app.command()
def test_one(
    executor_definition_dir: Path,
    executor_image_name: str,
):
    tests_dir = executor_definition_dir / 'tests'
    assert tests_dir.exists()

    command = resolve_container_tool_command(
        tool=None,
        tool_subcommand=[],
        tool_subcommand_args=[],
        executor_image_name=executor_image_name,
        executor_args=[],
    )

    def _run_test(test_case: ImageTestCase):
        run_one_test(test_case.test_dir, command, test_case.expected_result, test_case.repeats)

    run_tests(tests_dir, _run_test)


@app.command()
def test_all():
    import os

    cmd.start_no_ctx(None)

    repo_root = os.getenv('REPO_ROOT')
    if repo_root:
        repo_root_d = Path(repo_root)
    else:
        logger.info('Assuming we are running from the repo root...')
        repo_root_d = Path.cwd()

    executors_dir = repo_root_d/'executors'
    test_one(executors_dir/'cpp', 'agnostics-cpp-executor')
    test_one(executors_dir/'python', 'agnostics-python-executor')
    test_one(executors_dir/'lua', 'agnostics-lua-executor')
    test_one(executors_dir/'julia', 'agnostics-julia-executor')
    test_one(executors_dir/'r', 'agnostics-r-executor')
    test_one(executors_dir/'c', 'agnostics-c-executor')


if __name__ == '__main__':
    app()
