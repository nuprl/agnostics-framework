from typing import Any


def res_success(
    stderr: str = '',
):
    return {
        'result': 'success',
        'stderr': stderr,
    }


def fail_finegrained(
    subresults: list[dict],
):
    """
    Used by fine-grained test harnesses to report multiple results with at least one failure.
    (If no result is a failure, result:success should be used.)

    The number of subresults must match the number of test cases.
    """
    return {
        'result': 'fail:finegrained',
        'subresults': subresults,
    }


def res_fail_wrong_output(
    expected: str,
    got: str,
    stderr: str = '',
):
    """
    Indicates that snippet execution returned a wrong output on a test case.
    """
    return {
        'result': 'fail:wrong-output',
        'expected': expected,
        'got': got,
        'stderr': stderr,
    }


def res_fail_error(
    exit_code: int,
    stdout: str = '',
    stderr: str = '',
):
    """
    Indicates that snippet execution exited with an error on a test case.

    May be used for compilation errors, runtime errors, etc.
    """
    return {
        'result': 'fail:error',
        'stdout': stdout,
        'stderr': stderr,
        'exit_code': exit_code,
    }


def res_fail_timeout(
    stdout: str = '',
    stderr: str = '',
):
    """
    Indicates that snippet execution timed out on a test case.
    """
    return {
        'result': 'fail:timeout',
        'stdout': stdout,
        'stderr': stderr,
    }


def res_fail_other(
    stdout: str = '',
    stderr: str = '',
    details: Any = None,
):
    res = {
        'result': 'fail:other',
        'stdout': stdout,
        'stderr': stderr,
    }
    if details is not None:
        res['details'] = details
    return res