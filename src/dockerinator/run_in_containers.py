__all__ = ['ItemAndId', 'ExecutionArgs', 'ExecutionResultRow', 'Supervisor', 'run_supervisor']
from abc import ABCMeta, abstractmethod
import asyncio
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Iterator, Literal

from loguru import logger

from . import agnostics_cfg
from .agnostics_cfg import ContainerTool
from .docker_utils import running_container_count


@dataclass
class ItemAndId[X]:
    """
    Pairs an item to be executed with an ID, used for logging and reporting.
    """
    id: str
    item: X


@dataclass
class ExecutionArgs:
    container_tool: ContainerTool | None = None
    tool_subcommand: list[str] = field(default_factory=list)
    executor_args: list[str] = field(default_factory=list)
    max_workers: int | None = None
    timeout_s: int | None = None
    output_file: Path = Path('exec-result.json')



def resolve_max_workers(max_workers: int | None) -> int:
    res = max_workers
    if res in (None, 0):
        res = max(1, cpu_count() - 1)
        if max_workers is None:
            res = min(res, 16)
    return res


@dataclass
class ExecutionResultRow():
    """
    Supervisors should use this class to log results.
    """
    item_id: str
    status: str = 'success'
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def success(item_id: str, stdout: str, stderr: str) -> 'ExecutionResultRow':
        return ExecutionResultRow(item_id, 'success', {
            'stdout': stdout,
            'stderr': stderr,
        })

    @staticmethod
    def fail_nonzero_exit(item_id: str, stdout: str, stderr: str) -> 'ExecutionResultRow':
        return ExecutionResultRow(item_id, 'fail:nonzero-exit', {
            'stdout': stdout,
            'stderr': stderr,
        })

    @staticmethod
    def fail_timeout(item_id: str, timeout_s: int, details: str | None = None) -> 'ExecutionResultRow':
        extra = {
            'timeout-s': timeout_s,
        }
        if details:
            extra['details'] = details
        return ExecutionResultRow(item_id, 'fail:timeout', extra)

    def to_jsonable(self) -> dict:
        res = {
            'item_id': self.item_id,
            'status': self.status,
        }
        res.update(self.extra)
        return res


class Supervisor[X](metaclass=ABCMeta):
    """
    A supervisor executes items using a pool of workers and writes the results to the output file.
    """
    @abstractmethod
    async def process_items(
        self,
        items: list[ItemAndId[X]],
        max_workers: int,
    ) -> None:
        ...


def run_supervisor[X](
    executor_image_name: str,
    items: list[ItemAndId[X]],
    supervisor: Supervisor[X],
    execution_args: ExecutionArgs,
):
    """
    Runs a supervisor on the given items.
    """
    real_max_workers = resolve_max_workers(execution_args.max_workers)
    if execution_args.max_workers in (None, 0):
        logger.info('Defaulting worker count to: {}', real_max_workers)

    async def async_entrypoint():
        using_apptainer = (
            execution_args.container_tool or agnostics_cfg.get_container_tool()
        ) == 'apptainer'
        if not using_apptainer:
            running_containers = await running_container_count(executor_image_name)
            if running_containers > 0:
                print(f'Expected 0 running {executor_image_name!r} containers, but got: {running_containers}')
                confirm = input('This may cause issues (or at least will spam the logs). Continue? (y/N) ')
                if confirm and confirm.lower() != 'y':
                    return

        await supervisor.process_items(items, real_max_workers)

    asyncio.run(async_entrypoint())