__all__ = ['ExecutorLauncher', 'SimpleSupervisor', 'ContainerExecutionSpec', 'VolumeSpec', 'OneshotContainerItemExecutor']
from abc import ABCMeta, abstractmethod
import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, TextIO

from loguru import logger
from tqdm import tqdm

from .async_utils import run_proc
from .docker_utils import wait_for_running_containers, kill_container
from .run_in_containers import Supervisor, ItemAndId, ExecutionArgs, ExecutionResultRow
from .agnostics_resolve_container_tool_command import resolve_container_tool_command

_DEFAULT_TIMEOUT_S = 60


class ExecutorLauncher[X](metaclass=ABCMeta):
    """ An interface to name the method params & allow currying via the constructor. """
    @abstractmethod
    async def exec_item(
        self,
        item_id: str,
        item: X,
        max_workers: int,
    ) -> ExecutionResultRow:
        ...


class SimpleSupervisor[X](Supervisor[X]):
    def __init__(
        self,
        executor_launcher: ExecutorLauncher[X],
        execution_args: ExecutionArgs,
    ):
        self._executor_launcher = executor_launcher
        self._args = execution_args

    async def process_items(
        self,
        items: list[ItemAndId[X]],
        max_workers: int,
    ):
        sem = asyncio.Semaphore(max_workers)
        write_sem = asyncio.Semaphore(1)

        result_f = self._args.output_file
        with result_f.open('w') as fh:
            tasks = [asyncio.create_task(self._process_item_safely(
                item_id=it.id,
                item=it.item,
                max_workers=max_workers,
                result_fh=fh,
                sem=sem,
                write_sem=write_sem,
            )) for it in items]
            with tqdm(total=len(tasks), desc='Processing items') as pbar:
                for task in asyncio.as_completed(tasks):
                    await task
                    pbar.update(1)


    async def _process_item_safely(
        self,
        item_id: str,
        item: X,
        max_workers: int,
        result_fh: TextIO,
        sem: asyncio.Semaphore,
        write_sem: asyncio.Semaphore,
    ):
        exceptions = []
        async with sem:
            try:
                res = await self._executor_launcher.exec_item(item_id, item, max_workers=max_workers)
            except Exception as e:
                res = ExecutionResultRow(item_id, 'fail:exception', {'details': str(e)})
                exceptions.append(e)
        async with write_sem:
            try:
                print(json.dumps(res.to_jsonable()), file=result_fh)
                result_fh.flush()
            except Exception as e:
                exceptions.append(e)
            for e in exceptions:
                try:
                    logger.opt(exception=e).error('Error when processing item: {!r}', item_id)
                except:
                    pass


@dataclass
class ContainerExecutionSpec:
    stdin_source: str | Path | None
    volumes: list['VolumeSpec'] = field(default_factory=lambda: [])


@dataclass
class VolumeSpec:
    local_volume_path: Path
    mount_path: str


class OneshotContainerItemExecutor[X](ExecutorLauncher[X]):
    """
    Runs a single "item" in a container.

    An "item" is anything that can be passed to the `prepare_item` callback.
    The callback can set up the environment as needed before the container is run,
    and it returns a `ContainerExecutionSpec` that describes how to pass the item
    to the container.

    For examples of how to use this class, see `simple_run_in_containers.py`.
    """
    def __init__(
        self,
        executor_image_name: str,
        prepare_item: Callable[[X, Path | None], ContainerExecutionSpec],
        make_tmpdir: bool,
        execution_args: ExecutionArgs,
    ):
        self._executor_image_name = executor_image_name
        self._prepare_item = prepare_item
        self._make_tmpdir = make_tmpdir
        self._execution_args = execution_args

        if t := self._execution_args.timeout_s:
            timeout_s = t
        else:
            timeout_s = _DEFAULT_TIMEOUT_S
            logger.info('Defaulting timeout to {}s', timeout_s)
        self._timeout_s = timeout_s


    async def exec_item(
        self,
        item_id: str,
        item: X,
        max_workers: int,
    ) -> ExecutionResultRow:
        await wait_for_running_containers(
            image_name=self._executor_image_name,
            max_containers=max_workers,
            sleep_secs=10,
        )

        tmpdir = None
        if self._make_tmpdir:
            tmpdir = tempfile.mkdtemp()
        try:
            spec = self._prepare_item(item, Path(tmpdir) if tmpdir else None)
        except Exception as e:
            logger.exception('Exception when preparing item: {}', item_id)
            return ExecutionResultRow(item_id, 'fail:preparation-exception', {'error': str(e)})

        cidfile = tempfile.mktemp()
        try:
            subcommand_args = ['--cidfile', cidfile]
            for v in spec.volumes:
                subcommand_args.append('--volume')
                subcommand_args.append(f'{v.local_volume_path}:{v.mount_path}')
            command = resolve_container_tool_command(
                tool=self._execution_args.container_tool,
                tool_subcommand=self._execution_args.tool_subcommand,
                tool_subcommand_args=subcommand_args,
                executor_image_name=self._executor_image_name,
                executor_args=self._execution_args.executor_args,
            )
            in_src = spec.stdin_source
            if isinstance(in_src, Path):
                input = in_src.read_text()
            elif isinstance(in_src, str):
                input = in_src
            else:
                input = None
            proc, stdout, stderr = await run_proc(
                *command,
                input=input,
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            await kill_container(cidfile)
            return ExecutionResultRow.fail_timeout(item_id, timeout_s=self._timeout_s)
        finally:
            if os.path.exists(cidfile):
                os.unlink(cidfile)
            if tmpdir:
                shutil.rmtree(tmpdir)

        if proc.returncode != 0:
            return ExecutionResultRow.fail_nonzero_exit(item_id, stdout=stdout, stderr=stderr)
        return ExecutionResultRow.success(item_id, stdout=stdout, stderr=stderr)
