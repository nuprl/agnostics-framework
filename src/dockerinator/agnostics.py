__all__ = ['AgnosticsItem', 'AgnosticsTestCase', 'run_agnostics_items_in_oneshots', 'run_agnostics_items', 'AgnosticsContainerSupervisor']
import json
import os
from pathlib import Path
import pydantic
import asyncio
import tempfile
from typing import IO, Any, Awaitable, Callable, Iterator, Literal, Never, NotRequired, TypedDict, assert_never

from loguru import logger
from bounded_subprocess.interactive_async import Interactive
from tqdm import tqdm
from typeguard import typechecked

from . import agnostics_cfg
from .async_utils import Channel
from .docker_utils import kill_container
from .run_in_containers import ItemAndId, Supervisor, ExecutionArgs, ExecutionResultRow, run_supervisor
from .simple_supervisor import ContainerExecutionSpec, OneshotContainerItemExecutor, SimpleSupervisor
from .agnostics_resolve_container_tool_command import resolve_container_tool_command


class AgnosticsItem(pydantic.BaseModel):
    code: str
    test_cases: list['AgnosticsTestCase']
    lang: str | None = None


class AgnosticsTestCase(pydantic.BaseModel):
    input: str
    output: str


def run_agnostics_items(
    executor_image_name: str,
    inputs: list[ItemAndId[AgnosticsItem]],
    container_io_log_dir: Path,
    execution_args: ExecutionArgs,
):
    container_io_log_dir.mkdir(parents=True, exist_ok=True)
    for f in container_io_log_dir.iterdir():
        f.unlink()

    supervisor = AgnosticsContainerSupervisor(
        executor_image_name=executor_image_name,
        args=execution_args,
        io_log_dir=container_io_log_dir,
    )

    run_supervisor(
        executor_image_name=executor_image_name,
        items=inputs,
        supervisor=supervisor,
        execution_args=execution_args,
    )


def run_agnostics_items_in_oneshots(
    executor_image_name: str,
    inputs: list[ItemAndId[AgnosticsItem]],
    execution_args: ExecutionArgs,
):
    def prep_input(item: AgnosticsItem, tmpdir: Path | None) -> ContainerExecutionSpec:
        jsonable_dict = item.model_dump(mode='json')
        jsonable_dict['timeout_s'] = execution_args.timeout_s or _DEFAULT_TIMEOUT_S
        return ContainerExecutionSpec(
            stdin_source=json.dumps(jsonable_dict),
        )

    item_executor = OneshotContainerItemExecutor(
        executor_image_name=executor_image_name,
        prepare_item=prep_input,
        make_tmpdir=False,
        execution_args=execution_args,
    )
    supervisor = SimpleSupervisor(
        executor_launcher=item_executor,
        execution_args=execution_args,
    )
    return run_supervisor(
        executor_image_name,
        inputs,
        supervisor,
        execution_args=execution_args,
    )


def _run_supervisor_with_item_gen[X](
    executor_image_name: str,
    item_gen: Iterator[ItemAndId[X]],
    gen_size: int,
    supervisor: 'AgnosticsContainerSupervisor',
    execution_args: ExecutionArgs,
    resume: bool,
):
    """
    Runs a supervisor on the given items.
    """
    from .run_in_containers import resolve_max_workers, running_container_count
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

        await supervisor._process_item_gen(
            item_gen=item_gen,
            gen_size=gen_size,
            max_workers=real_max_workers,
            resume=resume,
        )

    asyncio.run(async_entrypoint())

_DEBUG_CONTAINER_INPUT_LOG = False
_READ_FAILURE_RETRY = 3
_DEFAULT_TIMEOUT_S = 30


class _Retry(Exception):
    """Raised internally to indicate an operation should be retried."""
    pass


class _AgnosticsSingleContainerWorker:
    _proc: Interactive | None
    _cidfile: str | None
    _container_freshness: None | Literal['fresh', 'used']

    def __init__(
        self,
        executor_id: int,
        executor_image_name: str,
        execution_args: ExecutionArgs,
        timeout_s: int,
        write_timeout_s: int,
        io_log_dir: Path | None,
    ):
        self._executor_id = executor_id
        self._executor_image_name = executor_image_name
        self._execution_args = execution_args
        self._timeout_s = timeout_s
        self._write_timeout_s = write_timeout_s
        self._io_log_dir = io_log_dir
        container_tool = self._execution_args.container_tool or agnostics_cfg.get_container_tool()
        self._using_apptainer = container_tool == 'apptainer' or container_tool == 'srun'

        self._proc = None
        self._cidfile = None
        self._container_freshness = None

        self._dbg_out_log: IO[str] | None = None
        self._dbg_in_log: IO[str] | None = None

        if not os.getenv('AGNOSTICS_NO_CONTAINER_IO_LOGS') and io_log_dir:
            self._dbg_out_log = open(io_log_dir / f'container-{executor_id}.log', 'a')
            if _DEBUG_CONTAINER_INPUT_LOG:
                self._dbg_in_log = open(io_log_dir / f'container-input-{executor_id}.log', 'a')

    async def kill_container(self):
        if self._proc is None:
            return
        if self._using_apptainer:
            self._proc._popen.kill()
            return
        assert self._cidfile
        try:
            await kill_container(self._cidfile)
        except Exception:
            logger.exception('Failed to kill the container.')
        finally:
            self._proc = self._cidfile = None

    def _start_container_if_needed(self, quiet: bool = False):
        if self._proc is not None:
            return

        tool_extra_args = []
        if not self._using_apptainer:
            self._cidfile = cidfile = tempfile.mktemp()
            tool_extra_args.append('--cidfile')
            tool_extra_args.append(cidfile)
        command = resolve_container_tool_command(
            tool=self._execution_args.container_tool,
            tool_subcommand=self._execution_args.tool_subcommand,
            tool_subcommand_args=tool_extra_args,
            executor_image_name=self._executor_image_name,
            executor_args=self._execution_args.executor_args,
        )
        self._proc = Interactive(
            command,
            read_buffer_size=1024*1024*5
        )
        if not quiet:
            logger.info(f'(Re)started container {self._executor_image_name}')
        self._container_freshness = 'fresh'

    async def _run_code_without_retry(
        self,
        item_id: str,
        item: AgnosticsItem,
    ) -> ExecutionResultRow:
        assert self._proc
        timeout = self._timeout_s
        write_timeout = self._write_timeout_s
        jsonable_item = item.model_dump(mode='json')
        jsonable_item['timeout_s'] = timeout
        input_ = json.dumps(jsonable_item) + '\n'
        if fh := self._dbg_in_log:
            fh.write(input_)
            fh.flush()
        if not await self._proc.write(
            input_.encode(errors='ignore'),
            timeout_seconds=write_timeout,
        ):
            await self.kill_container()
            if self._container_freshness == 'fresh':
                return ExecutionResultRow(item_id, 'fail:input-timeout', {
                    'timeout-s': write_timeout,
                })
            else:
                raise _Retry()

        response_line_bytes = await self._proc.read_line(timeout_seconds=timeout)
        if response_line_bytes is None:
            await self.kill_container()
            return ExecutionResultRow.fail_timeout(item_id, timeout_s=timeout)
        response, decoded_str = _robust_json_read(response_line_bytes)

        if response:
            log_entry = response
        else:
            log_entry = 'decoding-error'
        if fh := self._dbg_out_log:
            print(json.dumps({'_item_id': item_id, '_item': jsonable_item, **log_entry}), file=fh)
            fh.flush()

        if not response:
            await self.kill_container()
            raise _Retry()

        response_extra = response.copy()
        del response_extra['result']

        return ExecutionResultRow(item_id, response['result'], response_extra)


    async def run_code(
        self,
        item_id: str,
        item: AgnosticsItem,
    ) -> ExecutionResultRow:
        '''
        Send a code snippet to the container and receives the execution result.
        Communication is with JSON using stdin and stdout.
        '''
        for _ in range(_READ_FAILURE_RETRY):
            self._start_container_if_needed()
            try:
                return await self._run_code_without_retry(item_id, item)
            except _Retry:
                continue
            finally:
                self._container_freshness = 'used'

        return ExecutionResultRow(item_id, 'fail:run-failed')

    async def process_channel(
        self,
        input_ch: Channel[ItemAndId[AgnosticsItem]],
        result_cb: Callable[[ExecutionResultRow], Awaitable[None]],
    ):
        while not (input_ch.has_closed_input() and input_ch.qsize() == 0):
            try:
                item = await asyncio.wait_for(input_ch.get(), timeout=5)
            except asyncio.TimeoutError:
                continue
            result = await self.run_code(item.id, item.item)
            input_ch.task_done()
            await result_cb(result)
        await self.kill_container()


class AgnosticsContainerSupervisor(Supervisor[AgnosticsItem]):
    def __init__(
        self,
        executor_image_name: str,
        args: ExecutionArgs,
        io_log_dir: Path | None,
    ):
        self._executor_image_name = executor_image_name
        self._execution_args = args
        self._io_log_dir = io_log_dir

    async def process_items(
        self,
        items: list[ItemAndId[AgnosticsItem]],
        max_workers: int,
    ) -> None:
        input_ch = Channel[ItemAndId[AgnosticsItem]](maxsize=max_workers)

        if t := self._execution_args.timeout_s:
            timeout_s = t
        else:
            timeout_s = _DEFAULT_TIMEOUT_S
            logger.info('Defaulting timeout to {}s', timeout_s)

        result_sem = asyncio.Semaphore(1)
        result_f = self._execution_args.output_file
        with result_f.open('w') as result_fh:
            async def result_cb(result: ExecutionResultRow):
                async with result_sem:
                    print(json.dumps(result.to_jsonable()), file=result_fh)
                    result_fh.flush()

            tasks: list[asyncio.Task] = []
            for i in range(max_workers):
                subsupervisor = _AgnosticsSingleContainerWorker(
                    executor_id=i,
                    executor_image_name=self._executor_image_name,
                    execution_args=self._execution_args,
                    timeout_s=timeout_s,
                    write_timeout_s=timeout_s,
                    io_log_dir=self._io_log_dir,
                )
                tasks.append(asyncio.create_task(
                    subsupervisor.process_channel(input_ch, result_cb)))

            for it in tqdm(items, desc='Processing items'):
                await input_ch.put(it)
            input_ch.close_input()
            logger.info('The last items are being processed...')
            await input_ch.join()
            logger.info('The input queue was processed (# of items = {}), awaiting the tasks...', len(items))
            for task in tasks:
                await task

    async def _process_item_gen(
        self,
        item_gen: Iterator[ItemAndId[AgnosticsItem]],
        gen_size: int,
        max_workers: int,
        resume: bool,
    ) -> None:
        input_ch = Channel[ItemAndId[AgnosticsItem]](maxsize=max_workers)

        if t := self._execution_args.timeout_s:
            timeout_s = t
        else:
            timeout_s = _DEFAULT_TIMEOUT_S
            logger.info('Defaulting timeout to {}s', timeout_s)

        write_timeout_s = 300
        logger.info('Assuming this is a LiveCodeBenchX run.')
        logger.info('Setting write timeout to: {}s', write_timeout_s)

        result_sem = asyncio.Semaphore(1)
        result_f = self._execution_args.output_file
        already_processed_ids = set()
        skipped_items = 0
        if resume and result_f.exists():
            from agnostics.cli.cmd import ser
            for r in ser.jsonl_streamf(result_f):
                already_processed_ids.add(r['item_id'])
            logger.info('Resuming from the existing output file. # of already processed items: {}', len(already_processed_ids))

        with result_f.open('a' if resume else 'w') as result_fh:
            async def result_cb(result: ExecutionResultRow):
                async with result_sem:
                    print(json.dumps(result.to_jsonable()), file=result_fh)
                    result_fh.flush()

            tasks: list[asyncio.Task] = []
            for i in range(max_workers):
                subsupervisor = _AgnosticsSingleContainerWorker(
                    executor_id=i,
                    executor_image_name=self._executor_image_name,
                    execution_args=self._execution_args,
                    timeout_s=timeout_s,
                    write_timeout_s=write_timeout_s,
                    io_log_dir=self._io_log_dir,
                )
                tasks.append(asyncio.create_task(
                    subsupervisor.process_channel(input_ch, result_cb)))

            for it in tqdm(item_gen, desc='Processing items', total=gen_size):
                if resume and it.id in already_processed_ids:
                    skipped_items += 1
                    continue
                await input_ch.put(it)
            input_ch.close_input()
            logger.info('The last items are waiting to be processed...')
            await input_ch.join()
            logger.info(
                'The input queue was processed (# of items = {}{}{}), awaiting the tasks...',
                gen_size,
                ', # of skipped items: ' if resume else '',
                str(skipped_items) if resume else '',
            )
            for task in tasks:
                await task


def _robust_json_read(
    response_line: bytes
) -> tuple[dict|None, str]:
    decoded_response_line = response_line.decode(errors='ignore')
    try:
        return json.loads(decoded_response_line), decoded_response_line
    except:
        return None, decoded_response_line