import asyncio
from pathlib import Path

import ray

from .run_in_containers import ExecutionArgs, ExecutionResultRow
from .agnostics import AgnosticsItem, _AgnosticsSingleContainerWorker


class _RayAgnosticsContainerWorker:
    def __init__(
        self,
        executor_id: int,
        executor_image_name: str,
        execution_args: ExecutionArgs,
        timeout_s: int,
        io_log_dir: Path,
    ):
        self._supervisor = _AgnosticsSingleContainerWorker(
            executor_id=executor_id,
            executor_image_name=executor_image_name,
            execution_args=execution_args,
            timeout_s=timeout_s,
            write_timeout_s=timeout_s,
            io_log_dir=io_log_dir,
        )

    def start_container(self):
        self._supervisor._start_container_if_needed(quiet=True)

    async def run_code(self, item_id: str, item: AgnosticsItem) -> ExecutionResultRow:
        return await self._supervisor.run_code(item_id, item)


@ray.remote
class CodeExecutionActor:
    def __init__(
        self,
        executor_image_name: str,
        worker_num: int,
        io_log_dir: Path,
    ):
        self._lua_queue = asyncio.Queue(maxsize=worker_num)
        self._lua_workers = [
            _RayAgnosticsContainerWorker(
                executor_id=i,
                executor_image_name=executor_image_name,
                execution_args=ExecutionArgs(
                    output_file=Path('results.jsonl'),
                ),
                timeout_s=30,
                io_log_dir=io_log_dir,
            ) for i in range(worker_num)
        ]
        for w in self._lua_workers:
            w.start_container()
            self._lua_queue.put_nowait(w)
        from loguru import logger
        from agnostics.cli.cmd import start_no_ctx
        start_no_ctx(None)
        logger.info(f'Started {worker_num} {executor_image_name!r} containers')

    async def run_code(
        self,
        item_id: str,
        item: AgnosticsItem,
    ) -> ExecutionResultRow:
        worker = await self._lua_queue.get()
        try:
            return await worker.run_code(item_id, item)
        finally:
            self._lua_queue.put_nowait(worker)



