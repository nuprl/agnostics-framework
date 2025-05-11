__all__ = ['Channel', 'run_proc', 'push_items']
import asyncio
from typing import Any


class Channel[T]:
    """
    A type-safe wrapper around asyncio.Queue which allows signalling the end of the channel.

    Once the channel's input is closed, no more items can be added, but items may remain in the queue.

    (Feel free to forward more methods from the underlying queue, but mind the type and `input_closed`.)
    """
    def __init__(self, maxsize: int):
        self._queue = asyncio.Queue(maxsize)
        self._input_closed = False

    def close_input(self):
        self._input_closed = True

    def has_closed_input(self) -> bool:
        return self._input_closed

    async def put(self, item: T):
        assert not self._input_closed
        await self._queue.put(item)

    async def get(self) -> T:
        return await self._queue.get()

    def task_done(self):
        self._queue.task_done()

    def qsize(self) -> int:
        return self._queue.qsize()

    async def join(self):
        await self._queue.join()


async def run_proc(
    *cmd: str,
    input: str | None,
    timeout: float,
    **kwargs: Any,
) -> tuple[asyncio.subprocess.Process, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if input is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input.encode() if input is not None else None),
            timeout=timeout
        )
        return proc, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError as e:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        raise e


async def push_items[X](xs: list[X], ch: Channel[X]) -> None:
    for x in xs:
        await ch.put(x)
    ch.close_input()
