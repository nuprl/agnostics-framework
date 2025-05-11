__all__ = ['kill_container', 'running_container_count', 'wait_for_running_containers']
import asyncio
from pathlib import Path
import re

from loguru import logger

from .async_utils import run_proc
from . import agnostics_cfg

async def kill_container(cidfile: str):
    tool = agnostics_cfg.get_container_tool()
    cid = '???'
    timeout_s = 10
    try:
        cid = Path(cidfile).read_text()
        proc, stdout, stderr = await run_proc(
            tool, 'kill',
            '--signal', 'SIGKILL',
            cid,
            input=None,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            logger.warning(
                'Failed to kill container cid={}; {} kill returned {}\n#STDOUT#\n{}\n#STDERR#\n{}',
                cid, tool, proc.returncode, stdout, stderr,
            )
        else:
            await asyncio.sleep(1)
    except asyncio.TimeoutError:
        logger.warning('Failed to kill container cid={}; kill command took more than {}s', cid, timeout_s)
    except Exception:
        logger.exception('Failed to kill container cid={}', cid)


async def running_container_count(image_name: str) -> int:
    tool = agnostics_cfg.get_container_tool()
    proc, stdout, stderr = await run_proc(
        tool, 'ps',
        '-q',
        '--filter', f'ancestor={image_name}',
        input='', timeout=10)
    assert proc.returncode == 0, \
        f'{tool} ps failed with {proc.returncode}: \n#STDOUT#\n{stdout}\n#STDERR#\n{stderr}'
    return len(re.findall(r'\w\b', stdout))


async def wait_for_running_containers(*, image_name: str, max_containers: int, sleep_secs: int):
    while True:
        try:
            running_containers = await running_container_count(image_name)
            if running_containers < max_containers:
                break
            else:
                logger.error(
                    'Expected <{} running {} containers, but got {}; will re-count in {}s.',
                    max_containers,
                    image_name,
                    running_containers,
                    sleep_secs,
                )
        except Exception:
            logger.exception('Failed to get running container count, will retry in {} seconds', sleep_secs)
        await asyncio.sleep(sleep_secs)
