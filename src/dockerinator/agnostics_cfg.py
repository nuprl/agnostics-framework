"""
This module has project-specific configuration resolution for the dockerinator package.
"""
import os
from typing import Literal

import dotenv
from loguru import logger
import typer


type ContainerTool = Literal['docker', 'podman', 'podman-hpc', 'apptainer', 'srun']


_container_tool: ContainerTool | None = None
def get_container_tool() -> ContainerTool:
    global _container_tool
    if _container_tool is not None:
        return _container_tool

    _load_dotenv()
    env_var = 'AGNOSTICS_CONTAINER_TOOL'
    allowed_values = ('docker', 'podman', 'podman-hpc', 'apptainer', 'srun')
    tool = os.getenv(env_var)
    if tool is None:
        tool = 'podman'
        logger.info('${} not set, defaulting the container tool to: {}', env_var, tool)
    elif tool not in allowed_values:
        logger.error('{} must be one of {}; got: {}', env_var, ', '.join(allowed_values), tool)
        raise typer.Exit(1)

    _container_tool = tool
    return _container_tool


_loaded_dotenv = False
def _load_dotenv():
    global _loaded_dotenv
    if _loaded_dotenv:
        return
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    _loaded_dotenv = True
