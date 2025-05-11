"""
This module defines a function to resolve the container tool command.

It's an agnostics module since the apptainer-specific code is specific to this project.
Also, it needs to be a separate module from `dockerinator.agnostics` to avoid circular imports.
"""
from pathlib import Path

from . import agnostics_cfg


_DEFAULT_APPTAINER_SYSTEMD_COMMAND = [
    'systemd-run', '--user', '--scope',
    '-p', 'MemoryMax=1G',
    '--',
]
_DEFAULT_CONTAINER_SUBCOMMAND = ['run', '--rm', '-i', '--oom-score-adj', '1000', '--memory', '1G', '--tmpfs', '/ramdisk:size=128m,exec']
_DEFAULT_APPTAINER_SUBCOMMAND = ['run', '--contain', '--writable-tmpfs']


def resolve_container_tool_command(
    *,
    tool: agnostics_cfg.ContainerTool | None,
    tool_subcommand: list[str] = [],
    tool_subcommand_args: list[str] = [],
    executor_image_name: str,
    executor_args: list[str] = [],
) -> list[str]:
    """
    Resolves the container tool command to use based on the provided defaults.

    Returns a new unaliased list.
    """
    tool = tool or agnostics_cfg.get_container_tool()
    res = [tool]
    if tool_subcommand:
        res.extend(tool_subcommand)
    else:
        if tool in ('docker', 'podman', 'podman-hpc'):
            res.extend(_DEFAULT_CONTAINER_SUBCOMMAND)
        elif tool == 'apptainer':
            res.extend(_DEFAULT_APPTAINER_SUBCOMMAND)
        elif tool == 'srun':
            res = ["prlimit", f"--as={2 * (1024**3)}", "apptainer", "run", "--contain", "--writable-tmpfs"]
        else:
            raise ValueError(f'Unknown container tool: {tool}')
    res.extend(tool_subcommand_args)
    if tool in ('docker', 'podman', 'podman-hpc'):
        res.append(executor_image_name)
    elif tool == 'apptainer' or tool == 'srun':
        sif_f = Path.cwd()/f'executors/sifs/{executor_image_name}.sif'
        assert sif_f.exists(), f'Missing apptainer SIF file: {sif_f}'
        res.append(str(sif_f))
    else:
        raise ValueError(f'Unknown container tool: {tool}')
    res.extend(executor_args)
    return res
