__all__ = ['run_scripts_in_containers', 'run_items_in_workdir_containers']
from pathlib import Path
from typing import Callable

from .simple_supervisor import ContainerExecutionSpec, VolumeSpec, SimpleSupervisor, OneshotContainerItemExecutor
from .run_in_containers import ItemAndId, ExecutionArgs, run_supervisor


def run_scripts_in_containers(
    executor_image_name: str,
    inputs: list[Path],
    execution_args: ExecutionArgs,
    input_filename_glob: str = '*',
):
    """
    Executes given files, each in a (Docker) container, in parallel
    """

    use_short_ids = len(inputs) == 1 and inputs[0].is_dir()
    def long_id(path: Path) -> str:
        parent = path.parent
        if parent == Path('.'):
            return path.stem
        return str(parent / path.stem)
    def file_id(path: Path) -> str:
        if use_short_ids:
            return long_id(path.relative_to(inputs[0]))
        else:
            return long_id(path)

    files_as_items = []
    def visit_dir(path: Path):
        for file in (p for p in path.glob(input_filename_glob) if p.is_file()):
            files_as_items.append((file, file_id(file)))
        for subdir in (p for p in path.iterdir() if p.is_dir()):
            visit_dir(subdir)

    for path in inputs:
        if path.is_dir():
            visit_dir(path)
        else:
            files_as_items.append((path, file_id(path)))

    def prep_input(item: Path, tmpdir: Path | None) -> ContainerExecutionSpec:
        return ContainerExecutionSpec(
            stdin_source=item,
        )

    item_executor = OneshotContainerItemExecutor(
        executor_image_name=executor_image_name,
        prepare_item=prep_input,
        make_tmpdir=True,
        execution_args=execution_args,
    )
    supervisor = SimpleSupervisor(
        executor_launcher=item_executor,
        execution_args=execution_args,
    )
    return run_supervisor(
        executor_image_name,
        files_as_items,
        supervisor,
        execution_args=execution_args,
    )


def run_items_in_workdir_containers[X](
    executor_image_name: str,
    inputs: list[ItemAndId[X]],
    prepare_workdir: Callable[[X, Path], None],
    execution_args: ExecutionArgs,
):
    """
    A minimal API: prepares a temporary work directory for each input item,
    mounts it into the container at `/workdir`, and passes no input to the container.
    """
    def prep_input(
        item: X,
        tmpdir: Path | None,
    ) -> ContainerExecutionSpec:
        assert tmpdir
        prepare_workdir(item, tmpdir)
        return ContainerExecutionSpec(
            stdin_source=None,
            volumes=[VolumeSpec(
                mount_path='/workdir',
                local_volume_path=tmpdir,
            )],
        )

    item_executor = OneshotContainerItemExecutor(
        executor_image_name=executor_image_name,
        prepare_item=prep_input,
        make_tmpdir=True,
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
