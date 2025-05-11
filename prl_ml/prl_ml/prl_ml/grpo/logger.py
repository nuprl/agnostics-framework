"""
This module has some Ray wrappers around our loggers to allow distributed
actors to centralize logging.
"""

import json
import ray
from typing import Any, Optional, Union, List
from prl_ml.util import MultiLogger
from .util import IterationTimer
import datetime
import numpy as np
from pathlib import Path
import time

def init_logger(run_dir: str, hyperparameters: dict, project_name: Optional[str] = None, run_name: Optional[str] = None, verbose: Optional[bool] = True) -> "Logger":
    """
    An opinionated logger initialization.

    Creates a timestamped run directory under run_dir. Records git commit info
    to commit.txt. Sets up project name (defaults to timestamp if not provided).
    Returns a logging actor that can be passed to other actors.
    """
    cur_date_time = np.datetime64("now", "s")
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if project_name is None:
        project_name = 'prl_ml.grpo.ray_trainer'
    run_dir = Path(run_dir) / cur_date_time.astype(str)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        (run_dir/"hyperparameters.json").write_text(json.dumps(hyperparameters))
    except Exception as e:
        ray.logger.warning(f'Failed to save hyperparameters on disk: {e}')
    return Logger.remote(
        project_name=project_name, run_name=run_name, log_dir=str(run_dir),
        hyperparameters=hyperparameters
    )


@ray.remote
class Logger:
    """
    A Ray actor that wraps prl_ml.util.MultiLogger. The actor allows us
    to centralize logging across multiple processes.
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._multi_logger = MultiLogger(
            project_name, run_name, log_dir, *args, **kwargs
        )
        self._max_step = 0
        self._adjusted_step_keys = set()
        self._logged_keys = { }


    def _adjust_step(self, key: str, step: int) -> int:
        """
        Wandb requires monotonically increasing step numbers. This is a little
        hack to ensure that we don't log a step from a previous timestep.
        """
        if step > self._max_step:
            self._max_step = step
            return step
        if step == self._max_step:
            return step

        if key not in self._adjusted_step_keys:
            self._adjusted_step_keys.add(key)
            ray.logger.warning(f"Warning: Increasing the step number for {key} for Wandb. This may occur again but will not be logged.")
        return self._max_step

    def print_limit(self, key: str, limit: int, value: str) -> None:
        if key not in self._logged_keys:
            self._logged_keys[key] = 0
        if self._logged_keys[key] < limit:
            ray.logger.info(value)
            self._logged_keys[key] += 1


    def add_scalar(
        self,
        tag: str,
        scalar_value: Union[int, float],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        self._multi_logger.add_scalar(tag, scalar_value, self._adjust_step(tag, global_step), walltime)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: Optional[int] = None,
        bins: Union[str, int] = "tensorflow",
        walltime: Optional[float] = None,
    ) -> None:
        self._multi_logger.add_histogram(tag, values, self._adjust_step(tag, global_step), bins, walltime)

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        dataformats: str = "CHW",
    ) -> None:
        self._multi_logger.add_image(
            tag, img_tensor, self._adjust_step(tag, global_step), walltime, dataformats
        )

    def add_graph(
        self, model: Any, input_to_model: Optional[Any] = None, verbose: bool = False
    ) -> None:
        self._multi_logger.add_graph(model, input_to_model, verbose)

    def flush(self) -> None:
        self._multi_logger.flush()

    def close(self) -> None:
        self._multi_logger.close()

    def add_table(
        self, tag: str, data: List[List[Any]], global_step: Optional[int] = None
    ) -> None:
        self._multi_logger.add_table(tag, data, self._adjust_step(tag, global_step))

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        self._multi_logger.add_text(tag, text_string, self._adjust_step(tag, global_step), walltime)

    def log(
        self,
        data: dict[str, Any],
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        key = ",".join(data.keys())
        self._multi_logger.log(data, self._adjust_step(key, step), commit)

    def create_timer(self, iter: int) -> "TimerActor":
        """
        Create a new TimerActor instance for timing operations.

        Parameters:
            iter (int): The iteration number for this timer

        Returns:
            TimerActor: A new timer actor instance
        """
        return TimerActor.remote(iter=iter, logger=ray.get_runtime_context().current_actor)


@ray.remote
class TimerActor:
    """
    A Ray actor that wraps prl_ml.util.IterationTimer for distributed timing measurements.
    """

    def __init__(self, iter: int, logger: ray.ObjectRef) -> None:
        self._logger = logger
        self._timer = IterationTimer(iter=iter)

    def lap(self, name: str) -> None:
        """Record a lap with a given name."""
        self._timer.lap(name)

    def done(self) -> None:
        """Explicitly call the timer's cleanup to log results."""
        total_duration = time.perf_counter() - self._timer.start_time
        self._timer.laps["timer/total"] = total_duration
        ray.get(self._logger.log.remote(self._timer.laps, self._timer.iter))
        self._timer.__del__()
