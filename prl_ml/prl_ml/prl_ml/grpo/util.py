import io
from datetime import datetime
from typing import Generator, Any, Optional
from pathlib import Path
import time
import json

from typeguard import typechecked
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import ray


@typechecked
def batches(
    epochs: int, batch_size: int, data: datasets.Dataset
) -> Generator[datasets.Dataset, None, None]:
    """
    Produces a batch of items, but the last batch may be smaller than batch_size. It is surprising that this is not
    a built-in function in the datasets library.
    """
    for _ in range(epochs):
        for batch_start_index in range(0, len(data), batch_size):
            batch_end_index = min(batch_start_index + batch_size, len(data))
            yield data.select(range(batch_start_index, batch_end_index))


def save_model_and_tokenizer(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, path: Path, save_model_kwargs: dict = {}
):
    """
    Saves the model and tokenizer to path. Configures the model to use KV
    caching by default, which is what you want for inference. This does *not*
    save the optimizer state, so is not suitable for resuming training.
    """
    model.save_pretrained(path, **save_model_kwargs)
    tokenizer.save_pretrained(path)
    model_config_path = path / "config.json"
    model_config = json.loads(model_config_path.read_text())
    model_config["use_cache"] = True
    model_config_path.write_text(json.dumps(model_config, indent=2))


class IterationTimer:
    """
    A simple timer class to measure the duration of code execution, and automatically log to MultiLogger.
    Example usage:

        logger = MultiLogger(log_dir="logs/run1")
        timer = ProfilingTimer(logger)
        timer.start()
        # Block 1 code
        # Create a named lap.
        timer.lap("block1")
        # Block 2 code
        # Create another named lap.
        timer.lap("block2")
        # Timer stops automatically on destruction, and logs the results.
    """

    def __init__(self, iter: int, logger: Optional[Any] = None) -> None:
        """
        Initialize the timer.

        Parameters:
            logger (MultiLogger, optional): The logger to use for logging.
        """
        self.start_time: float = time.perf_counter()
        self.lap_start_time: float = self.start_time
        self.laps: dict[str, float] = {}
        self.logger: Optional[Any] = logger
        self.iter: int = iter

    def lap(self, name: str) -> None:
        """
        Record a lap with a given name.

        Parameters:
            name (str): The name of the lap.
        """
        t_now = time.perf_counter()
        self.laps["timer/"+name] = t_now - self.lap_start_time
        self.lap_start_time = t_now

    def __del__(self) -> None:
        """Log the total duration on destruction."""
        total_duration = time.perf_counter() - self.start_time
        self.laps["timer/total"] = total_duration
        if self.logger is not None:
            self.logger.log(self.laps, self.iter)
        b = io.StringIO()
        ts = datetime.now().strftime('%Y%m%dT%H%M%S')
        print(f"{ts} Timer laps (step {self.iter}):", file=b)
        max_name_length = max(len(name) for name in self.laps.keys())
        for name, duration in self.laps.items():
            print(f"{name:<{max_name_length}}: {duration:8.3f} seconds", file=b)
        print(f"Total: {total_duration:.6f} seconds", end='', file=b)
        ray.logger.info(b.getvalue())