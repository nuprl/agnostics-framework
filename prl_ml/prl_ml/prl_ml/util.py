import logging
from typing import Any, Optional, Union, List
from torch.utils.tensorboard import SummaryWriter

def create_logger(name: str) -> logging.Logger:
    """
    Creates a reasonable logger with a reasonable output format.

    In a vanilla Python application, you should instantiate the logger once per
    file at the top-level:

    ```python
    logger = create_logger(__name__)
    ```

    When using Ray, you must instead instantiate the logger in the constructor
    for each actor.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class MultiLogger:
    """
    A drop-in replacement for TensorBoard's SummaryWriter that also logs to wandb (if available).

    Example usage:

        logger = MultiLogger(log_dir="logs/run1")
        for step in range(100):
            loss = compute_loss()
            logger.add_scalar("loss", loss, global_step=step)
        logger.close()
    """
    def __init__(self,
                 project_name: Optional[str] = None,
                 run_name: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 hyperparameters: dict = {},
                 *args: Any,
                 **kwargs: Any) -> None:
        """
        Initialize the logger.

        Parameters:
            log_dir (str): Directory where tensorboard logs will be stored.
            *args, **kwargs: Additional arguments passed to the underlying SummaryWriter.
        """

        self.tb_writer: SummaryWriter = SummaryWriter(log_dir=log_dir, *args, **kwargs)

        self.log_dir: Optional[str] = log_dir

        if project_name is None:
            from datetime import datetime
            project_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


        print(hyperparameters)

        try:
            import wandb
            self.wandb: Optional[Any] = wandb
            wandb.init(
                project=project_name,
                config={"log_dir": log_dir, **hyperparameters},
                name=run_name
            )
            print(f"Initialized wandb with project name: {project_name}")
        except ImportError:
            self.wandb = None


    def add_scalar(self, tag: str,
                   scalar_value: Union[int, float],
                   global_step: Optional[int] = None,
                   walltime: Optional[float] = None) -> None:
        """
        Log a scalar variable.

        Parameters:
            tag (str): Data identifier.
            scalar_value (float or int): Value to record.
            global_step (int, optional): Global step value to record.
            walltime (float, optional): Optional override default walltime (time.time()).
        """
        self.tb_writer.add_scalar(tag, scalar_value, global_step, walltime)
        if self.wandb is not None:
            self.wandb.log({tag: scalar_value}, step=global_step)

    def add_histogram(self, tag: str, values: Any, global_step: Optional[int] = None, bins: Union[str, int] = 'tensorflow', walltime: Optional[float] = None) -> None:
        """
        Log a histogram of the tensor of values.

        Parameters:
            tag (str): Data identifier.
            values (array-like): Values to build histogram.
            global_step (int, optional): Global step value to record.
            bins (str or int): Binning strategy.
            walltime (float, optional): Optional override default walltime (time.time()).
        """
        self.tb_writer.add_histogram(tag, values, global_step, bins, walltime)
        if self.wandb is not None:
            self.wandb.log({tag: self.wandb.Histogram(values)}, step=global_step)

    def add_image(self, tag: str, img_tensor: Any, global_step: Optional[int] = None, walltime: Optional[float] = None, dataformats: str = 'CHW') -> None:
        """
        Log an image.

        Parameters:
            tag (str): Data identifier.
            img_tensor (Tensor or ndarray): Image data.
            global_step (int, optional): Global step value to record.
            walltime (float, optional): Optional override default walltime (time.time()).
            dataformats (str): Specifies image data format, default is 'CHW'.
        """
        self.tb_writer.add_image(tag, img_tensor, global_step, walltime, dataformats)
        if self.wandb is not None:
            self.wandb.log({tag: self.wandb.Image(img_tensor)}, step=global_step)

    def add_graph(self, model: Any, input_to_model: Optional[Any] = None, verbose: bool = False) -> None:
        """
        Log a model graph.

        Note: wandb does not support graph logging in the same way as tensorboard.
        This method only logs to tensorboard.

        Parameters:
            model (torch.nn.Module or equivalent): The model to log.
            input_to_model (Tensor or tuple, optional): Model input for tracing the graph.
            verbose (bool, optional): Whether to print the graph structure.
        """
        self.tb_writer.add_graph(model, input_to_model, verbose)

    def flush(self) -> None:
        """
        Flush the event file to disk.
        """
        self.tb_writer.flush()

    def close(self) -> None:
        """
        Close the logger. This flushes the SummaryWriter and finishes the wandb run if applicable.
        """
        self.tb_writer.close()
        if self.wandb is not None:
            self.wandb.finish()

    def add_table(self, tag: str, data: List[List[Any]], global_step: Optional[int] = None) -> None:
        """
        Log a table.

        Parameters:
            tag (str): Data identifier.
            dataframe (pandas.DataFrame): Data to log as a table.
            global_step (int, optional): Global step value to record.
        """
        if len(set(len(row) for row in data)) != 1:
            raise ValueError("All tuples must have the same length.")
        import pandas as pd
        dataframe = pd.DataFrame(data[1:], columns=data[0])
        self.tb_writer.add_text(tag, dataframe.to_string(index=False), global_step)
        if self.wandb is not None:
            self.wandb.log({tag: self.wandb.Table(dataframe=dataframe)}, step=global_step)

    def add_text(self, tag: str, text_string: str, global_step: Optional[int] = None, walltime: Optional[float] = None) -> None:
        if self.wandb is not None:
            self.wandb.log({tag: text_string}, step=global_step)
        self.tb_writer.add_text(tag, text_string, global_step, walltime)

    def log(self,
            data: dict[str, Any],
            step: (int | None) = None,
            commit: (bool | None) = None
        ) -> None:
        """
        Log data to the logger.
        Parameters:
            data (dict): Data to log.
            step (int, optional): Step value for logging.
            commit (bool, optional): Whether to commit the log.
            sync (bool, optional): Whether to sync the log.
        """
        if self.wandb is not None:
            self.wandb.log(data, step=step, commit=commit)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying tensorboard writer.
        This allows the MultiLogger to be used as a drop-in replacement.
        """
        return getattr(self.tb_writer, name)
