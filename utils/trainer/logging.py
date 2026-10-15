import os
from typing import Callable, Optional
import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def default_log(
    logger: Optional[SummaryWriter],
    avg_loss: dict,
    global_iter: int,
    log_config: dict,
    *args,
    **kwargs,
):
    return None


class LogWrapper:
    """Tracks rolling averages of training losses, drives user-provided
    train/test log hooks, and writes to TensorBoard on the main process."""

    def __init__(
        self,
        train_log_func: Callable,
        test_log_func: Callable,
        log_config: dict,
        test_config: Optional[dict] = None,
        is_main_process: bool = False,
    ):
        self.train_log_func = train_log_func or default_log
        self.test_log_func = test_log_func or default_log
        self.log_config = log_config
        self.log_period = log_config['log_period_loss']
        self.test_config = test_config

        self.total_loss: dict[str, list[float]] = {}
        self.avg_loss: dict[str, float] = {}
        self.iter_loss: dict[str, list[float]] = {}
        self.log_dir = log_config['log_dir']
        self.logger = SummaryWriter(os.path.join('./runs', self.log_dir)) if is_main_process else None

    def close(self) -> None:
        """Close this stage's writer without affecting later stages."""
        if self.logger is None:
            return
        self.logger.close()
        self.logger = None

    def log_test(self, loss_dict: dict, **kwargs):
        global_iter = kwargs.pop('global_iter')
        if self.logger is None:
            return
        for name, loss in loss_dict.items():
            value = loss.item() if isinstance(loss, torch.Tensor) else loss
            self.logger.add_scalar(f'test_loss_{name}', value, global_step=global_iter)

    def __call__(
        self,
        loss_dict: dict,
        progress_bar,
        mods: int,
        test_dataloader: DataLoader,
        **kwargs,
    ):
        global_iter = kwargs.pop('global_iter')
        accumulation_steps = kwargs.pop('accumulation_steps')

        for name, loss in loss_dict.items():
            if name not in self.total_loss:
                self.total_loss[name] = []
                self.avg_loss[name] = 0.0
                self.iter_loss[name] = []

            value = loss.item() if isinstance(loss, torch.Tensor) else loss
            self.iter_loss[name].append(value)

            if mods == accumulation_steps - 1:
                self.total_loss[name].append(sum(self.iter_loss[name]) / accumulation_steps)
                if len(self.total_loss[name]) > self.log_period:
                    self.total_loss[name].pop(0)
                self.avg_loss[name] = sum(self.total_loss[name]) / len(self.total_loss[name])
                self.iter_loss[name] = []

        if self.logger is not None and isinstance(progress_bar, tqdm.tqdm):
            progress_bar.set_postfix(self.avg_loss)

        if mods != accumulation_steps - 1:
            return None

        ret_val = self.train_log_func(
            self.logger,
            self.avg_loss,
            global_iter,
            self.log_config,
            **kwargs,
        )

        with torch.no_grad():
            self.test_log_func(
                self.logger,
                global_iter,
                self.test_config,
                test_dataloader,
                **kwargs,
            )

        return ret_val
