from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader


StepResult = tuple[torch.Tensor, dict, dict]
TrainingStep = Callable[..., StepResult]
BeforeOptimizerStep = Callable[..., Optional[dict[str, float]]]


@dataclass
class TrainSpec:
    """Bundles everything Trainer needs to run a finetune.

    A `prepare_model` factory in each entry script returns a TrainSpec; Trainer
    consumes it and owns the train loop, accelerator, checkpointing, and logging.
    """

    modules: dict[str, nn.Module]
    training_step: TrainingStep
    train_loader: DataLoader
    optimizer: torch.optim.Optimizer
    before_optimizer_step: Optional[BeforeOptimizerStep] = None
    eval_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    log_train: Optional[Callable] = None
    log_test: Optional[Callable] = None
    train_args: tuple = ()
    train_kwargs: dict[str, Any] = field(default_factory=dict)
    start_epoch: int = 0
    # Snapshot of the launch config (plain dict), persisted into each
    # checkpoint's run.json for provenance / reproducibility. Optional so
    # legacy callers that don't set it keep working.
    run_config: Optional[dict] = None
