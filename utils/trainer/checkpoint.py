import json
import os
import shutil
from typing import Optional
import torch
import torch.nn as nn
import torch.optim
from torch.nn.parallel import DistributedDataParallel as DDP


# Sidecar file written next to the *.ckpt weights so a resume can recover the
# step/epoch/global_iter the checkpoint was taken at, plus the launch config.
RUN_METADATA_FILENAME = 'run.json'


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def save_run_metadata(save_folder: str, metadata: dict) -> str:
    """Write `run.json` into `save_folder` and return its path.

    `metadata` is any JSON-serializable dict; the harness records step / epoch /
    global_iter / config there so resume does not have to parse the folder name.
    """
    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, RUN_METADATA_FILENAME)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    return path


def load_run_metadata(ckpt_dir: str) -> Optional[dict]:
    """Read `run.json` from a checkpoint dir, or None if it is absent.

    Returns None (rather than raising) when the sidecar is missing or malformed
    so legacy checkpoints without metadata fall back to the old resume path.
    """
    if not ckpt_dir:
        return None
    path = os.path.join(ckpt_dir, RUN_METADATA_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_modules(
    modules: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    base_folder: str,
    context: str,
    metadata: Optional[dict] = None,
):
    save_folder = os.path.join(base_folder, context)
    os.makedirs(save_folder, exist_ok=True)

    for name, model in modules.items():
        torch.save(_unwrap(model).state_dict(), os.path.join(save_folder, f'{name}.ckpt'))

    torch.save(optimizer.state_dict(), os.path.join(save_folder, 'optimizer.ckpt'))

    if metadata is not None:
        save_run_metadata(save_folder, metadata)


def rotate_checkpoints(save_queue: list, max_saves: int, base_folder: str):
    while len(save_queue) > max_saves:
        del_save = save_queue.pop(0)
        del_path = os.path.join(base_folder, f'train_{del_save}')
        if os.path.isdir(del_path):
            shutil.rmtree(del_path)
