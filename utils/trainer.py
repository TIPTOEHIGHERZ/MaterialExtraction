import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import accelerate
import torch.optim.optimizer
from torch.utils.data import DataLoader
from typing import Callable
import tqdm
import time
import os
from torch.utils.tensorboard import SummaryWriter
import omegaconf


def shadow_paramters(module: nn.Module, device='cpu'):
    return {name: param.detach().to(device) for name, param in module.named_parameters()}


@torch.no_grad()
def apply_shadow_parameters(module: nn.Module, state_dict: dict):
    for name, param in module.named_parameters():
        param.copy_(state_dict[name])

    return


def set_group_lr(optimizer: accelerate.optimizer.torch.optim.Optimizer, lr_list: list):
    assert len(optimizer.param_groups) == len(lr_list)

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_list[i]

    return


def log_function(
    logger: SummaryWriter,
    avg_loss: dict,
    global_iter: int,
    log_config: dict,
    *args,
    **kwargs,
):
    pass


class LogWrapper:
    def __init__(self, train_log_func: Callable, test_log_func: Callable, log_config, test_config=None, is_main_process=False):
        self.train_log_func = train_log_func
        self.test_log_func = test_log_func
        self.log_config = log_config
        self.log_period = log_config['log_period_loss']

        self.test_config = test_config

        self.total_loss = dict()
        self.avg_loss = dict()
        self.iter_loss = dict()
        self.log_dir = log_config['log_dir']
        self.logger = SummaryWriter(os.path.join('./runs', self.log_dir)) if is_main_process else None
        
        self.cnt = 0
        return

    def log_test(self, loss_dict: dict, **kwargs):
        global_iter = kwargs.pop('global_iter')
        
        for name, loss in loss_dict.items():
            self.logger.add_scalar(f'test_loss_{name}', loss.item() if isinstance(loss, torch.Tensor) else loss, global_step=global_iter)
        
        return

    def __call__(self, loss_dict: dict, progress_bar: tqdm.tqdm, mods, test_dataloader: DataLoader, **kwargs):
        global_iter = kwargs.pop('global_iter')
        accumulation_steps = kwargs.pop('accumulation_steps')
        # self.cnt += 1
        for name, loss in loss_dict.items():
            if name not in self.total_loss.keys():
                self.total_loss[name] = list()
                self.avg_loss[name] = 0.
                self.iter_loss[name] = list()
            
            self.iter_loss[name] += [loss.item()] if isinstance(loss, torch.Tensor) else [loss]

            if mods == (accumulation_steps - 1):
                self.total_loss[name] += [sum(self.iter_loss[name]) / accumulation_steps]
                if len(self.total_loss[name]) > self.log_period:
                    self.total_loss[name].pop(0)

                self.avg_loss[name] = sum(self.total_loss[name]) / len(self.total_loss[name])
                self.iter_loss[name] = list()

            # if int(os.environ['RANK']) == 0:
            #     print(self.total_loss[name], self.cnt, self.avg_loss[name])
        
        if self.logger is not None:
            progress_bar.set_postfix(self.avg_loss)
        
        ret_val = None
        if mods == (accumulation_steps - 1):
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
                    **kwargs
                )

        return ret_val
        

def save_model(
    module: dict[str: nn.Module],
    optimizer: torch.optim.Optimizer,
    base_folder: str,
    context: str,
):
    save_folder = os.path.join(base_folder, context)
    os.makedirs(save_folder, exist_ok=True)
    
    for name, model in module.items():
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), os.path.join(save_folder, f'{name}.ckpt'))
        else:
            torch.save(model.state_dict(), os.path.join(save_folder, f'{name}.ckpt'))
    
    torch.save(optimizer.state_dict(), os.path.join(save_folder, 'optimizer.ckpt'))
    return


def test_training(
    module: dict[str: nn.Module], # trainable modules
    training_step: Callable, 
    dataloader: DataLoader,
    eval_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: accelerate.Accelerator,
    log_func: Callable,
    test_steps=1,
    train_args: tuple=None,
    train_kwargs: dict=None,
):
    state_dicts = {name: shadow_paramters(m, 'cpu') for name, m in module.items()}

    dataloader = accelerator.prepare_data_loader(dataloader)
    eval_dataloader = accelerator.prepare_data_loader(eval_dataloader) if eval_dataloader is not None else None

    # optimizer = accelerator.prepare_optimizer(optimizer

    cnt = 0
    # test forward
    progress_bar = tqdm.tqdm(dataloader, desc='test dataloader') if accelerator.is_main_process else dataloader
    for data in progress_bar:
        optimizer.zero_grad()
        loss, loss_dict, log_parameters = training_step(module, data, *train_args, **train_kwargs)

        accelerator.backward(loss)
        optimizer.step()

        cnt += 1
        if cnt >= test_steps:
            break

    if accelerator.is_main_process:
        print(f'memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2: .3f} MB')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if eval_dataloader is None:
        for name, m in module.items():
            state_dict = state_dicts[name]
            apply_shadow_parameters(m, state_dict)
        return

    # test eval
    cnt = 0
    progress_bar = tqdm.tqdm(eval_dataloader, desc='test eval dataloader') if accelerator.is_main_process else eval_dataloader
    for data in progress_bar:
        optimizer.zero_grad()
        loss, loss_dict, log_parameters = training_step(module, data, *train_args, **train_kwargs)

        accelerator.backward(loss)
        optimizer.step()

        cnt += 1
        if cnt >= test_steps:
            break
    if accelerator.is_main_process:
        print(f'memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2: .3f} MB')

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # if log_func is not None:
    #     log_func(
    #         loss_dict, 
    #         progress_bar, 
    #         0,
    #         **log_parameters, 
    #         global_iter=0,
    #         accumulation_steps=1
    #     )
    
    for name, m in module.items():
        state_dict = state_dicts[name]
        apply_shadow_parameters(m, state_dict)

    del optimizer
    del state_dicts
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return
          

def train(
    module: dict[str: nn.Module], # trainable modules
    training_step: Callable, 
    dataloader: DataLoader,
    eval_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train_config: dict,
    eval_config: dict,
    test_config: dict,
    log_func: Callable=None, # check log period inside function
    log_config: dict=None,
    test_func: Callable=None,
    train_args: tuple=None,
    train_kwargs: dict=None,
):
    # training configuration
    summon_ckpt = train_config.get('summon_ckpt', False)
    epochs = train_config['epochs']
    accumulation_steps = train_config['accumulation_steps']
    save_period = train_config['save_period']
    max_saves = train_config['max_saves']
    percision = train_config.get('percision', None)

    # log configuration
    save_folder = os.path.join('./checkpoints', log_config.get('log_dir'))

    accelerator = accelerate.Accelerator(mixed_precision=percision, gradient_accumulation_steps=accumulation_steps)

    optimizer = accelerator.prepare_optimizer(optimizer)
    original_dataloader = dataloader.dataset
    collate_fn = dataloader.collate_fn

    if eval_config is not None:
        # eval configuration
        eval_period = eval_config.get('eval_period', None)
        original_eval_dataloader = eval_dataloader.dataset
        eval_batch_size = eval_dataloader.batch_size
        eval_num_workers = eval_dataloader.num_workers
    
    batch_size = dataloader.batch_size
    num_workers = dataloader.num_workers

    module = {n: accelerator.prepare_model(m) for n, m in module.items()}

    log_func = LogWrapper(log_func, test_func, log_config, test_config, accelerator.is_main_process)
    if accelerator.is_main_process:
        print(f'using training configs: {train_config}\nusing eval configs: {eval_config}')
        time.sleep(3)        

    global_iter = -1
    save_queue = list()

    try:
        cnt = 0
        optimizer.zero_grad()
        for epoch in range(epochs):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            if eval_config is not None and eval_period is not None and (epoch + 1) % eval_period == 0:
                original_eval_dataloader.shuffle()
                data_loader = DataLoader(original_eval_dataloader, batch_size=eval_batch_size, num_workers=eval_num_workers, shuffle=False, collate_fn=collate_fn)
                lr_list = [eval_config['lr'] * gain for gain in eval_config['lr_gain']]
            else:
                original_dataloader.shuffle()
                data_loader = DataLoader(original_dataloader, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
                lr_list = [train_config['lr'] * gain for gain in train_config['lr_gain']]

            set_group_lr(optimizer, lr_list)
            data_loader = accelerator.prepare_data_loader(data_loader)

            progress_bar = tqdm.tqdm(data_loader) if accelerator.is_main_process else data_loader

            for data in progress_bar:
                if percision is not None:
                    with accelerator.autocast():
                        loss, loss_dict, log_parameters = training_step(module, data, *train_args, **train_kwargs)
                else:
                    loss, loss_dict, log_parameters = training_step(module, data, *train_args, **train_kwargs)

                loss /= accumulation_steps
                accelerator.backward(loss)

                cnt += 1
                cnt = cnt % accumulation_steps
                if cnt == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                global_iter += 1
                global_iter_normalized = global_iter // accumulation_steps
                global_iter_mods = global_iter % accumulation_steps

                if log_func is not None:
                    with accelerator.autocast():
                        log_func(
                            loss_dict, 
                            progress_bar, 
                            global_iter_mods,
                            test_dataloader=test_dataloader,
                            **log_parameters,
                            global_iter=global_iter_normalized,
                            accumulation_steps=accumulation_steps,
                            name_space='eval' if (eval_config is not None and eval_period is not None and (epoch + 1) % eval_period == 0) else 'evaluate'
                        )

                if accelerator.is_main_process:
                    progress_bar.set_description(f'{epoch} / {epochs}')
                    if (global_iter_normalized + 1) % save_period == 0 and global_iter_mods == 0:
                        save_queue.append(global_iter_normalized + 1)
                        if len(save_queue) > max_saves:
                            del_save = save_queue.pop(0)
                            os.system(f'rm -r {os.path.join(save_folder, f"train_{del_save}")}')
                        save_model(module, optimizer, base_folder=save_folder, context=f'train_{global_iter_normalized + 1}')
    
    except KeyboardInterrupt:
        if summon_ckpt and accelerator.is_main_process:
            print('summoning ckpt...')
            save_model(module, optimizer, base_folder=save_folder, context='last')
            train_legacy = {
                'global_iter': global_iter,
            }
            omegaconf.OmegaConf.save(train_legacy, os.path.join(save_folder, 'last/train_legacy.yaml'))
        exit()
