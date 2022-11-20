import datetime
import sys
import os
import time
from typing import Any, Dict, Generic, Iterable, Iterator, Mapping, Optional, TextIO, Tuple, Type, TypeVar, Union

import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import HDF5Dataset
from .model import ModelBase, SimpleSyncModelBase

_T = TypeVar('_T')


class ProgressCounter(Generic[_T]):
    def __init__(
        self, iterable: Iterable[_T], prefix: str = '', length: Optional[int] = None, stream: TextIO = sys.stdout,
    ):
        if length is None:
            self._length = len(iterable)
        else:
            self._length = length
        self._iterator = iter(iterable)
        self._prefix = prefix
        self._stream = stream
        self.postfix: Dict[str, Any] = {}

    def __iter__(self) -> Iterator[_T]:
        prev_line_width: Optional[int] = None

        def write_line(line: str) -> None:
            nonlocal prev_line_width
            if prev_line_width is None:
                self._stream.write(line)
            else:
                self._stream.write('\r' + line.ljust(prev_line_width))
            prev_line_width = len(line)
            self._stream.flush()

        def format_postfix_value(value: Any) -> Any:
            if isinstance(value, float):
                return f'{value:.4g}'
            return value

        t_start = time.time()
        write_line(f'{self._prefix} 0/{self._length} [{datetime.timedelta(seconds=0)}<? ?s/it]')
        for index, element in zip(range(1, self._length + 1), self._iterator):
            yield element
            t_current = time.time()
            t_elapsed = t_current - t_start
            seconds_per_it = t_elapsed / index
            t_remaining = seconds_per_it * (self._length - index)
            message = f'{self._prefix} {index}/{self._length} ' \
                      f'[{datetime.timedelta(seconds=round(t_elapsed))}<' \
                      f'{datetime.timedelta(seconds=round(t_remaining))}, ' \
                      f'{seconds_per_it:.3g}s/it'
            if len(self.postfix) > 0:
                message += ', ' + ', '.join(f'{k}={format_postfix_value(v)}' for k, v in self.postfix.items())
            message += ']'
            write_line(message)
        self._stream.write('\n')
        self._stream.flush()


def fit_model(
    model: Union[ModelBase, Type[ModelBase]],
    training_loader: Iterable[Any],
    validation_loader: Iterable[Any],
    epochs: Union[int, Tuple[int, int]],
    checkpoint_dir: str,
    tensorboard_dir: str,
    device: Union[None, str, torch.device] = None,
    model_config: Optional[Mapping[str, Any]] = None,
    optimizer_config: Optional[Mapping[str, Any]] = None,
    lr_scheduler_config: Optional[Mapping[str, Any]] = None,
    metric_config: Optional[Mapping[str, Any]] = None,
) -> None:
    if isinstance(model, type):
        model = model(**model_config)
        if device is not None:
            model.to(device)
    if optimizer_config is not None:
        model.configure_optimizer(**optimizer_config)
    if lr_scheduler_config is not None:
        model.configure_lr_scheduler(**lr_scheduler_config)
    if metric_config is not None:
        model.configure_metric(**metric_config)

    os.makedirs(checkpoint_dir, exist_ok=True)

    def get_lr_as_string(scheduler: Any) -> str:
        if hasattr(scheduler, 'get_last_lr'):
            lr_list = scheduler.get_last_lr()[0]
        else:
            lr_list = [param_group['lr'] for param_group in scheduler.optimizer.param_groups]
        if len(lr_list) == 1:
            return f'{lr_list[0]:.4g}'
        else:
            return '[' + ', '.join(f'{lr:.4g}' for lr in lr_list) + ']'

    def print_lr_scheduler() -> None:
        if model.lr_scheduler is None:
            pass
        elif not isinstance(model.lr_scheduler, Mapping):
            print(f'  Learning rate:', get_lr_as_string(model.lr_scheduler))
        else:
            print(
                f'  Learning rates:',
                ', '.join(f'{name}={get_lr_as_string(scheduler)}' for name, scheduler in model.lr_scheduler.items()),
            )

    def reset_metric():
        if model.metric is None:
            pass
        elif not isinstance(model.metric, Mapping):
            model.metric.reset()
        else:
            for metric in model.metric.values():
                metric.reset()

    if model.metric is None:
        training_writer = None
        validation_writer = None
    else:
        training_writer = SummaryWriter(os.path.join(tensorboard_dir, 'Training'))
        validation_writer = SummaryWriter(os.path.join(tensorboard_dir, 'Validation'))

    def write_tensorboard(writer: Optional[SummaryWriter], epoch: int):
        if writer is None:
            pass
        elif not isinstance(model.metric, Mapping):
            model.metric.write_tensorboard(writer, epoch)
        else:
            for metric in model.metric.values():
                metric.write_tensorboard(writer, epoch)

    epoch_range = range(1, epochs + 1) if isinstance(epochs, int) else range(epochs[0], epochs[1] + 1)
    for epoch in epoch_range:
        print('Epoch', epoch)
        print_lr_scheduler()

        reset_metric()
        model.requires_grad_(True)
        model.train()
        counter = ProgressCounter(training_loader, '  Training  ')
        for data in counter:
            if device is not None:
                data = {key: value.to(device=device) for key, value in data.items()}
            postfix = model.training_step(data)
            if postfix is not None:
                counter.postfix.update(postfix)
        write_tensorboard(training_writer, epoch)

        reset_metric()
        model.requires_grad_(False)
        model.eval()
        counter = ProgressCounter(validation_loader, '  Validation')
        for data in counter:
            if device is not None:
                data = {key: value.to(device=device) for key, value in data.items()}
            postfix = model.validation_step(data)
            if postfix is not None:
                counter.postfix.update(postfix)
        write_tensorboard(validation_writer, epoch)

        model.validation_epoch_end()
        model.save_checkpoint(os.path.join(checkpoint_dir, f'epoch_{epoch}.pt'))

    if model.metric is not None:
        training_writer.close()
        validation_writer.close()


class Trainer:
    def __init__(
        self,
        dataset_config: Mapping[str, Any],
        dataloader_config: Mapping[str, Any],
        checkpoint_dir: str,
        tensorboard_dir: str,
        hdf5_dir: str,
        device: Union[None, str, torch.device] = None,
    ) -> None:
        self._data_config_yaml = {'dataset': dataset_config, 'dataloader': dataloader_config}
        self._base_dataset = HDF5Dataset.from_config(dataset_config['dataset'], hdf5_dir)
        training_set = self._base_dataset.make_split(*dataset_config['splits']['training'])
        validation_set = self._base_dataset.make_split(*dataset_config['splits']['validation'])
        self._training_loader = DataLoader(training_set, **{'shuffle': True, **dataloader_config})
        self._validation_loader = DataLoader(validation_set, **{'shuffle': False, **dataloader_config})
        self._checkpoint_dir = checkpoint_dir
        self._tensorboard_dir = tensorboard_dir
        self._device = device

    def fit_simple_sync_model(
        self,
        model: Union[SimpleSyncModelBase, Type[SimpleSyncModelBase]],
        log_prefix: str,
        epochs: Union[int, Tuple[int, int]],
        model_config: Optional[Mapping[str, Any]] = None,
        optimizer_config: Optional[Mapping[str, Any]] = None,
        lr_scheduler_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if model_config is not None:
            model_config = {
                'in_leads': len(self._base_dataset.in_leads),
                'out_leads': len(self._base_dataset.out_leads),
                **model_config,
            }

        checkpoint_dir = os.path.join(self._checkpoint_dir, log_prefix)
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, 'data_config.yaml'), 'w') as fp:
            yaml.safe_dump(self._data_config_yaml, fp)
        if isinstance(model, type):
            model_config_yaml = {}
            if model_config is not None:
                model_config_yaml['model'] = model_config
            if optimizer_config is not None:
                model_config_yaml['optimizer'] = optimizer_config
            if lr_scheduler_config is not None:
                model_config_yaml['lr_scheduler'] = lr_scheduler_config
            with open(os.path.join(checkpoint_dir, 'model_config.yaml'), 'w') as fp:
                yaml.safe_dump(model_config_yaml, fp)

        return fit_model(
            model,
            self._training_loader,
            self._validation_loader,
            epochs,
            checkpoint_dir,
            os.path.join(self._tensorboard_dir, log_prefix),
            self._device,
            model_config,
            optimizer_config,
            lr_scheduler_config,
            {'out_lead_names': tuple(self._base_dataset.lead_names[i] for i in self._base_dataset.out_leads)},
        )
