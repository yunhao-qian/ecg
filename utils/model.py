from abc import abstractmethod
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Type, Union

import yaml
import torch
from torch import nn, optim, Tensor

from .metric import MetricBase, SimpleMetricAggregate


class ModelBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer: Union[None, optim.Optimizer, Mapping[str, optim.Optimizer]] = None
        self.lr_scheduler: Union[None, Any, Mapping[str, Any]] = None
        self.metric: Union[None, MetricBase, Mapping[str, MetricBase]] = None

    def save_checkpoint(self, filename: str) -> None:
        checkpoint = {'model': self.state_dict()}

        if self.optimizer is None:
            pass
        elif not isinstance(self.optimizer, Mapping):
            checkpoint['optimizer'] = self.optimizer.state_dict()
        else:
            checkpoint['optimizer'] = {name: optimizer.state_dict() for name, optimizer in self.optimizer.items()}

        if self.lr_scheduler is None:
            pass
        elif not isinstance(self.lr_scheduler, Mapping):
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        else:
            checkpoint['lr_scheduler'] = {name: scheduler.state_dict() for name, scheduler in self.lr_scheduler.items()}

        torch.save(checkpoint, filename)

    @classmethod
    def load_checkpoint(
        cls,
        config_yaml_file: str,
        checkpoint_pt_file: str,
        with_optimizer: bool = True,
        with_lr_scheduler: bool = True,
        device: Union[None, str, torch.device] = None
    ) -> 'ModelBase':
        assert with_optimizer or not with_lr_scheduler
        with open(config_yaml_file, 'r') as fp:
            config = yaml.safe_load(fp)
        checkpoint = torch.load(checkpoint_pt_file)

        model = cls(**config['model'])
        model.load_state_dict(checkpoint['model'])
        if device is not None:
            model.to(device)

        if with_optimizer:
            model.configure_optimizer(**config.get('optimizer', {}))
            if model.optimizer is None:
                pass
            elif not isinstance(model.optimizer, Mapping):
                model.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                for name, optimizer in model.optimizer.items():
                    optimizer.load_state_dict(checkpoint['optimizer'][name])

        if with_lr_scheduler:
            model.configure_lr_scheduler(**config.get('lr_scheduler', {}))
            if model.lr_scheduler is None:
                pass
            elif not isinstance(model.optimizer, Mapping):
                model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            else:
                for name, scheduler in model.lr_scheduler.items():
                    scheduler.load_state_dict(checkpoint['lr_scheduler'][name])

        return model

    @abstractmethod
    def configure_optimizer(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def configure_lr_scheduler(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def configure_metric(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def training_step(self, data: Any) -> Optional[Mapping[str, float]]:
        ...

    @abstractmethod
    def validation_step(self, data: Any) -> Optional[Mapping[str, float]]:
        ...

    @abstractmethod
    def test_step(self, data: Any) -> Optional[Mapping[str, float]]:
        ...

    @abstractmethod
    def validation_epoch_end(self) -> None:
        ...


class SimpleSyncModelBase(ModelBase):
    def __init__(
        self,
        in_leads: int,
        out_leads: int,
        loss_fn: Union[Callable[[Tensor, Tensor], Tensor], str] = nn.functional.mse_loss,
    ) -> None:
        super().__init__()
        self.in_leads = in_leads
        self.out_leads = out_leads
        if not isinstance(loss_fn, str):
            self.loss_fn = loss_fn
        elif loss_fn in globals():
            self.loss_fn = globals()[loss_fn]
        else:
            self.loss_fn = getattr(nn.functional, loss_fn)

    def configure_optimizer(self, type: Union[Type[optim.Optimizer], str] = optim.Adam, **kwargs) -> None:
        if isinstance(type, str):
            type = getattr(optim, type)
        self.optimizer = type(self.parameters(), **kwargs)

    def configure_lr_scheduler(self, type: Optional[Union[Type, str]] = None, **kwargs) -> None:
        if type is None:
            return

        def make_lr_scheduler(type: Union[Type, str], **kwargs) -> Any:
            if isinstance(type, str):
                type = getattr(optim.lr_scheduler, type)
            if 'lr_lambda' in kwargs and isinstance(kwargs['lr_lambda'], str):
                kwargs['lr_lambda'] = eval(kwargs['lr_lambda'], globals())
            if 'schedulers' in kwargs:
                schedulers = []
                for scheduler in kwargs['schedulers']:
                    if isinstance(scheduler, Mapping):
                        schedulers.append(make_lr_scheduler(**scheduler))
                    else:
                        schedulers.append(scheduler)
                kwargs['schedulers'] = schedulers
            return type(self.optimizer, **kwargs)

        self.lr_scheduler = make_lr_scheduler(type, **kwargs)

    def configure_metric(self, out_lead_names: Sequence[str]) -> None:
        self.metric = SimpleMetricAggregate(out_lead_names)

    def training_step(self, data: Mapping[str, Tensor]) -> Dict[str, float]:
        output = self(data['input'])
        target = data['target']
        loss = self.loss_fn(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        self.metric(loss, output, target)
        return {'batch_loss': loss, 'average_loss': self.metric.loss.result}

    def validation_step(self, data: Mapping[str, Tensor]) -> Dict[str, float]:
        output = self(data['input'])
        target = data['target']
        loss = self.loss_fn(output, target).item()
        self.metric(loss, output, target)
        return {'batch_loss': loss, 'average_loss': self.metric.loss.result}

    def test_step(self, data: Mapping[str, Tensor]) -> Dict[str, float]:
        return SimpleSyncModelBase.validation_step(self, data)

    def validation_epoch_end(self) -> None:
        if self.lr_scheduler is None:
            return
        if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(self.metric.loss.result)
        else:
            self.lr_scheduler.step()
