from abc import abstractmethod
from typing import Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class MetricBase:
    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def write_tensorboard(self, writer: SummaryWriter, epoch: int) -> None:
        ...


class Mean(MetricBase):
    def __init__(self, name: str = 'Loss') -> None:
        self._name = name
        self._value_sum = 0.0
        self._weight_sum = 0.0

    @property
    def result(self) -> float:
        return self._value_sum / self._weight_sum

    def __call__(self, value: Union[Tensor, float], weight: float = 1.0) -> None:
        if isinstance(value, Tensor):
            value = value.item()
        self._value_sum += value * weight
        self._weight_sum += weight

    def reset(self) -> None:
        self._value_sum = 0.0
        self._weight_sum = 0.0

    def write_tensorboard(self, writer: SummaryWriter, epoch: int) -> None:
        writer.add_scalar(self._name, self.result, epoch)


class ChannelWiseMetricBase(MetricBase):
    def __init__(self, lead_names: Sequence[str], name: str) -> None:
        self._name = name
        self._lead_names = tuple(lead_names)
        self._batches = []

    def __call__(self, output: Tensor, target: Tensor) -> None:
        def canonicalize(x: Tensor) -> Tensor:
            x = x.detach()
            assert x.dim() >= 2
            assert x.size(dim=-2) == len(self._lead_names)
            if x.dim() == 2:
                x = x.unsqueeze(dim=0)
            elif x.dim() > 3:
                x = x.flatten(end_dim=-3)
            return x

        batch = self._compute(canonicalize(output), canonicalize(target))
        self._batches.append(batch.to(device='cpu'))

    def reset(self) -> None:
        self._batches.clear()

    def write_tensorboard(self, writer: SummaryWriter, epoch: int) -> None:
        average_scalar, scalars, histograms = self._compute_values_for_tensorboard()
        writer.add_scalar(f'{self._name}/Average', average_scalar, epoch)
        for i, lead_name in enumerate(self._lead_names):
            scalar_tag = f'{self._name}/Scalar/{lead_name}'
            writer.add_scalar(scalar_tag, scalars[i], epoch)
            histogram_tag = f'{self._name}/Histogram/{lead_name}'
            writer.add_histogram(histogram_tag, histograms[:, i], epoch)

    @abstractmethod
    def _compute(self, output: Tensor, target: Tensor) -> Tensor:
        ...

    def _compute_values_for_tensorboard(self) -> Tuple[Tensor, Tensor, Tensor]:
        assert len(self._batches) > 0
        if len(self._batches) == 1:
            histograms = self._batches[0]
        else:
            histograms = torch.cat(self._batches)
            self._batches = [histograms]
        scalars = histograms.mean(dim=0)
        average_scalar = scalars.mean()
        return average_scalar, scalars, histograms


class RMSE(ChannelWiseMetricBase):
    def __init__(self, lead_names: Sequence[str], name: str = 'RMSE') -> None:
        super().__init__(lead_names, name)

    def _compute(self, output: Tensor, target: Tensor) -> Tensor:
        return (output - target).square_().mean(dim=-1)

    def _compute_values_for_tensorboard(self) -> Tuple[Tensor, Tensor, Tensor]:
        average_scalar, scalars, histograms = super()._compute_values_for_tensorboard()
        return average_scalar.sqrt_(), scalars.sqrt_(), histograms.sqrt()


class PearsonR(ChannelWiseMetricBase):
    def __init__(self, lead_names: Sequence[str], name: str = 'PearsonR') -> None:
        super().__init__(lead_names, name)

    def _compute(self, output: Tensor, target: Tensor) -> Tensor:
        def normalize(x: torch.Tensor) -> torch.Tensor:
            residual = x - x.mean(dim=-1, keepdim=True)
            norm = torch.linalg.norm(residual, dim=-1, keepdim=True)
            return residual.div_(norm)

        return normalize(output).mul_(normalize(target)).sum(dim=-1)


class SimpleMetricAggregate(MetricBase):
    def __init__(self, lead_names: Sequence[str]) -> None:
        self.loss = Mean()
        self.rmse = RMSE(lead_names)
        self.pearsonr = PearsonR(lead_names)

    def __call__(self, loss: Union[Tensor, float], output: Tensor, target: Tensor) -> None:
        batch_size = target.numel() // (target.size(dim=-2) * target.size(dim=-1))
        self.loss(loss, batch_size)
        self.rmse(output, target)
        self.pearsonr(output, target)

    def reset(self) -> None:
        self.loss.reset()
        self.rmse.reset()
        self.pearsonr.reset()

    def write_tensorboard(self, writer: SummaryWriter, epoch: int) -> None:
        self.loss.write_tensorboard(writer, epoch)
        self.rmse.write_tensorboard(writer, epoch)
        self.pearsonr.write_tensorboard(writer, epoch)
