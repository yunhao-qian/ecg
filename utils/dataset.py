import math
import os.path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import scipy.signal
from torch.utils.data import Dataset


def make_default_preprocess_fn(
    in_leads: Sequence[int],
    out_leads: Sequence[int],
    sampling_rate: float,
    _: int,
) -> Callable[[np.ndarray], Dict[str, np.ndarray]]:
    sos = scipy.signal.butter(
        N=3, Wn=(0.05, min(150.0, sampling_rate * 0.5 - 0.1)), btype='bandpass', output='sos', fs=sampling_rate,
    )

    def preprocess(signal: np.ndarray) -> Dict[str, np.ndarray]:
        signal = scipy.signal.sosfiltfilt(sos, signal.astype(np.float64, copy=False)).astype(np.float32)
        return {'input': signal.take(in_leads, axis=0), 'target': signal.take(out_leads, axis=0)}

    return preprocess


class HDF5Dataset(Dataset):
    def __init__(
        self,
        filename: str,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        make_preprocess_fn: Callable[
            [Sequence[int], Sequence[int], float, int], Callable[[np.ndarray], Mapping[str, Any]],
        ] = make_default_preprocess_fn,
        slice_range: Optional[Tuple[int, int]] = None,
        signal_length: Optional[int] = None,
        sampling_rate: Optional[float] = None,
        lead_names: Optional[Sequence[str]] = None,
    ) -> None:
        self._filename = filename
        self._in_leads = tuple(in_leads)
        self._out_leads = tuple(out_leads)
        self._make_preprocess_fn = make_preprocess_fn
        self._hdf5_file: Optional[h5py.File] = None
        self._preprocess_fn: Optional[Callable[[np.ndarray], Mapping[str, Any]]] = None

        hdf5_file = None

        def get_hdf5_file() -> h5py.File:
            nonlocal hdf5_file
            if hdf5_file is None:
                hdf5_file = h5py.File(filename)
            return hdf5_file

        if slice_range is not None:
            self._slice_range = tuple(slice_range)
        else:
            self._slice_range = 0, get_hdf5_file()['signal'].shape[0]

        if signal_length is not None:
            self._signal_length = int(signal_length)
        else:
            self._signal_length = get_hdf5_file()['signal'].shape[-1]

        if sampling_rate is not None:
            self._sampling_rate = float(sampling_rate)
        else:
            self._sampling_rate = float(get_hdf5_file()['signal'].attrs['sampling_rate'])

        if lead_names is not None:
            self._lead_names = tuple(lead_names)
        else:
            self._lead_names = tuple(name.decode('ascii') for name in get_hdf5_file()['signal'].attrs['lead_names'])

        if hdf5_file is not None:
            hdf5_file.close()

    @property
    def in_leads(self) -> Tuple[int, ...]:
        return self._in_leads

    @property
    def out_leads(self) -> Tuple[int, ...]:
        return self._out_leads

    @property
    def signal_length(self) -> int:
        return self._signal_length

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def lead_names(self) -> Tuple[str, ...]:
        return self._lead_names

    def make_split(self, start: float = 0.0, end: float = 1.0) -> 'HDF5Dataset':
        length = len(self)
        start_index = math.ceil(length * start) + self._slice_range[0]
        end_index = math.ceil(length * end) + self._slice_range[0]
        return HDF5Dataset(
            self._filename,
            self._in_leads,
            self._out_leads,
            self._make_preprocess_fn,
            (start_index, end_index),
            self._signal_length,
            self._sampling_rate,
            self._lead_names,
        )

    @staticmethod
    def from_config(config: Mapping[str, Any], hdf5_dir: Optional[str] = None) -> 'HDF5Dataset':
        config = config.copy()
        if hdf5_dir is not None:
            config['filename'] = os.path.join(hdf5_dir, config['filename'])
        if 'make_preprocess_fn' in config and isinstance(config['make_preprocess_fn'], str):
            config['make_preprocess_fn'] = globals()[config['make_preprocess_fn']]
        split = config.pop('split', None)
        dataset = HDF5Dataset(**config)
        if split is not None:
            dataset = dataset.make_split(*split)
        return dataset

    def __len__(self) -> int:
        return self._slice_range[1] - self._slice_range[0]

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self._filename)
        if self._preprocess_fn is None:
            self._preprocess_fn = self._make_preprocess_fn(
                self._in_leads, self._out_leads, self._sampling_rate, self._signal_length,
            )
        index += self._slice_range[0]
        element = self._preprocess_fn(self._hdf5_file['signal'][index])
        element['id'] = self._hdf5_file['id'][index]
        return element

    def __del__(self) -> None:
        if self._hdf5_file is not None:
            self._hdf5_file.close()
