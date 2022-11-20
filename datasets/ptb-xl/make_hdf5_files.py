from pathlib import Path

import h5py
import pandas
import wfdb


def make_hdf5_files(dataset_dir: str, lr_path: str, hr_path: str) -> None:
    dataset_dir = Path(dataset_dir)
    data_frame = pandas.read_csv(dataset_dir / 'ptbxl_database.csv')
    data_frame = data_frame[data_frame['electrodes_problems'].isnull()]
    data_frame.sort_values(by='ecg_id', inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    lead_names = ('I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    for hdf5_path, signal_length, sampling_rate, filename_key in \
            (lr_path, 1000, 100, 'filename_lr'), (hr_path, 5000, 500, 'filename_hr'):
        hdf5_file = h5py.File(hdf5_path, 'w')
        id_dataset = hdf5_file.create_dataset('id', shape=(len(data_frame),), dtype='int16')
        signal_dataset = hdf5_file.create_dataset('signal', shape=(len(data_frame), 12, signal_length), dtype='float16')
        signal_dataset.attrs.create('sampling_rate', sampling_rate, dtype='int16')
        signal_dataset.attrs.create('unit', 'mV', dtype=h5py.string_dtype('ascii', length=2))
        signal_dataset.attrs.create('lead_names', lead_names, dtype=h5py.string_dtype('ascii', length=3))
        for i, row in data_frame.iterrows():
            id_dataset[i] = row['ecg_id']
            signal, metadata = wfdb.rdsamp(dataset_dir / row[filename_key], return_res=16)
            assert all(unit == 'mV' for unit in metadata['units'])
            assert tuple(metadata['sig_name']) == lead_names
            signal_dataset[i] = signal.transpose()
        hdf5_file.close()


if __name__ == '__main__':
    import sys

    assert len(sys.argv) == 4
    make_hdf5_files(sys.argv[1], sys.argv[2], sys.argv[3])
