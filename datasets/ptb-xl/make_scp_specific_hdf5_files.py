import ast
from collections import defaultdict
from pathlib import Path

import h5py
import pandas
import wfdb

def make_hdf5_files(dataset_dir: str, hdf5_dir: str) -> None:
    dataset_dir = Path(dataset_dir)
    hdf5_dir = Path(hdf5_dir)
    data_frame = pandas.read_csv(dataset_dir / 'ptbxl_database.csv')
    data_frame = data_frame[data_frame['electrodes_problems'].isnull()]
    data_frame.sort_values(by='ecg_id', inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    scp_codes_of_interest = {'NDT', 'LAFB', 'ASMI', 'ISC_', 'LVH', 'IRBBB', 'PVC', 'IMI'}
    scp_to_df_idx = defaultdict(list)
    for i, row in data_frame.iterrows():
        for scp_code, likelihood in ast.literal_eval(row['scp_codes']).items():
            if likelihood > 99 and scp_code in scp_codes_of_interest:
                scp_to_df_idx[scp_code].append(i)
    for scp_code, df_indices in scp_to_df_idx.items():
        hdf5_path = str(hdf5_dir / f'ptb_xl_hr_{scp_code}.h5')
        hdf5_file = h5py.File(hdf5_path, 'w')
        id_dataset = hdf5_file.create_dataset('id', shape=(len(df_indices),), dtype='int16')
        signal_dataset = hdf5_file.create_dataset('signal', shape=(len(df_indices), 12, 5000), dtype='float16')
        signal_dataset.attrs.create('sampling_rate', 500, dtype='int16')
        signal_dataset.attrs.create('unit', 'mV', dtype=h5py.string_dtype('ascii', length=2))
        signal_dataset.attrs.create(
            'lead_names',
            ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
            dtype=h5py.string_dtype('ascii', length=3),
        )
        for i, df_idx in enumerate(df_indices):
            df_row = data_frame.iloc[df_idx]
            id_dataset[i] = df_row['ecg_id']
            signal, _ = wfdb.rdsamp(dataset_dir / row['filename_hr'], return_res=16)
            signal_dataset[i] = signal.transpose()
        hdf5_file.close()


if __name__ == '__main__':
    import sys

    assert len(sys.argv) == 3
    make_hdf5_files(sys.argv[1], sys.argv[2])
