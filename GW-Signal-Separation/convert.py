"""
In this file, we will read the generated data files in *.h5 file format and 
change it to *.npy format for faster reads. 

How is data stored in .npy files?
The strategy is like this: 

The .h5 file contains these 5 things:
1. MIXTURE (N, D)
2. H1      (N, D)
3. H2      (N, D)
4. PARAMS1 (N, 11)
5. PARAMS2 (N, 11)

Where N = 10_000 and D = 16384

Now we will store this in .npy file and in order to do that we will combine the data and store it according to following scheme:
1. MIXTURE + H1 + H2 (3, N, D) in signal_*.npy file
2. PARAMS1 + PARAMS2 (2, N, 11) in params_*.npy file

How is data stored in signal_*.npy file?
1. MIXTURE = signal[0]
2. h

For Each shard_*.h5 file there will be two file and we will be storing all the data in np.float32 format to save disk space.

Signal files will be found in signal dir in data/
Params files will be found in params dir in data/

Note: 
    Run this file from the dir GW-Signal-Seperation folder to generate the desired folder structure.
"""

from pathlib import Path
import h5py
import numpy as np

cwd = Path.cwd()
data = cwd / "data"
signal_path = data / "signal"
params_path = data / "params"

signal_path.mkdir(exist_ok=True)
params_path.mkdir(exist_ok=True)
shard_files = [str(p) for p in data.glob("shard*.h5")]
file_names = [str(p).split("/")[-1].strip(".h5") + ".npy" for p in data.glob("shard*.h5")]
print(file_names)
# print(shard_files)
# print(data, signal_path, params_path, sep="\n")

def convert_to_npy(path: str, out_file_name: str):
    with h5py.File(path, "r") as f:
        mixture = np.empty(f["mixture"].shape, dtype=np.float64)
        h1      = np.empty(f["h1"].shape, dtype=np.float64)
        h2      = np.empty(f["h2"].shape, dtype=np.float64)
        params1 = np.empty(f["params1"].shape, dtype=np.float64)
        params2 = np.empty(f["params2"].shape, dtype=np.float64)

        f["mixture"].read_direct(mixture)
        f["h1"].read_direct(h1)
        f["h2"].read_direct(h2)
        f["params1"].read_direct(params1)
        f["params2"].read_direct(params2)

        signal = np.stack((mixture, h1, h2), dtype=np.float32)
        params = np.stack((params1, params2), dtype=np.float32)

        np.save(signal_path / f"s_{out_file_name}", signal)
        np.save(params_path / f"p_{out_file_name}", params)
    
for path, fileName in zip(shard_files, file_names):
    convert_to_npy(path, fileName)

