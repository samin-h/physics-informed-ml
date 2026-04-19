"""
GW Signal Separation -- Dataloader
=====================================

Reads sharded HDF5 files, applies STFT + whitening on the fly,
and yields batches ready for the model.

Data spces:
    h1, h2 , mixture, noise : float64 (10000, 16384)
    params1, params2        : float64 (10000, 11)
    total samples = 100000
    
STFT Parameters:
================

SAMPLE_RATE = 4096 Hz
    Samples per second. Set during data generation.
    Nyquist = 2048 Hz - Highest frequency without aliasing.
    
N_SAMPLES_T = 16384
    Total samples per signal = 4s * 4096 = 16384
    
N_FFT = 512
    Window length in samples = 512 / 4096 = 125 ms.
    Frequency resolution = SAMPLE_RATE / N_FFT = 8 Hz per bin.
    Chosen for 4s signals - need fine time resolution to resolve
    0.1s merger offsets between two overlapping signals.
    Larger window -> finer freq res, coarser time res.
    Smaller window -> finer time res, coarser freq res.
    
HOP = 256
    Step between windows = 256 / 4096 = 62.5 per frame.
    50 % overlap (HOP = N_FFT / 2) - standard choice.
    A 0.1s merger offset = 1.6 frames at this resolution.
    A 0.5s offset = 8 frames - clearly visible to attention.
    
N_FRAMES = (16384 - 512) // 256 + 1 = 62
    Time frames in spectrogram. Attention bottleneck attends
    over these 62 positions to track chirp trajectories.
    
N_BINS = 512 // 2 + 1 = 257
    Frequency bins from rfft. Covers 0 to 2048 Hz.

BIN_RES = 4096 / 512 = 8 Hz
    Each bin covers an 8 Hz band.

F_LOW_BIN = ceil (20 / 8) = 3  -> 24 Hz actual
F_HI_BIN = floarr(1024/8) + 1 = 129 -> 1024 Hz actual
N_FREQ = 129 - 3 = 126 bins
    We crop to [20, 1024] Hz - the GW signal band.
    Below 20 Hz: seismic noise, no signal.
    Above 1024 Hz: above merger freq for total mass > 20 Mo

PSD_SMOOTH = 15 frames = 15 * 62.5 ms = 1 second
    Running mean window for PSD estimation.
    Smooths power over ~1s to get stable noise floar.
    Estimated from MIXTURE so input and targets share same PSD.
    
Shape : (62 frames, 126 bins) complex64
"""
# -- 1. Setup and Imports --
import os
import glob
import h5py
import numpy as np
import jax
from time import perf_counter
import jax.numpy as jnp
from scipy.signal import get_window
from typing import Iterator

# -- 2. Constants --
SAMPLE_RATE = 4096
N_SAMPLES_T = 16384
N_FFT       = 512
HOP         = 256
WIN_TYPE    = "hann"

N_FRAMES = (N_SAMPLES_T - N_FFT)  // HOP + 1
N_BINS   = N_FFT // 2 + 1
BIN_RES  = SAMPLE_RATE / N_FFT

# GW signal band: 20 - 1024 Hz
F_LOW_BIN  = int(np.ceil(20.0 / BIN_RES))
F_HI_BIN   = int(float(1024.0 / BIN_RES)) + 1
N_FREQ     = F_HI_BIN - F_LOW_BIN

PSD_SMOOTH = 15

WINDOW = get_window(WIN_TYPE, N_FFT).astype(np.float32)

# -- 3. STFT --
def stft(x: np.ndarray) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform.
    
    Uses stride tricks to extract overlapping frames efficiently
    without copying the full array N_FRAMES times.

    Args:
        x : float32 (N_SAMPLES_T,)

    Returns:
        complex64 (N_FRAMES, N_BINS)
    """
    x = x.astype(np.float32)
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape = (N_FRAMES, N_FFT),
        strides = (x.strides[0] * HOP, x.strides[0])
    ).copy()
    frames *= WINDOW
    return np.fft.rfft(frames, n=N_FFT, axis = -1).astype(np.complex64)

# -- 4. PSD estimation --
def estimate_psd(X: np.ndarray) -> np.ndarray:
    """
    Estimate non-stationary PSD via running mean over time frames.
    
    Power at tile (t, f) = |x[t, f]|^2
    Smooth over PSD_SMOOTH frames to get stable noise floar.

    Args:
        X : complex64 (N_FRAMES, N_BINS)

    Returns:
        float32 (N_FRAMES, N_BINS)
    """
    power = np.abs(X) ** 2
    kernel = np.ones(PSD_SMOOTH, dtype=np.float32) / PSD_SMOOTH
    psd = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"),
        axis=0, arr=power
    )
    return np.maximum(psd, 1e-40).astype(np.float32)

# -- 4. Whiten + crop
def whiten_and_crop(X: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Whiten spectrogram and crop to GW signal band.
    
    Whitening: divide by sqrt (PSD) at each tile.
    After whitening, Gaussian noise has unit variance everywhere.
    The network sees a flat noise floar and learns signal structure.
    
    All signals (mixture, h1, h2) are whitened with the SAME
    mixture PSD so they all live in the same domain.
    
    Args:
        X   : complex64 (N_FRAMES, N_BINS)
        psd : float32 (N_FRAMES, N_BINS)
    Returns:
        complex64 (N_FRAMES, N_FREQ)
    """ 
    X_white = X / np.sqrt(psd)
    return X_white[:, F_LOW_BIN:F_HI_BIN].astype(np.complex64)

# -- 5. Full preprocessing for one sample
def preprocess_sample(mixture, h1, h2):
    """
    STFT + PSD estimate + whiten + crop for one sample.

    Args:
        mixture, h1, h2 : float64 (N_SAMPLES_T,)
    
    Returns:
        mix_w  : complex64 (N_FRAMES, N_FREQ) - model input
        h1_w   : complex64 (N_FRAMES, N_FREQ) - target 1
        h2_w   : complex64 (N_FRAMES, N_FREQ) - target 2
        psd_c  : float32 (N_FRAMES, N_FREQ) - PSD for unwhitening 
    """
    X_mix = stft(mixture.astype(np.float32))
    psd   = estimate_psd(X_mix)
    
    mix_w = whiten_and_crop(X_mix, psd)
    h1_w  = whiten_and_crop(stft(h1.astype(np.float32)), psd)
    h2_w  = whiten_and_crop(stft(h2.astype(np.float32)), psd)
    psd_c = psd[:, F_LOW_BIN:F_HI_BIN].astype(np.float32)
    
    return mix_w, h1_w, h2_w, psd_c

# -- 6. Shard loader --
# def load_shard(path: str) -> dict:
#     """
#     Load and preprocess all samples in one HDF5 shard.
#     """
#     with h5py.File(path, "r") as f:
#         mixture = np.emptyf["mixture"][:].astype(np.float32)
#         h1      = f["h1"][:].astype(np.float32)
#         h2      = f["h2"][:].astrype(np.float32)
#         params1 = f["params1"][:].astype(np.float32)
#         params2 = f["params2"][:].astype(np.float32)
    
#     N   = mixture.shape[0]
#     MIX = np.zeros((N, N_FRAMES, N_FREQ), dtype=np.complex64)
#     H1  = np.zeros((N, N_FRAMES, N_FREQ), dtype=np.complex64)
#     H2  = np.zeros((N, N_FRAMES, N_FREQ), dtype=np.complex64)
#     PSD = np.zeros((N, N_FRAMES, N_FREQ), dtype=np.float32)
    
#     for i in range(N):
#         MIX[i], H1[i], H2[i], PSD[i] = preprocess_sample(mixture[i], h1[i], h2[i])
        
#     return {
#         "mixture"  : MIX,
#         "h1"       : H1,
#         "h2"       : H2,
#         "psd"      : PSD,
#         "params1"  : params1, 
#         "params2"  : params2,
#     }
def load_shard(path: str) -> dict:
    """
    Load and preprocess all samples in one HDF5 shard.
    Compatible with the JAX environment.
    """
    t0 = perf_counter()
    # --- Load raw data safely ---
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
    
    t1 = perf_counter()
    print(f"HDF5 READ: {t1 - t0:.2f}s")

    # Convert to float32 for efficient JAX training
    mixture = mixture.astype(np.float32)
    h1      = h1.astype(np.float32)
    h2      = h2.astype(np.float32)
    params1 = params1.astype(np.float32)
    params2 = params2.astype(np.float32)

    t2 = perf_counter()
    print(f"Type Convert Time: {t2 - t1:.2f}s")

    # --- Preallocate arrays ---
    N = mixture.shape[0]
    MIX = np.zeros((N, N_FRAMES, N_FREQ), dtype=np.complex64)
    H1  = np.zeros((N, N_FRAMES, N_FREQ), dtype=np.complex64)
    H2  = np.zeros((N, N_FRAMES, N_FREQ), dtype=np.complex64)
    PSD = np.zeros((N, N_FRAMES, N_FREQ), dtype=np.float32)

    # --- Preprocess samples ---
    for i in range(N):
        MIX[i], H1[i], H2[i], PSD[i] = preprocess_sample(
            mixture[i], h1[i], h2[i]
        )
    t3 = perf_counter()
    print(f"Preprocess {N} Sample Time: {t3 - t2:.2f}s")
    print(f"Total Time: {t3-t0:.2f}s")
    return {
        "mixture": MIX,
        "h1": H1,
        "h2": H2,
        "psd": PSD,
        "params1": params1,
        "params2": params2,
    }
    
# -- 7. Batch iterator --
def batch_iterator(shard_paths: list,
                   batch_size: int = 8,
                   shuffle: bool = True,
                   rng: np.random.Generator = None) -> Iterator[dict]:
    """
    Yield batches of JAX arrays for training.
    
    Loads one shard at a time
    GPU sees one batch at a time.

    Args:
        shard_paths : list of HDF5 file paths
        batch_size  : samples per batch
        shuffle     : shuffle shard order and samples
        rng.        : numpy random generator

    Yields:
        dict of JAX arrays on GPU
    """
    if rng is None:
        rng = np.random.default_rng()
    
    paths = list(shard_paths)
    if shuffle:
        rng.shuffle(paths)
    
    for path in paths:
        print(f" Loading: {os.path.basename(path)}")
        t0_ = perf_counter()
        shard = load_shard(path)
        print(f"Finished Loading in {perf_counter() - t0_}")
        N     = shard["mixture"].shape[0]
        idx   = np.arange(N)
        if shuffle:
            rng.shuffle(idx)
        
        for start in range(0, N - batch_size + 1, batch_size):
            b  = idx[start : start + batch_size]
            yield {
                "mixture" : jnp.array(shard["mixture"][b]),
                "h1"      : jnp.array(shard["h1"][b]),
                "h2"      : jnp.array(shard["h2"][b]),
                "psd"     : jnp.array(shard["psd"][b]),
                "params1" : jnp.array(shard["params1"][b]),
                "params2" : jnp.array(shard["params2"][b]),
            }
# -- 8. Dataset split --
def get_shard_splits(data_dir: str,
                     train_frac: float = 0.8,
                     val_frac : float = 0.1) -> tuple:
    """
    Split shards into train / val / test.
    
    With 10 shards:
        train = 8 shards = 80,000 samples
        val   = 1 shards = 10,000 samples
        test  = 1 shards = 10,100 samples
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.h5")))
    assert len(paths) > 0, f"No shards found in {data_dir}"
    
    N       = len(paths)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)
    
    train_paths = paths[:n_train]
    val_paths   = paths[n_train : n_train + n_val]
    test_paths  = paths[n_train + n_val :]
    
    print(f"\nDataset split ({N}) shards :")
    print(f"   Train : {len(train_paths)} shards"
          f"-> {len(train_paths) * 10000:,} samples")
    
    print(f"   Val : {len(val_paths)} shards"
          f"-> {len(val_paths) * 10000:,} samples")
    
    print(f"   Test : {len(test_paths)} shards"
          f"-> {len(test_paths) * 10000:,} samples")
    
    return train_paths, val_paths, test_paths

# # -- 9. Sanity check --
# if __name__ == "__main__":
#     import sys
#     data_dir = "/scratch/ph24mscs11029.ph.iith/gw_data/4s"
#     paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.h5")))
#     if len(paths) == 0:
#         print(f"No shards found in {data_dir}")
#         sys.exit(1)
        
#     print(f"\nFound {len(paths)} shards")
#     print(f"\nSTFT settings:")
#     print(f" Window : {N_FFT} samples = {N_FFT/SAMPLE_RATE*1000:.0f} ms")
#     print(f" Hop : {HOP} samples = {HOP/SAMPLE_RATE*1000:.0f} ms")
#     print(f" T frames : {N_FRAMES}")
#     print(f" F bins : {N_FREQ} "
#     f"({F_LOW_BIN*BIN_RES:.0f}–{(F_HI_BIN-1)*BIN_RES:.0f} Hz)")
#     print(f" Shape : ({N_FRAMES}, {N_FREQ}) complex64")
    
#     shard = load_shard(paths[0])
    
#     print("\nOutput shapes:")
#     for k, v in shard.items():
#         print(f" {k:10s}: {v.shape} {v.dtype}")
        
#     print(f"\nWhitening check:")
#     print(f" mixture |X| mean : {np.abs(shard['mixture']).mean():.4f}"
#     f" (should be ~1.0)")
#     print(f" h1 |X| mean : {np.abs(shard['h1']).mean():.4f}")
#     print(f" h2 |X| mean : {np.abs(shard['h2']).mean():.4f}")
#     print(f"\nBatch test:")
#     it = batch_iterator(paths[:1], batch_size=8, shuffle=False)
#     batch = next(it)
    
#     for k, v in batch.items():
#         print(f" {k:10s}: {v.shape} {v.dtype}")
#     get_shard_splits(data_dir)
