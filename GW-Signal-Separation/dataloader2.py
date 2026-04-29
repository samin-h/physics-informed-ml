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
from pathlib import Path
import jax
from time import perf_counter
import jax.numpy as jnp
from jax.scipy.signal import convolve
from scipy.signal import get_window
from scipy.ndimage import convolve1d
from typing import Iterator, Generator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque
from utils import timeit

# -- 2. Constants --
SAMPLE_RATE = 4096
N_SAMPLES_T = 16384
N_FFT = 512
HOP = 256
WIN_TYPE = "hann"

N_FRAMES = (N_SAMPLES_T - N_FFT) // HOP + 1
N_BINS = N_FFT // 2 + 1
BIN_RES = SAMPLE_RATE / N_FFT

# GW signal band: 20 - 1024 Hz
F_LOW_BIN = int(np.ceil(20.0 / BIN_RES))
F_HI_BIN = int(np.floor(1024.0 / BIN_RES)) + 1
N_FREQ = F_HI_BIN - F_LOW_BIN

PSD_SMOOTH = 15

WINDOW = get_window(WIN_TYPE, N_FFT).astype(np.float32)

isTimed = True  # If True will print the time taken by all the function


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
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(N_FRAMES, N_FFT), strides=(x.strides[0] * HOP, x.strides[0])
    ).copy()
    frames *= WINDOW
    return np.fft.rfft(frames, n=N_FFT, axis=-1).astype(np.complex64)


@jax.jit
def stft_jax(x: jnp.ndarray, WINDOW: jnp.ndarray = jnp.asarray(WINDOW)) -> jnp.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT) of a 1D signal using JAX.

    This implementation extracts overlapping frames using
    `jax.lax.conv_general_dilated_patches`, which is the JAX-equivalent of
    sliding-window (stride-based) framing. Each frame is then windowed and
    transformed using a real FFT.

    Parameters
    ----------
    x : jnp.ndarray, shape (N_SAMPLES_T,)
        Input 1D time-domain signal.

    WINDOW : jnp.ndarray, shape (N_FFT,)
        Window function applied to each frame (e.g., Hann or Hamming).
        Must match the frame length `N_FFT`.

    Returns
    -------
    jnp.ndarray, shape (N_FRAMES, N_FFT // 2 + 1), dtype=complex64
        STFT of the input signal. Each row corresponds to the FFT of one frame.

    Notes
    -----
    - Frames are extracted with:
        - frame length = `N_FFT`
        - hop size = `HOP`
        - padding = "VALID" (no implicit padding; only full frames are used)
    - Internally, patches are returned in a flattened convolution layout and
      transposed to obtain shape (N_FRAMES, N_FFT).
    - This function is compatible with `jax.jit` and can be efficiently batched
      using `jax.vmap`.
    - The FFT is computed using `jnp.fft.rfft`, returning only non-negative
      frequency components.

    """
    x = x[jnp.newaxis, jnp.newaxis, :]  # (N, C, H)
    patches = jax.lax.conv_general_dilated_patches(
        x, filter_shape=(N_FFT,), window_strides=(HOP,), padding="VALID"
    )
    frames = patches[0].T
    frames *= WINDOW
    return jnp.fft.rfft(frames, n=N_FFT, axis=-1).astype(jnp.complex64)


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
        lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=power
    )
    psd = np.maximum(psd, 1e-40)

    whitened_power = power / psd
    scale = np.mean(whitened_power)
    psd = psd * scale

    return psd.astype(np.float32)


def estimate_psd_fast(X: np.ndarray) -> np.ndarray:
    """Vectorized PSD estimation."""
    power = np.abs(X) ** 2
    kernel = np.ones(PSD_SMOOTH) / PSD_SMOOTH
    psd = convolve1d(power, weights=kernel, axis=0, mode="constant", cval=0)
    psd = np.maximum(psd, 1e-40)

    whitened_power = power / psd
    scale = np.mean(whitened_power)
    psd = psd * scale
    return psd.astype(np.float32)


@jax.jit
def estimate_psd_jax(X: jnp.ndarray) -> jnp.ndarray:
    """Jax Implementation of PSD Estimation"""
    power = jnp.abs(X) ** 2
    kernel = jnp.ones((PSD_SMOOTH,)) / PSD_SMOOTH
    psd = jax.vmap(
        lambda col: convolve(col, kernel, mode="same"), in_axes=1, out_axes=1
    )(power)
    psd = jnp.maximum(psd, 1e-40)

    whitened_power = power / psd
    scale = jnp.mean(whitened_power)
    psd = psd * scale
    return psd


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


def whiten_and_crop_jax(X: jnp.ndarray, psd: jnp.ndarray) -> jnp.ndarray:
    """
    Jax implementation of Whiten spectrogram and crop to GW signal band.

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
    X_white = X / jnp.sqrt(psd)
    return X_white[:, F_LOW_BIN:F_HI_BIN].astype(jnp.complex64)


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
    # psd   = estimate_psd(X_mix)
    psd = estimate_psd_fast(X_mix)

    mix_w = whiten_and_crop(X_mix, psd)
    h1_w = whiten_and_crop(stft(h1.astype(np.float32)), psd)
    h2_w = whiten_and_crop(stft(h2.astype(np.float32)), psd)
    psd_c = psd[:, F_LOW_BIN:F_HI_BIN].astype(np.float32)

    return mix_w, h1_w, h2_w, psd_c


@jax.jit
def preprocess_sample_jax(mixture, h1, h2):
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
    X_mix = stft_jax(mixture)
    # psd   = estimate_psd(X_mix)
    psd = estimate_psd_jax(X_mix)

    mix_w = whiten_and_crop_jax(X_mix, psd)
    h1_w = whiten_and_crop_jax(stft_jax(h1), psd)
    h2_w = whiten_and_crop_jax(stft_jax(h2), psd)
    psd_c = psd[:, F_LOW_BIN:F_HI_BIN].astype(np.float32)

    return mix_w, h1_w, h2_w, psd_c


@timeit(enabled=isTimed)
def load_shard_only(signal_path: str, params_path: str) -> tuple[jnp.ndarray, ...]:
    """
    Load data from files and convert them into jax arrays
    """
    mixture, h1, h2 = np.load(signal_path, mmap_mode="r")
    params1, params2 = np.load(params_path, mmap_mode="r")

    mixture = jnp.asarray(mixture, dtype=jnp.float32)
    h1 = jnp.asarray(h1, dtype=jnp.float32)
    h2 = jnp.asarray(h2, dtype=jnp.float32)
    params1 = jnp.asarray(params1, dtype=jnp.float32)
    params2 = jnp.asarray(params2, dtype=jnp.float32)

    return mixture, h1, h2, params1, params2


def create_batches(
    mixture, h1, h2, params1, params2, batch_size: int = 80
) -> tuple[jnp.ndarray, ...]:
    """Creates the batches of batch size `batch_size(default = 128)` of mixture, h1, h2 and returned"""
    D = mixture.shape[-1]
    mixture = mixture.reshape(-1, batch_size, D)
    h1 = h1.reshape(-1, batch_size, D)
    h2 = h2.reshape(-1, batch_size, D)
    params1 = params1.reshape(-1, batch_size, params1.shape[-1])
    params2 = params2.reshape(-1, batch_size, params1.shape[-1])

    return mixture, h1, h2, params1, params2


@timeit(enabled=True)
@jax.jit
def process_data(mixture_batch, h1_batch, h2_batch) -> tuple[jnp.ndarray, ...]:

    N, batch_size, _ = mixture_batch.shape
    MIX = jnp.zeros((N, batch_size, N_FRAMES, N_FREQ), dtype=np.complex64)
    H1 = jnp.zeros((N, batch_size, N_FRAMES, N_FREQ), dtype=np.complex64)
    H2 = jnp.zeros((N, batch_size, N_FRAMES, N_FREQ), dtype=np.complex64)
    PSD = jnp.zeros((N, batch_size, N_FRAMES, N_FREQ), dtype=np.float32)

    for i in range(N):
        MIX_B, H1_B, H2_B, PSD_B = jax.jit(
            jax.vmap(preprocess_sample_jax, in_axes=(0, 0, 0), out_axes=(0, 0, 0, 0))
        )(mixture_batch[i], h1_batch[i], h2_batch[i])

        MIX = MIX.at[i].set(MIX_B)
        H1 = H1.at[i].set(H1_B)
        H2 = H2.at[i].set(H2_B)
        PSD = PSD.at[i].set(PSD_B)

    return MIX, H1, H2, PSD

    # batch_size=32
    # N_BATCH = 10000
    # t0 = perf_counter()
    # for i in range(0, N_BATCH, batch_size):
    #     mix_batch, h1_batch, h2_batch = mixture[i:i +batch_size], h1[i:i+batch_size], h2[i:i+batch_size]
    # MIX_batch, H1_batch, H2_batch, PSD_batch = jax.vmap(preprocess_sample_jax, in_axes=(0,0,0), out_axes=(0, 0, 0, 0))(mix_batch, h1_batch, h2_batch)

    # t3 = perf_counter()
    # print(f"Processing Time: {t3-t0:.2f}s")
    # return {
    #     "mixture": MIX,
    #     "h1": H1,
    #     "h2": H2,
    #     "psd": PSD,
    #     "params1": params1,
    #     "params2": params2,
    # }


def get_param_path(signal_path: Path) -> Path:
    param_path = signal_path.parents[1] / "params"
    file_name = signal_path.stem
    param_file_name = "p" + file_name[1:] + ".npy"
    param_file_path = param_path / param_file_name
    return param_file_path


# -- 7. Batch iterator --
def batch_iterator(
    signal_paths: list[Path],
    batch_size: int = 8,
    shuffle: bool = True,
    rng: np.random.Generator = None,
) -> Iterator[dict]:
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

    signal_paths = list(signal_paths)
    if shuffle:
        rng.shuffle(signal_paths)

    for s_path in signal_paths:
        print(f" Loading: {s_path.name}")
        p_path = get_param_path(s_path)
        mixture, h1, h2, params1, params2 = load_shard_only(
            signal_path=s_path, params_path=p_path
        )
        # print(f"Finished Loading: {s_path.name}")

        # print(f"Creating Batches:  {s_path}")
        mix_b, h1_b, h2_b, params1_b, params2_b = create_batches(
            mixture, h1, h2, params1, params2, batch_size=batch_size
        )
        # print(f"Finished Creating Batches: {s_path}")

        # print(f"Processing Data: {s_path}")
        MIX_b, H1_b, H2_b, PSD_b = process_data(mix_b, h1_b, h2_b)
        # print(f"Finished Processing Data: {s_path}")

        # print(f"Yielding Batches: {s_path}")
        N = MIX_b.shape[0]
        batch_indices = np.arange(N)
        if shuffle:
            rng.shuffle(batch_indices)

        for batch_idx in batch_indices:
            shuffle_in_batch = np.arange(MIX_b[batch_idx].shape[0])
            if shuffle:
                rng.shuffle(shuffle_in_batch)
            batch = {
                "mixture": MIX_b[batch_idx][shuffle_in_batch],
                "h1": H1_b[batch_idx][shuffle_in_batch],
                "h2": H2_b[batch_idx][shuffle_in_batch],
                "psd": PSD_b[batch_idx][shuffle_in_batch],
                "params1": params1_b[batch_idx][shuffle_in_batch],
                "params2": params2_b[batch_idx][shuffle_in_batch],
            }
            # t1 = perf_counter()
            # print(f"Time taken to construct batch : {t1 - t0:.3f}s")
            yield batch


# -- 8. Dataset split --
def get_shard_splits(
    data_dir: str, train_frac: float = 0.8, val_frac: float = 0.1
) -> tuple[list[Path], ...]:
    """
    Split shards into train / val / test.

    With 10 shards:
        train = 8 shards = 80,000 samples
        val   = 1 shards = 10,000 samples
        test  = 1 shards = 10,100 samples
    """
    data_path = Path(data_dir)
    signal_dir = data_path / "signal"
    paths = sorted([path for path in signal_dir.iterdir()])
    assert len(paths) > 0, f"No shards found in {data_dir}"

    N = len(paths)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    train_paths = paths[:n_train]
    val_paths = paths[n_train : n_train + n_val]
    test_paths = paths[n_train + n_val :]

    print(f"\nDataset split ({N}) shards :")
    print(
        f"   Train : {len(train_paths)} shards-> {len(train_paths) * 10000:,} samples"
    )

    print(f"   Val : {len(val_paths)} shards-> {len(val_paths) * 10000:,} samples")

    print(f"   Test : {len(test_paths)} shards-> {len(test_paths) * 10000:,} samples")

    return train_paths, val_paths, test_paths


# # -- 9. Sanity check --
if __name__ == "__main__":
    path1 = [r"data/signal/s_shard_0001.npy", r"data/signal/s_shard_0002.npy"]
    # path2 = [r"data/shard_0001.h5", r"data/shard_0002.h5", r"data/shard_0003.h5", r"data/shard_0004.h5", r"data/shard_0005.h5"]
    for data in batch_iterator(path1, batch_size=80):
        ...
        # mix = data["mixture"]
        # h1 = data["h1"]
        # h2 = data["h2"]
        # psd = data["psd"]
        # params1 = data["params1"]
        # params2 = data["params2"]
        # print(f"mix: {mix.shape}")
        # print(f"h1: {h1.shape}")
        # print(f"h2: {h2.shape}")
        # print(f"psd: {psd.shape}")
        # print(f"params1: {params1.shape}")
        # print(f"params2: {params2.shape}")
    # load_shard(path)
    # load_shard_jax(path)
    # load_shard_jax(path)

    # with h5py.File(path, "r") as f:
    #   x = np.empty(f["mixture"].shape)
    #   f["mixture"].read_direct(x)

    # y1 = stft(x[0])
    # print(x.shape)
    # print(y1.shape, y1.dtype)
    # x = jnp.array(x[0])
    # y2 = stft_jax(x)
    # print(y2.shape, y2.dtype)
    # for _ in range(3):
    #     t0 = perf_counter()
    #     for i in batch_iterator(path):
    #         pass
    #     t1 = perf_counter()

    #     for i in batch_iterator_prefetch(path):
    #         pass
    #     t2 = perf_counter()
    #     print(f"Time taken by batch iterator: {t1 - t0:.6f}s")
    #     print(f"Time taken by batch iterator prefetch: {t2 - t1:.6f}s")
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


# v2.0
# def create_batches(mixture, h1, h2, params1, params2, batch_size = 128) -> tuple[jnp.ndarray, ...]:

#     mixture = mixture.reshape(-1, batch_size, N_SAMPLES_T)
#     h1 = h1.reshape(-1, batch_size, N_SAMPLES_T)
#     h2 = h2.reshape(-1, batch_size, N_SAMPLES_T)
#     params1 = params1.reshape(-1, batch_size, N_SAMPLES_T)
#     params2 = params2.reshape(-1, batch_size, N_SAMPLES_T)
#     return mixture, h1, h2, params1, params2
