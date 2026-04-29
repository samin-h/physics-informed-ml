"""
GW Signal Recovery — iSTFT Reconstruction
==========================================
Takes separated complex spectrograms from the model,
reconstructs time-domain strain h(t) via iSTFT,
and compares with mixture and ground truth.

Pipeline:
  pred (N, T, F_crop) complex64
      → unwhiten: X * sqrt(PSD)
      → zero-pad to full F range: (T, N_BINS)
      → iSTFT overlap-add: h(t)  float32

Outputs per sample:
  recovered_h1(t), recovered_h2(t)  float32 (N_SAMPLES_T,)

Plots:
  - Time domain: mixture vs recovered h1+h2 vs true h1+h2
  - Time domain: recovered h1 vs true h1, recovered h2 vs true h2
  - Q-transform (spectrogram): all signals side by side
  - Residual: mixture - recovered h1 - recovered h2

Usage:
  python recover.py \
      --data-dir /scratch/ph24mscs11029.ph.iith/gw_data \
      --ckpt-dir ./checkpoints \
      --out-dir  ./results \
      --n-samples 5
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.signal import get_window
from flax.training import checkpoints, train_state
import optax

from gw_separator import GWSeparator
from dataloader   import (batch_iterator, get_shard_splits,
                          N_FRAMES, N_FREQ, N_BINS, N_FFT, HOP,
                          SAMPLE_RATE, N_SAMPLES_T,
                          F_LOW_BIN, F_HI_BIN, BIN_RES, WINDOW)
from loss         import si_snr_loss, matched_filter_loss, PERMS_2


# -- Config --

class Config:
    n_sources    = 2
    n_freqs      = N_FREQ
    encoder_dims = [256, 256]
    decoder_dims = [256, 256]
    latent_dim   = 128
    n_heads      = 4

cfg = Config()


# -- iSTFT — inverse STFT via overlap-add --

def istft(X: np.ndarray) -> np.ndarray:
    """
    Inverse STFT via overlap-add reconstruction.

    Args:
        X : complex64 (N_FRAMES, N_BINS) — full frequency range
    Returns:
        float32 (N_SAMPLES_T,) — time domain signal
    """
    window  = WINDOW
    n_out   = (X.shape[0] - 1) * HOP + N_FFT
    output  = np.zeros(n_out, dtype=np.float64)
    norm    = np.zeros(n_out, dtype=np.float64)

    for i, frame_freq in enumerate(X):
        frame_time = np.fft.irfft(frame_freq, n=N_FFT).real
        start      = i * HOP
        output[start : start + N_FFT] += frame_time * window
        norm[start   : start + N_FFT] += window ** 2

    norm   = np.maximum(norm, 1e-8)
    output = output / norm
    return output[:N_SAMPLES_T].astype(np.float32)


# -- Unwhiten + zero-pad + iSTFT --

def recover_signal(X_crop: np.ndarray, psd_crop: np.ndarray) -> np.ndarray:
    """
    Recover time-domain signal from separated whitened spectrogram.

    Steps:
      1. Unwhiten: X_phys = X_crop * sqrt(psd_crop)
      2. Zero-pad back to full frequency range (T, N_BINS)
      3. iSTFT → h(t)

    Args:
        X_crop   : complex64 (T, N_FREQ) — separated whitened spectrogram
        psd_crop : float32   (T, N_FREQ) — PSD used during whitening
    Returns:
        float32 (N_SAMPLES_T,) — recovered time-domain strain
    """
    # Step 1: unwhiten
    X_phys = X_crop * np.sqrt(psd_crop)   # (T, N_FREQ)

    # Step 2: zero-pad to full frequency range
    T      = X_phys.shape[0]
    X_full = np.zeros((T, N_BINS), dtype=np.complex64)
    X_full[:, F_LOW_BIN:F_HI_BIN] = X_phys

    # Step 3: iSTFT
    return istft(X_full)


# -- Load model --

def load_model(ckpt_dir: str):
    model = GWSeparator(
        n_sources    = cfg.n_sources,
        n_freqs      = cfg.n_freqs,
        encoder_dims = cfg.encoder_dims,
        decoder_dims = cfg.decoder_dims,
        latent_dim   = cfg.latent_dim,
        n_heads      = cfg.n_heads,
    )
    dummy  = jnp.ones((1, N_FRAMES, cfg.n_freqs), dtype=jnp.complex64)
    params = model.init(jax.random.PRNGKey(0), dummy)
    state  = train_state.TrainState.create(
        apply_fn = model.apply,
        params   = params,
        tx       = optax.adamw(1e-3),
    )
    state = checkpoints.restore_checkpoint(ckpt_dir, state,
                                           prefix="ckpt_epoch_")
    print(f"Loaded checkpoint from {ckpt_dir}")
    return model, state.params


# -- Find best permutation --

def best_permutation(pred, target):
    best_perm = (0, 1)
    best_sisnr = -np.inf
    for perm in PERMS_2:
        s = sum(-float(si_snr_loss(pred[i], target[j]))
                for i, j in enumerate(perm)) / len(perm)
        if s > best_sisnr:
            best_sisnr = s
            best_perm  = perm
    return best_perm


# -- Time-domain comparison plot --

def plot_time_domain(mixture_td, rec_h1, rec_h2,
                     true_h1, true_h2, out_path, sample_idx=0):
    """
    Four-panel time domain plot:
      1. Mixture vs reconstructed sum (rec_h1 + rec_h2)
      2. True h1 vs recovered h1
      3. True h2 vs recovered h2
      4. Residual: mixture - rec_h1 - rec_h2
    """
    t = np.arange(N_SAMPLES_T) / SAMPLE_RATE

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Time Domain Recovery — Sample {sample_idx}", fontsize=12)

    # Panel 1: mixture vs reconstructed sum
    axes[0].plot(t, mixture_td, color="gray", alpha=0.7,
                 linewidth=0.5, label="Mixture h(t)")
    axes[0].plot(t, rec_h1 + rec_h2, color="red", alpha=0.8,
                 linewidth=0.7, label="Recovered h1+h2")
    axes[0].set_ylabel("Strain")
    axes[0].set_title("Mixture vs Recovered Sum")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # Panel 2: h1 comparison
    axes[1].plot(t, true_h1, color="blue", alpha=0.8,
                 linewidth=0.7, label="True h1(t)")
    axes[1].plot(t, rec_h1, color="orange", alpha=0.8,
                 linewidth=0.7, linestyle="--", label="Recovered h1(t)")
    axes[1].set_ylabel("Strain")
    axes[1].set_title("Signal 1: True vs Recovered")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.2)

    # Panel 3: h2 comparison
    axes[2].plot(t, true_h2, color="green", alpha=0.8,
                 linewidth=0.7, label="True h2(t)")
    axes[2].plot(t, rec_h2, color="purple", alpha=0.8,
                 linewidth=0.7, linestyle="--", label="Recovered h2(t)")
    axes[2].set_ylabel("Strain")
    axes[2].set_title("Signal 2: True vs Recovered")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.2)

    # Panel 4: residual
    residual = mixture_td - rec_h1 - rec_h2
    axes[3].plot(t, residual, color="darkred", alpha=0.8,
                 linewidth=0.5, label="Residual: mixture - h1 - h2")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Strain")
    axes[3].set_title("Residual (should look like noise)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# -- Spectrogram comparison plot --

def plot_spectrograms(mix_spec, pred_h1_spec, pred_h2_spec,
                      true_h1_spec, true_h2_spec,
                      out_path, sample_idx=0):
    """
    2x3 spectrogram grid:
      Row 1: mixture | pred h1 | pred h2
      Row 2: [blank] | true h1 | true h2
    """
    freqs = (np.arange(N_FREQ) + F_LOW_BIN) * BIN_RES

    def spec_mag(X):
        return np.log1p(np.abs(X).T)

    vmax = max(spec_mag(mix_spec).max(),
               spec_mag(true_h1_spec).max(),
               spec_mag(true_h2_spec).max())
    vmin = 0

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Spectrogram Comparison — Sample {sample_idx}",
                 fontsize=12)

    extent = [0, N_FRAMES * HOP / SAMPLE_RATE,
              freqs[0], freqs[-1]]

    kw = dict(aspect="auto", origin="lower", cmap="inferno",
              vmin=vmin, vmax=vmax, extent=extent)

    axes[0, 0].imshow(spec_mag(mix_spec),     **kw)
    axes[0, 0].set_title("Input mixture")
    axes[0, 1].imshow(spec_mag(pred_h1_spec), **kw)
    axes[0, 1].set_title("Predicted h1")
    axes[0, 2].imshow(spec_mag(pred_h2_spec), **kw)
    axes[0, 2].set_title("Predicted h2")

    axes[1, 0].axis("off")
    axes[1, 1].imshow(spec_mag(true_h1_spec), **kw)
    axes[1, 1].set_title("True h1")
    axes[1, 2].imshow(spec_mag(true_h2_spec), **kw)
    axes[1, 2].set_title("True h2")

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# -- Print per-sample metrics --

def print_sample_metrics(pred, target, perm, sample_idx):
    print(f"\n  Sample {sample_idx} metrics (best perm={perm}):")
    for i, j in enumerate(perm):
        si  = -float(si_snr_loss(pred[i], target[j]))
        rho = 1.0 - float(matched_filter_loss(pred[i], target[j]))
        print(f"    pred[{i}] → true[{j}]: "
              f"SI-SNR={si:.2f} dB  ρ={rho:.4f}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  type=str,
                        default="/scratch/ph24mscs11029.ph.iith/gw_data")
    parser.add_argument("--ckpt-dir",  type=str, default="./checkpoints")
    parser.add_argument("--out-dir",   type=str, default="./results")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of examples to visualize")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model, params = load_model(args.ckpt_dir)

    # Get test shard
    _, _, test_paths = get_shard_splits(args.data_dir)

    # Load one batch
    it      = batch_iterator(test_paths[:1], batch_size=args.n_samples,
                             shuffle=False)
    batch   = next(it)

    mixture = batch["mixture"]   # (B, T, F) complex64
    h1_true = batch["h1"]        # (B, T, F) complex64
    h2_true = batch["h2"]        # (B, T, F) complex64
    psd     = batch["psd"]       # (B, T, F) float32

    targets = jnp.stack([h1_true, h2_true], axis=1)

    # Run model
    preds = model.apply(params, mixture)   # (B, 2, T, F)

    # Also load raw time-domain signals for comparison
    import h5py
    with h5py.File(test_paths[0], "r") as f:
        mix_td_batch = f["mixture"][:args.n_samples]   # (B, T_samples)
        h1_td_batch  = f["h1"][:args.n_samples]
        h2_td_batch  = f["h2"][:args.n_samples]

    print(f"\nProcessing {args.n_samples} samples...")

    for idx in range(args.n_samples):
        pred_i   = preds[idx]      # (2, T, F)
        target_i = targets[idx]    # (2, T, F)
        psd_i    = np.array(psd[idx])  # (T, F)

        # Find best permutation
        perm = best_permutation(pred_i, target_i)
        print_sample_metrics(pred_i, target_i, perm, idx)

        # Reorder predictions to match ground truth
        pred_h1_spec = np.array(pred_i[perm[0]])   # (T, F)
        pred_h2_spec = np.array(pred_i[perm[1]])   # (T, F)

        # Recover time domain signals
        rec_h1 = recover_signal(pred_h1_spec, psd_i)
        rec_h2 = recover_signal(pred_h2_spec, psd_i)

        mix_td  = mix_td_batch[idx].astype(np.float32)
        h1_td   = h1_td_batch[idx].astype(np.float32)
        h2_td   = h2_td_batch[idx].astype(np.float32)

        # Time domain plot
        td_path = os.path.join(args.out_dir,
                               f"time_domain_sample_{idx}.png")
        plot_time_domain(mix_td, rec_h1, rec_h2,
                         h1_td, h2_td, td_path, idx)

        # Spectrogram plot
        sp_path = os.path.join(args.out_dir,
                               f"spectrogram_sample_{idx}.png")
        plot_spectrograms(
            np.array(mixture[idx]),
            pred_h1_spec, pred_h2_spec,
            np.array(h1_true[idx]), np.array(h2_true[idx]),
            sp_path, idx
        )

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
