"""
GW Signal Separation - Training Loop
=====================================
Wires: dataloader -> model -> loss -> optimizer -> checkpoints

Usage:
    # Smoke test (1 epoch CPU)
    python train.py --data-dir
                    --out-dir ./checkpoints --epochs 1
    # Full training (GPU)
    sbatch train.sh

    # Resume from checkpoint
    python train.py ... -resume
"""

# -- 1. Setup and Imports --
import os
import glob
import json
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints

from model import GWSeparator
from dataloader2 import batch_iterator, get_shard_splits, N_FRAMES, N_FREQ
from loss import compute_loss, compute_metrics
from utils import timeit


# -- 2. Config --
class Config:
    # Model
    n_sources = 2
    n_freqs = N_FREQ
    encoder_dims = [256, 256]
    decoder_dims = [256, 256]
    latent_dim = 128
    n_heads = 4

    # Training
    batch_size = 80
    lr = 1e-3
    weight_decay = 1e-4
    n_epochs = 50
    alpha_mf = 0.5

    # LR schedule
    warmup_steps = 50
    decay_steps = 50000

    # Logging
    save_every = 5  # checkpoint every N epochs
    log_every = 50  # print loss every N batches


cfg = Config()


# -- 3. Train state --
def create_train_state(key, model):
    dummy = jnp.ones((1, N_FRAMES, cfg.n_freqs), dtype=jnp.complex64)
    params = model.init(key, dummy)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.decay_steps,
        end_value=cfg.lr * 0.01,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=cfg.weight_decay),
    )

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters : {n_params:,}")
    print(f"Input shape : (B, {N_FRAMES}, {cfg.n_freqs}) complex64")
    print(f"Output shape: (B, {cfg.n_sources}, {N_FRAMES}, {cfg.n_freqs}) complex")

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# -- 4. Train / eval steps --
@jax.jit
def train_step(state, mixture, targets):
    def loss_fn(params):
        preds = state.apply_fn(params, mixture)
        return compute_loss(preds, targets, cfg.alpha_mf)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def eval_step(state, mixture, targets):
    preds = state.apply_fn(state.params, mixture)
    loss = compute_loss(preds, targets, cfg.alpha_mf)
    return loss, preds


# -- 5. Epoch --
@timeit(enabled=True)
def run_epoch(state, shard_paths, rng, training=True):
    total_loss, n_batches = 0.0, 0
    si_snrs, rhos = [], []

    it = batch_iterator(shard_paths, cfg.batch_size, shuffle=training, rng=rng)

    for batch in it:
        mixture = batch["mixture"]  # (B, T, F)
        targets = jnp.stack([batch["h1"], batch["h2"]], axis=1)  # (B, 2, T, F)

        if training:
            state, loss = train_step(state, mixture, targets)
        else:
            loss, preds = eval_step(state, mixture, targets)
            m = compute_metrics(preds, targets)
            si_snrs.append(m["si_snr_db"])
            rhos.append(m["overlap_rho"])

        total_loss += float(loss)
        n_batches += 1

        if n_batches % cfg.log_every == 0:
            print(f"  batch {n_batches:5d} | loss {float(loss):.4f}")

    mean_loss = total_loss / max(n_batches, 1)
    metrics = {}
    if not training and si_snrs:
        metrics = {
            "si_snr_db": float(np.mean(si_snrs)),
            "overlap_rho": float(np.mean(rhos)),
        }
    return state, mean_loss, metrics


# -- 6. Main --
def train(data_dir, out_dir, resume=False, max_shards=None):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_paths, val_paths, test_paths = get_shard_splits(data_dir)

    if max_shards is not None:
        train_paths = train_paths[:max_shards]

    model = GWSeparator(
        n_sources=cfg.n_sources,
        n_freqs=cfg.n_freqs,
        encoder_dims=cfg.encoder_dims,
        decoder_dims=cfg.decoder_dims,
        latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads,
    )

    state = create_train_state(jax.random.PRNGKey(42), model)
    start_epoch = 0

    if resume:
        state = checkpoints.restore_checkpoint(out_dir, state, prefix="ckpt_epoch_")
        ckpts = glob.glob(os.path.join(out_dir, "ckpt_epoch_*"))
        if ckpts:
            start_epoch = int(max(ckpts, key=os.path.getmtime).split("ckpt_epoch_")[-1])
            print(f"Resumed from epoch {start_epoch}")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_sisnr": [],
        "val_rho": [],
        "epoch_time": [],
    }
    best_val = jnp.inf
    rng = np.random.default_rng(0)

    print(f"\n{'=' * 55}")
    print("GW Signal Separator Training")
    print(f" Epochs : {cfg.n_epochs}")
    print(f" Batch size : {cfg.batch_size}")
    print(f" LR : {cfg.lr}")
    print(f" Alpha MF : {cfg.alpha_mf}")
    print(f" Train shards: {len(train_paths)}")
    print(f" Val shards : {len(val_paths)}")
    print(f"{'=' * 55}\n")

    # -- Epochs --

    for epoch in range(start_epoch, cfg.n_epochs):
        t0 = time.time()
        print(f"Epoch {epoch + 1} / {cfg.n_epochs}")

        # Train
        print("  [Train]")
        state, train_loss, _ = run_epoch(state, train_paths, rng, training=True)

        # Val
        print("  [Val]")
        _, val_loss, val_m = run_epoch(state, val_paths, rng, training=False)

        dt = time.time() - t0

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_sisnr"].append(val_m.get("si_snr_db", 0.0))
        history["val_rho"].append(val_m.get("overlap_rho", 0.0))
        history["epoch_time"].append(dt)

        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"  train_loss : {train_loss:.4f}")
        print(f"  val_loss : {val_loss:.4f}")
        print(f"  val_SI-SNR : {val_m.get('si_snr_db', 0):.2f} dB")
        print(f"  val_rho : {val_m.get('overlap_rho', 0):.4f}")
        print(f"  time : {dt:.1f}s\n")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            checkpoints.save_checkpoint(
                out_dir, state, epoch + 1, prefix="ckpt_epoch_", overwrite=False, keep=3
            )
            print(f" *** Best model saved (val_loss = {best_val:.4f}) ***\n")

        elif (epoch + 1) % cfg.save_every == 0:
            checkpoints.save_checkpoint(
                out_dir, state, epoch + 1, prefix="ckpt_epoch_", overwrite=False, keep=3
            )
        print()

    # Test
    print("Test set evaluation:")

    state = checkpoints.restore_checkpoint(out_dir, state, prefix="ckpt_epoch_")
    _, test_loss, test_m = run_epoch(state, test_paths, rng, training=False)
    print(f" test_loss : {test_loss:.4f}")
    print(f" test_SI-SNR: {test_m.get('si_snr_db', 0):.2f} dB")
    print(f" test_rho : {test_m.get('overlap_rho', 0):.4f}")

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = float(best_val)
    history["test_loss"] = float(test_loss)
    history["test_sisnr"] = test_m.get("si_snr_db", 0.0)
    history["test_rho"] = test_m.get("overlap_rho", 0.0)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Results saved to {out_dir}/history.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--out-dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=cfg.n_epochs)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    cfg.n_epochs = args.epochs
    train(args.data_dir, args.out_dir, args.resume, args.max_shards)


if __name__ == "__main__":
    main()
