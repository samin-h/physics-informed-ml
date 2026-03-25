# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────
import time
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import flax.linen as nn
from flax.training import train_state

# Enforce 64-bit precision for strict physical accuracy
jax.config.update("jax_enable_x64", True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Hyperparameter Configuration 
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Data
    n_samples: int       = 500
    x_min: float         = -5.0
    x_max: float         = 5.0
    noise_std: float     = 0.05
    val_fraction: float  = 0.2          # 80 / 20 train-val split

    # Architecture
    # Wider + use Fourier input features to directly encode periodicity.
    # SIREN-inspired first layer (sin activation) handles sin(3x) better.
    hidden_dims: Sequence[int] = field(default_factory=lambda: [256, 256, 128, 64])
    # Dropout OFF: with 500 samples + mini-batches, dropout inflates train loss
    # above val loss (inverted gap). Weight decay handles regularization instead.
    dropout_rate: float  = 0.0

    # Optimisation
    learning_rate: float  = 1e-3        # peak LR
    warmup_epochs: int    = 300         # linear warmup → prevents early LR spikes
    weight_decay: float   = 5e-5        # lighter with dropout off
    grad_clip_norm: float = 1.0
    # Near-full batch: 500 samples → 2 batches/epoch = very smooth gradients
    batch_size: int       = 256
    max_epochs: int       = 30_000

    # Early stopping
    patience: int        = 4_000       # more patience: cosine LR takes time
    min_delta: float     = 1e-8

    # Logging — more frequent so loss curve is readable
    log_every: int       = 200

    # Reproducibility
    master_seed: int     = 0


cfg = Config()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_data(
    key: jax.Array,
    cfg: Config
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate noisy samples of the quantum harmonic oscillator wavefunction:
        Ψ(x) = sin(3x) · exp(−0.2 x²)

    Returns (X_train, Y_train, X_val, Y_val,
             X_train_raw, X_val_raw, X_mean, X_std)
    where X_* are standardised inputs and X_*_raw are physical coordinates.

    CRITICAL: We shuffle before splitting. The original code split a *sorted*
    array, so val received only x∈[3,5] while train never saw that region.
    This caused the model to completely fail on the val domain.
    """
    k_pos, k_noise, k_shuffle = jax.random.split(key, 3)

    # Uniform grid over physical domain (NOT sorted — shuffle handles ordering)
    X_raw = jax.random.uniform(k_pos, (cfg.n_samples,),
                               minval=cfg.x_min, maxval=cfg.x_max)

    # Standardise inputs — critical for gradient stability
    X_mean = jnp.mean(X_raw)
    X_std  = jnp.std(X_raw)
    X      = (X_raw - X_mean) / X_std

    # Ground truth wavefunction
    Y_true  = jnp.sin(3.0 * X_raw) * jnp.exp(-0.2 * X_raw ** 2)

    # Add Gaussian measurement noise
    Y_noisy = Y_true + jax.random.normal(k_noise, (cfg.n_samples,)) * cfg.noise_std

    # ── SHUFFLE before split so both train and val cover the full x domain ──
    perm    = jax.random.permutation(k_shuffle, cfg.n_samples)
    X_raw   = X_raw[perm]
    X       = X[perm]
    Y_noisy = Y_noisy[perm]

    # Train / validation split
    n_val   = int(cfg.n_samples * cfg.val_fraction)
    n_train = cfg.n_samples - n_val

    X_train_raw, X_val_raw  = X_raw[:n_train],   X_raw[n_train:]
    X_train,     X_val      = X[:n_train],        X[n_train:]
    Y_train,     Y_val      = Y_noisy[:n_train],  Y_noisy[n_train:]

    return (
        X_train.reshape(-1, 1), Y_train.reshape(-1, 1),
        X_val.reshape(-1, 1),   Y_val.reshape(-1, 1),
        X_train_raw,            X_val_raw,
        X_mean,                 X_std
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
class QuantumAnsatz(nn.Module):
    """
    Feedforward MLP ansatz for wavefunction regression.

    Architecture:
        Input(1) → FourierEmbed(32) → [Dense → tanh] × len(hidden_dims) → Dense(1)

    Key design choices:
    ──────────────────
    1. Fourier input embedding: Instead of feeding raw x, we map x to
       [sin(kx), cos(kx)] for k = 1..K. This directly encodes the periodicity
       of sin(3x) into the input representation, bypassing the spectral bias
       problem (MLPs learn low-frequency components much faster than high-freq).

    2. tanh hidden activations: smooth, bounded, sign-aware — suited to
       oscillating wavefunctions. relu produces piecewise-linear artifacts.

    3. No dropout: with N=500 and mini-batches, dropout inflates train loss
       above val loss (inverted gap). L2 weight decay (via adamw) is sufficient.

    4. Xavier/Glorot initialization: maintains variance across layers for tanh.
    """
    hidden_dims:  Sequence[int]
    dropout_rate: float = 0.0           # kept for API compatibility; set to 0
    n_fourier:    int   = 16            # number of Fourier frequency pairs

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # ── Fourier Input Embedding ──────────────────────────────────────────
        # Learnable frequencies initialized near the target frequency (3 rad/unit)
        # Shape: x (N,1) → freqs (n_fourier,) → features (N, 2*n_fourier)
        freqs = self.param(
            'fourier_freqs',
            lambda rng, shape: jax.random.uniform(rng, shape, minval=0.5, maxval=6.0),
            (self.n_fourier,)
        )
        # Broadcast: (N,1) * (n_fourier,) → (N, n_fourier)
        angles = x * freqs[None, :]
        x = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        # Shape: (N, 2*n_fourier)

        # ── Hidden layers ────────────────────────────────────────────────────
        for dim in self.hidden_dims:
            x = nn.Dense(
                features=dim,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros
            )(x)
            x = nn.tanh(x)

        # ── Output layer (linear — regression to scalar amplitude) ───────────
        x = nn.Dense(
            features=1,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros
        )(x)
        return x

# ─────────────────────────────────────────────────────────────────────────────
# 5. Custom TrainState (carries dropout key)
# ─────────────────────────────────────────────────────────────────────────────
class WavefunctionTrainState(train_state.TrainState):
    """
    Extends Flax TrainState.
    Dropout removed — weight decay alone handles regularization for N=500.
    Kept as a subclass for forward compatibility (easy to re-add dropout key).
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 6. Loss Function
# ─────────────────────────────────────────────────────────────────────────────
def mse_loss(params, state, X, Y, training: bool) -> jnp.ndarray:
    """
    Mean Squared Error: L = (1/N) Σ (Ψ_pred − Ψ_true)²
    Uses ½ MSE convention (matches analytic gradient scaling).
    No dropout → no PRNG key needed in apply.
    """
    Y_pred = state.apply_fn({'params': params}, X, training=training)
    return jnp.mean(0.5 * (Y_pred - Y) ** 2)

# ─────────────────────────────────────────────────────────────────────────────
# 7. JIT-compiled Training and Evaluation Steps
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def train_step(
    state: WavefunctionTrainState,
    X_batch: jnp.ndarray,
    Y_batch: jnp.ndarray
) -> Tuple[WavefunctionTrainState, jnp.ndarray]:
    """Single gradient descent step. Returns updated state and batch loss."""
    loss, grads = jax.value_and_grad(mse_loss)(
        state.params, state, X_batch, Y_batch, training=True
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


@jax.jit
def eval_step(
    state: WavefunctionTrainState,
    X: jnp.ndarray,
    Y: jnp.ndarray
) -> jnp.ndarray:
    """Deterministic evaluation (no dropout). Returns validation loss."""
    return mse_loss(state.params, state, X, Y, training=False)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Mini-batch Iterator
# ─────────────────────────────────────────────────────────────────────────────
def get_batches(key, X, Y, batch_size):
    """Yield (X_batch, Y_batch) tuples from a random permutation."""
    n = X.shape[0]
    perm = jax.random.permutation(key, n)
    X_shuf, Y_shuf = X[perm], Y[perm]
    for start in range(0, n, batch_size):
        yield X_shuf[start:start + batch_size], Y_shuf[start:start + batch_size]

# ─────────────────────────────────────────────────────────────────────────────
# 9. Build Optimizer
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(cfg: Config, n_train: int) -> optax.GradientTransformation:
    """
    Linear warmup → cosine decay schedule + gradient clipping + AdamW.

    Warmup rationale: without it, the first few gradient steps use a large LR
    on randomly initialized weights, causing large loss spikes that can destabilize
    early training (visible as the sharp jump at epoch 0 in earlier plots).

    Cosine decay: smoothly reduces LR to 1% of peak, allowing the optimizer to
    settle into the minimum rather than oscillating around it.
    """
    steps_per_epoch  = max(1, n_train // cfg.batch_size)
    total_steps      = cfg.max_epochs * steps_per_epoch
    warmup_steps     = cfg.warmup_epochs * steps_per_epoch

    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=cfg.learning_rate,
                transition_steps=warmup_steps
            ),
            optax.cosine_decay_schedule(
                init_value=cfg.learning_rate,
                decay_steps=total_steps - warmup_steps,
                alpha=0.01          # final LR = 1% of peak
            )
        ],
        boundaries=[warmup_steps]
    )

    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(schedule, weight_decay=cfg.weight_decay)
    )

# ─────────────────────────────────────────────────────────────────────────────
# 10. Training Loop with Early Stopping + Best-Model Checkpointing
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: Config):
    """Full training pipeline. Returns (best_state, history)."""

    # ── PRNG key splitting — one key per concern ──────────────────────────
    master_key                         = jax.random.PRNGKey(cfg.master_seed)
    data_key, init_key, dropout_key, batch_key = jax.random.split(master_key, 4)

    # ── Data ─────────────────────────────────────────────────────────────
    (X_train, Y_train,
     X_val,   Y_val,
     X_train_raw, X_val_raw,
     X_mean, X_std) = generate_data(data_key, cfg)

    print(f"Dataset:  {X_train.shape[0]} train  |  {X_val.shape[0]} val")

    # ── Model ─────────────────────────────────────────────────────────────
    model = QuantumAnsatz(
        hidden_dims=cfg.hidden_dims,
        dropout_rate=cfg.dropout_rate,
        n_fourier=16
    )

    variables = model.init(
        {'params': init_key},
        jnp.ones((1, 1)),
        training=False
    )
    params = variables['params']

    print("\nArchitecture (Fourier embedding + MLP):")
    for layer_name, lp in sorted(params.items()):
        if layer_name == 'fourier_freqs':
            print(f"  {'fourier':10s}  freqs shape: {lp.shape}")
        else:
            print(f"  {layer_name:10s}  kernel: {lp['kernel'].shape}  "
                  f"bias: {lp['bias'].shape}")


    # ── Optimizer + State ─────────────────────────────────────────────────
    optimizer = build_optimizer(cfg, X_train.shape[0])

    state = WavefunctionTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    best_val_loss  = float('inf')
    best_params    = params
    patience_count = 0
    t0             = time.time()

    print(f"\nBeginning training (max {cfg.max_epochs} epochs, "
          f"patience={cfg.patience}) ...")
    print(f"{'Epoch':>7}  {'Train Loss':>12}  {'Val Loss':>12}  {'Status'}")
    print("─" * 55)

    for epoch in range(cfg.max_epochs):

        # Mini-batch pass over training set
        batch_key, subkey = jax.random.split(batch_key)
        epoch_losses = []
        for X_b, Y_b in get_batches(subkey, X_train, Y_train, cfg.batch_size):
            state, batch_loss = train_step(state, X_b, Y_b)
            epoch_losses.append(float(batch_loss))

        train_loss = float(np.mean(epoch_losses))
        val_loss   = float(eval_step(state, X_val, Y_val))

        # ── Best-model checkpoint ─────────────────────────────────────────
        improved = val_loss < (best_val_loss - cfg.min_delta)
        if improved:
            best_val_loss  = val_loss
            best_params    = jax.tree_util.tree_map(lambda x: x.copy(), state.params)
            patience_count = 0
            status = "✓ best"
        else:
            patience_count += 1
            status = f"  ({patience_count}/{cfg.patience})"

        # ── Logging ───────────────────────────────────────────────────────
        if epoch % cfg.log_every == 0 or improved:
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            print(f"{epoch:>7d}  {train_loss:>12.6f}  {val_loss:>12.6f}  {status}")

        # ── Early stopping ────────────────────────────────────────────────
        if patience_count >= cfg.patience:
            print(f"\n⚡ Early stopping at epoch {epoch}  "
                  f"(best val loss: {best_val_loss:.6f})")
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s  |  "
          f"Best val loss: {best_val_loss:.6f}")

    # Restore best weights into state
    best_state = state.replace(params=best_params)

    return best_state, history, X_train, Y_train, X_val, Y_val, \
           X_train_raw, X_val_raw, X_mean, X_std


# ─────────────────────────────────────────────────────────────────────────────
# 11. Evaluation Utilities
# ─────────────────────────────────────────────────────────────────────────────
def predict(state: WavefunctionTrainState, X: jnp.ndarray) -> jnp.ndarray:
    """Run deterministic inference."""
    return state.apply_fn({'params': state.params}, X, training=False)


def residuals(Y_pred, Y_true):
    return Y_pred.squeeze() - Y_true.squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# 12. Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(best_state, history,
                 X_train, Y_train,
                 X_val,   Y_val,
                 X_train_raw, X_val_raw,
                 X_mean, X_std):
    """Four-panel diagnostic figure."""

    # Predictions on train + val
    Y_pred_train = predict(best_state, X_train)
    Y_pred_val   = predict(best_state, X_val)

    # Dense grid for smooth curve — the key visual check
    X_grid_raw  = jnp.linspace(-5, 5, 1000)
    X_grid      = ((X_grid_raw - X_mean) / X_std).reshape(-1, 1)
    Y_pred_grid = predict(best_state, X_grid)
    Y_true_grid = jnp.sin(3 * X_grid_raw) * jnp.exp(-0.2 * X_grid_raw ** 2)

    # Sort train/val by x_raw for clean residual line plots
    train_sort = jnp.argsort(X_train_raw)
    val_sort   = jnp.argsort(X_val_raw)

    X_train_raw_s = X_train_raw[train_sort]
    X_val_raw_s   = X_val_raw[val_sort]
    res_train = (Y_pred_train.squeeze()[train_sort]
                 - (jnp.sin(3*X_train_raw_s)*jnp.exp(-0.2*X_train_raw_s**2)))
    res_val   = (Y_pred_val.squeeze()[val_sort]
                 - (jnp.sin(3*X_val_raw_s)*jnp.exp(-0.2*X_val_raw_s**2)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Quantum Wavefunction Approximation — JAX / Flax / Optax",
                 fontsize=14, fontweight='bold')

    # ── Panel 1: Function approximation ──────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(X_train_raw, Y_train, color='steelblue', alpha=0.3, s=18,
               label='Train (noisy)')
    ax.scatter(X_val_raw,   Y_val,   color='orange',    alpha=0.5, s=18,
               label='Val (noisy)')
    ax.plot(X_grid_raw, Y_true_grid,  'k--', lw=2,   label=r'True $\Psi(x)$')
    ax.plot(X_grid_raw, Y_pred_grid,  'r-',  lw=2.5, label='MLP prediction')
    ax.set_title("Universal Approximation")
    ax.set_xlabel(r"Position $x$")
    ax.set_ylabel(r"Amplitude $\Psi(x)$")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ── Panel 2: Loss curves ──────────────────────────────────────────────
    ax = axes[0, 1]
    ax.semilogy(history['epoch'], history['train_loss'],
                'b-o', ms=4, lw=1.5, label='Train loss')
    ax.semilogy(history['epoch'], history['val_loss'],
                'r-s', ms=4, lw=1.5, label='Val loss')
    ax.set_title("Loss Convergence (log scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("½ MSE Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # ── Panel 3: Residuals (train) ────────────────────────────────────────
    ax = axes[1, 0]
    ax.scatter(X_train_raw_s, res_train, color='steelblue', alpha=0.5, s=15)
    ax.plot(X_train_raw_s, res_train, color='steelblue', alpha=0.3, lw=0.8)
    ax.axhline(0, color='k', lw=1.2)
    ax.set_title("Residuals — Train Set  (want: white noise around 0)")
    ax.set_xlabel(r"Position $x$")
    ax.set_ylabel(r"$\Psi_{pred} - \Psi_{true}$")
    ax.grid(True, linestyle='--', alpha=0.5)

    # ── Panel 4: Residuals (val) ──────────────────────────────────────────
    ax = axes[1, 1]
    ax.scatter(X_val_raw_s, res_val, color='orange', alpha=0.6, s=15)
    ax.plot(X_val_raw_s, res_val, color='orange', alpha=0.3, lw=0.8)
    ax.axhline(0, color='k', lw=1.2)
    ax.set_title("Residuals — Validation Set  (want: white noise around 0)")
    ax.set_xlabel(r"Position $x$")
    ax.set_ylabel(r"$\Psi_{pred} - \Psi_{true}$")
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("wavefunction_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure saved to wavefunction_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    (best_state, history,
     X_train, Y_train,
     X_val,   Y_val,
     X_train_raw, X_val_raw,
     X_mean, X_std) = train(cfg)

    plot_results(best_state, history,
                 X_train, Y_train,
                 X_val,   Y_val,
                 X_train_raw, X_val_raw,
                 X_mean, X_std)
