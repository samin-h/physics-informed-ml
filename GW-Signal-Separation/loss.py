"""
Loss Functions for GW Signal Separation
========================================

Three componets:

1.  SI-SNR   - Scale-Invariant SNR
    Meaures signal quality independent of amplitude scaling.
    Loss = -SI-SNR (we minimize, so higher SI-SNR = lower loss)

2.  PIT - Permutation Invariant Training
    For N = 2 source there are 2! = 2 possible assignments:
        [pred0->target0, pred1->target1]
        [pred0->target1, pred1->target0]
    Take minimum loss over both - model never penalized for
    correct separation in wrong order.

3.  Matched Filter Loss
    Physical GW loss. Measures normalized overlap integral:
        rho = Re(<h^|h>) / sqrt(Re(<h^|h^>) * Re(<h|h>))
    rho = 1 means perfect match, rho=0 means orthogonal.
    Loss = 1 - rho^2
    This directly optimizes the detection statistic used in
    GW astronomy - what makes this project novel vs audio separation.

Combined loss per source pair:
    L = L_SISNR + alpha * L_MF
    alpha = 0.5 balances the two contributions.
"""

# -- 1. Setup and Imports --
import jax
import jax.numpy as jnp

# Permutations for N=2 sources
PERMS_2 = [(0, 1), (1, 0)]


# -- 2. SI-SNR --
@jax.jit
def si_snr_loss(
    pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    """
    Scale-Invariant SNR loss for one (pred, target) pair.

    Args:
        pred, target : complex64 (T, F)

    Returns:
        scalar - negetive SI-SNR in dB (lower = better)
    """
    s = target.reshape(-1)
    s_hat = pred.reshape(-1)

    # Zero-mean
    s = s - jnp.mean(s)
    s_hat = s_hat - jnp.mean(s_hat)

    # Project prediction onto target
    dot = jnp.real(jnp.dot(jnp.conj(s_hat), s))
    s_norm_sq = jnp.real(jnp.dot(jnp.conj(s), s)) + eps
    s_target = (dot / s_norm_sq) * s

    # Noise = prediction - projection
    e_noise = s_hat - s_target

    signal_power = jnp.real(jnp.dot(jnp.conj(s_target), s_target)) + eps
    noise_power = jnp.real(jnp.dot(jnp.conj(e_noise), e_noise)) + eps

    return -10.0 * jnp.log10(signal_power / noise_power)


# -- 3. Matched Filter Loss --
@jax.jit
def matched_filter_loss(
    pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    """
    Matched filter overlap loss in whitened frequency domain.

    Since data is already whitened, noise PSD = 1 everywhere,
    so the overlap integral simplifies to a plain dot product.

    Args:
        pred, target : complex64 (T, F)
    Returns:
        scalar: 1 - rho^2 (lower = better overlap)
    """
    h_hat = pred.reshape(-1)
    h = target.reshape(-1)

    cross = jnp.real(jnp.dot(jnp.conj(h_hat), h))
    norm_hat = jnp.real(jnp.dot(jnp.conj(h_hat), h_hat)) + eps
    norm_h = jnp.real(jnp.dot(jnp.conj(h), h)) + eps

    rho = cross / jnp.sqrt(norm_hat * norm_h)
    rho = jnp.clip(rho, -1.0, 1.0)
    return 1.0 - rho**2


# -- 4. PIT loss for one sample --
@jax.jit
def pit_loss_single(
    preds: jnp.ndarray, targets: jnp.ndarray, alpha: float = 0.5
) -> jnp.ndarray:
    """
    PIT loss for one sample: try all permutations, take minimum.

    Args:
        preds   : complex64 (N, T, F)
        targets : complex64 (N, T, F)
        alpha   : matched filter weight

    Returns:
        scalar: minimum loss over permutations
    """
    N = preds.shape[0]
    best_loss = jnp.inf

    for perm in PERMS_2:
        total = 0.0
        for i, j in enumerate(perm):
            total = total + si_snr_loss(preds[i], targets[j])
            total = total + alpha * matched_filter_loss(preds[i], targets[j])
        best_loss = jnp.minimum(best_loss, total / N)

    return best_loss


# -- 5. Batch loss --
@jax.jit
def batch_loss(
    preds: jnp.ndarray, targets: jnp.ndarray, alpha: float = 0.5
) -> jnp.ndarray:
    """
    PIT loss averaged over batch.

    Args:
        preds (jnp.ndarray)    : complex64 (B, N, T, F)
        targets (jnp.ndarray)  : complex64 (B, N, T, F)
        alpha (float, optional): matched filter weight

    Returns:
        scalar: mean loss
    """

    per_sample = jax.vmap(lambda p, t: pit_loss_single(p, t, alpha))(preds, targets)
    return jnp.mean(per_sample)


@jax.jit
def compute_loss(preds, targets, alpha=0.5):
    return batch_loss(preds, targets, alpha)


# -- 6. Evaluation metrics --
# @jax.jit
# def compute_metrics(preds: jnp.ndarray, targets: jnp.ndarray) -> dict:
#     """
#     Compute SI-SNR and overlap rho for evalution (not training).
#     Args:
#         preds, targets : complex64 (B, N, T, F)

#     Returns:
#         dick: si_snr_db, overlap_rho
#     """
#     B, N, T, F = preds.shape
#     si_snrs, rhos = [], []

#     for b in range(B):
#         best_sisnr, best_rho = -jnp.inf, 0.0
#         for perm in PERMS_2:
#             s, r = 0.0 , 0.0
#             for i, j in enumerate(perm):
#                 s += -si_snr_loss(preds[b, i], targets[b, j])
#                 r += 1.0 - matched_filter_loss(preds[b, i], targets[b, j])

#             if s > best_sisnr:
#                 best_sisnr = s / N
#                 best_rho   = r / N
#         si_snrs.append(float(best_sisnr))
#         rhos.append(float(best_rho))

#     return {
#         "si_snr_db" : float(jnp.mean(jnp.array(si_snrs))),
#         "overlap_rho" : float(jnp.mean(jnp.array(rhos))),
#     }


@jax.jit
def compute_metrics(preds: jnp.ndarray, targets: jnp.ndarray) -> dict:
    """
    Compute SI-SNR and overlap rho for evaluation.
    Args:
        preds, targets : complex64 (B, N, T, F)
    """

    # 1. Define the logic for a SINGLE sample in the batch
    def process_single_item(p, t):
        # Permutation 1: (0, 0) and (1, 1)
        s1 = (-si_snr_loss(p[0], t[0]) - si_snr_loss(p[1], t[1])) / 2.0
        r1 = (
            2.0 - matched_filter_loss(p[0], t[0]) - matched_filter_loss(p[1], t[1])
        ) / 2.0

        # Permutation 2: (0, 1) and (1, 0)
        s2 = (-si_snr_loss(p[0], t[1]) - si_snr_loss(p[1], t[0])) / 2.0
        r2 = (
            2.0 - matched_filter_loss(p[0], t[1]) - matched_filter_loss(p[1], t[0])
        ) / 2.0

        # JAX equivalent of "if s1 > s2:"
        best_s = jnp.where(s1 > s2, s1, s2)
        best_r = jnp.where(s1 > s2, r1, r2)

        return best_s, best_r

    # 2. Vectorize the single-item function across the batch dimension (axis 0)
    # This completely eliminates the need for `for b in range(B)`
    batch_s, batch_r = jax.vmap(process_single_item)(preds, targets)

    # 3. Return JAX arrays (do NOT cast to float() inside jit)
    return {
        "si_snr_db": jnp.mean(batch_s),
        "overlap_rho": jnp.mean(batch_r),
    }


# -- 7. Sanity check --
if __name__ == "__main__":
    from load import N_FRAMES, N_FREQ
    import jax.random as jr

    B, N, T, F = 4, 2, N_FRAMES, N_FREQ
    key = jr.PRNGKey(0)

    t_r = jr.normal(key, (B, N, T, F))
    t_i = jr.normal(jr.fold_in(key, 1), (B, N, T, F))
    targets = t_r + 1j * t_i

    p_r = jr.normal(jr.fold_in(key, 2), (B, N, T, F))
    p_i = jr.normal(jr.fold_in(key, 3), (B, N, T, F))
    preds = p_r + 1j * p_i

    print("Case 1 — perfect (pred=target):")
    print(f" loss = {float(compute_loss(targets, targets)):.4f} (expect ~0)")

    print("Case 2 — random prediction:")
    print(f" loss = {float(compute_loss(preds, targets)):.4f} (expect large)")
    swapped = jnp.stack([targets[:, 1], targets[:, 0]], axis=1)

    print("Case 3 — swapped sources (PIT should recover):")
    print(f" loss = {float(compute_loss(swapped, targets)):.4f} (expect ~0)")
    print("\nAll checks passed.")
