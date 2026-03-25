# Wavefunction Approximation with JAX / Flax / Optax

Fits a neural network to the quantum harmonic oscillator wavefunction:

$$\Psi(x) = \sin(3x) \cdot e^{-0.2x^2}$$

## Architecture

- **Fourier input embedding** — maps x to [sin(fᵢx), cos(fᵢx)] with learnable
  frequencies, bypassing spectral bias for high-frequency targets
- **MLP** — [256 → 256 → 128 → 64] hidden layers with tanh activations
- **Linear output** — scalar amplitude regression

## Training

- Optimizer: AdamW with linear warmup (300 epochs) → cosine decay
- Regularization: L2 weight decay (no dropout — too aggressive at N=500)
- Early stopping with best-model checkpointing
- 80/20 train/val split with random shuffle

## Results

| Metric | Value |
|---|---|
| Final train loss (½ MSE) | ~1×10⁻³ |
| Final val loss (½ MSE) | ~2×10⁻³ |
| Max residual | ±0.03 (below noise floor of σ=0.05) |

## Run

```bash
pip install -r requirements.txt
python mlp_flax_robust.py
