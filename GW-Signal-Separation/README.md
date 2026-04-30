# Gravitational Wave Signal Separation using Physics-Informed Deep Learning

This project implements an end-to-end pipeline for separating overlapping gravitational-wave (GW) signals from noisy observations using a complex-valued neural network trained in the time–frequency domain.

---

## 1. Problem Statement

Gravitational-wave detectors such as LIGO observe signals embedded in noise:

    x(t) = h1(t) + h2(t) + n(t)

Where:
- h1(t), h2(t): signals from independent compact binary coalescences (CBC)
- n(t): detector noise
- x(t): observed mixture

The objective is to recover individual signals (h1, h2) from the mixture.

This is a **source separation problem** under realistic astrophysical conditions.

---

## 2. Dataset Description

### 2.1 Sampling Configuration

- Sampling rate: 4096 Hz  
- Duration: 4 seconds  
- Samples per signal: 16384  

---

### 2.2 Dataset Size

- Total samples: 100,000  
- Number of shards: 10  
- Samples per shard: 10,000  

---

### 2.3 File Format (.h5)

Each shard contains:

| Key        | Shape            | Description |
|------------|-----------------|------------|
| mixture    | (N, 16384)      | noisy signal |
| h1         | (N, 16384)      | first GW signal |
| h2         | (N, 16384)      | second GW signal |
| noise      | (N, 16384)      | detector noise |
| params1    | (N, 11)         | physical parameters (h1) |
| params2    | (N, 11)         | physical parameters (h2) |

---

## 3. Data Generation

### 3.1 Waveform Model

- Approximant: **IMRPhenomD**
- Generated using **PyCBC**
- Includes:
  - Inspiral
  - Merger
  - Ringdown

---

### 3.2 Parameter Sampling

- Masses: 25–100 M☉ (power-law distribution)
- Spins: uniform in [-0.99, 0.99]
- Distance: volumetric distribution (∝ d²)
- Coalescence time: 2–3.5 seconds
- Sky location: isotropic

---

### 3.3 Noise Model

- Gaussian, stationary noise
- Generated from detector PSD (A+ design)
- Frequency-domain noise realization per sample

---

### 3.4 SNR Estimation

Signal strength is evaluated using matched filtering:

    ρ = max | matched_filter(h, d) |

Where:
- h: template waveform
- d = h + n

This ensures SNR is computed in the presence of noise.

:contentReference[oaicite:0]{index=0}

---

## 4. Preprocessing Pipeline

The raw time-domain signals are transformed into a representation suitable for learning.

---

### 4.1 Short-Time Fourier Transform (STFT)

- FFT size: 512  
- Hop length: 128  
- Window: Hann  
- Output shape: (T, F) = (62, 126)

Frequency range used:
- 20 Hz – 1024 Hz (GW-sensitive band)

---

### 4.2 PSD Estimation

- Estimated from mixture
- Smoothed over time frames
- Used for whitening

---

### 4.3 Whitening

Each spectrogram is normalized by noise power:

    X_white(f, t) = X(f, t) / sqrt(PSD(f, t))

This ensures uniform noise statistics across frequencies.

---

### 4.4 Final Input Representation

- Complex-valued spectrogram
- Shape: (T, F)
- Used as input to the model

:contentReference[oaicite:1]{index=1}

---

## 5. Model Architecture

The model operates entirely in the complex domain.

---

### 5.1 Input

    (B, T, F) complex64

---

### 5.2 Components

#### (a) Complex Encoder
- Stack of complex linear layers
- modReLU activation
- Compresses frequency features into latent space

---

#### (b) Attention Bottleneck
- Learnable queries represent sources
- Multi-head attention tracks signal trajectories
- Outputs separate latent tracks per source

---

#### (c) Complex Decoder
- Generates complex masks
- Applies masks to mixture spectrogram

---

### 5.3 Output

    (B, 2, T, F)

Separated spectrograms for two sources

:contentReference[oaicite:2]{index=2}

---

## 6. Loss Function

Training uses a combination of signal-processing and physics-based losses.

---

### 6.1 SI-SNR Loss

Measures reconstruction quality independent of scale.

---

### 6.2 Matched Filter Loss

Based on GW detection statistic:

    ρ = <h_pred, h_true> / sqrt(<h_pred, h_pred><h_true, h_true>)

Loss:

    L = 1 - ρ²

---

### 6.3 Permutation Invariant Training (PIT)

Handles ambiguity in source ordering:

- Evaluates both permutations
- Selects minimum loss

---

### 6.4 Total Loss

    L = SI-SNR + α × Matched Filter Loss

:contentReference[oaicite:3]{index=3}

---

## 7. Training Pipeline

- Framework: JAX + Flax  
- Optimizer: AdamW  
- Learning rate: cosine decay with warmup  
- Batch size: 80  
- Training epochs: 50  

Training loop:
- Load batches from shards
- Compute forward pass
- Apply PIT loss
- Update parameters
- Save checkpoints

:contentReference[oaicite:4]{index=4}

---

## 8. Inference and Signal Recovery

Model outputs are post-processed to reconstruct time-domain signals.

---

### 8.1 Steps

1. Unwhitening:
       X_phys = X_pred × sqrt(PSD)

2. Zero-padding to full frequency range

3. Inverse STFT (overlap-add)

---

### 8.2 Output

Recovered signals:

    h1(t), h2(t)

---

### 8.3 Evaluation

- Time-domain comparison
- Spectrogram visualization
- Residual analysis

:contentReference[oaicite:5]{index=5}

---

## 9. Data Conversion

For efficient training:

- Convert `.h5` → `.npy`

Storage format:

- signal_*.npy → (mixture, h1, h2)
- params_*.npy → physical parameters

:contentReference[oaicite:6]{index=6}

---

## 10. Key Contributions

- Complex-valued neural network for GW separation
- Integration of matched filtering into loss function
- Attention-based multi-source tracking
- End-to-end pipeline from simulation to reconstruction

---

## 11. Limitations

- Single detector (no network analysis)
- Gaussian noise assumption
- No real detector artifacts (glitches)
- Limited to two overlapping sources

---

## 12. Applications

- Gravitational-wave source separation
- Multi-source signal recovery
- Noise-robust astrophysical inference
- Machine learning for time-series physics

---

## 13. Author

Samin Hasan  
MSc Physics, IIT Hyderabad
