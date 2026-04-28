""" 
GW Signal Separation - Model
=============================

Complex-domain encoder-bottleneck-decoder architecture.

Input  : complex64 (B, T, F)  = (B, 62, 126) - whitened mixture spectrogram
Output : complex64 (B, N, T, F) = (B, 2, 62, 126) - separated sources

Architecture:
    ComplexLinear  : W = W_r + iW_i applied to z = z_r + iz_i
                     Output = (W_r z_r - W_i z_i) + i(W_r z_i + W_i z_r)
                     Preserves phase: critical for GW separation
    
    modReLU : ReLU(|z| + b) * z/|z|
              Gates magnitude with learned bias, phase unchanged
              Only activation that respects complex geometry
    
    ComplexEncoder : Stack of ComplexLinear + modReLU
                     Compresses (B, T, F) -> (B, T, d)
                     
    AttentionBottleneck : N learnable queries attend over encoder output
                          Each query learns to track one source chirp
                          Output: (B, N, T, d) - N source tracks
    
    ComplexDecoder : Each source track -> complex ratio mask -> (B, N, T, F)
                     Mask * mixture = separated signal
                     
Parameters:
    n_sources  = 2 ; number of overlapping signals to separate
    n_freqs = 126 ; frequency bins in input (20 - 1024 Hz)
    encoder_dims = [256, 256] ; hidden dimentions in encoder layers
    decoder_dims = [256, 256] ; hidden dimentions in decoder layers
    latent_dim = 128 ; bottleneck dimension d
    n_heads = 4 ; attention heads in bottleneck
"""
# -- 1. Setup and Imports --
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

# -- 2. Complex Linear Layer --
class ComplexLinear(nn.Module):
    """
    Linear layer for complex inputs.
    
    W = W_r + iW_i where W_r, W_i are independent real matrices
    Output : (W_r z_r - W_i z_i) + i(W_r z_i + W_i z_r)
    """
    features: int
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z_r = z.real
        z_i = z.imag
        
        W_r = nn.Dense(self.features, use_bias=self.use_bias, name="W_r")
        W_i = nn.Dense(self.features, use_bias=self.use_bias, name="W_i")
        
        out_r = W_r(z_r) - W_i(z_i)
        out_i = W_r(z_i) + W_i(z_r)
        
        return out_r + 1j * out_i

# -- 3. modReLU --
class ModReLU(nn.Module):
    """
    modReLU activation for complex inputs.
    
    modReLU (z) = ReLU(|z| + b) * z / |z|
    -> b is a learned bias per feature (initialized to zero)
    -> Gates the magnitude while leaving phase completely unchanged
    -> When |z| + b <= 0: output is 0 (neuron "off")
    -> when |z| + b > 0: output has same phase as input, scaled magnitude
    """
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        b = self.param("bias", nn.initializers.zeros, (z.shape[-1],))
        magnitude = jnp.abs(z)
        phase = z / (magnitude + 1e-8)
        gated_mag = nn.relu(magnitude + b)
        return gated_mag * phase
    
# -- 4. Complex Encoder --
class ComplexEncoder(nn.Module):
    """
    Stack of ComplexLinear + modReLU layers.
    
    Compresses the frequency-time mixture spectrogram into a 
    compact latent representation while preserving phase.
    
    (B, T, F) -> (B, T, latent_dim)
    """
    hidden_dims: Sequence[int]
    latent_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = x
        for dim in self.hidden_dims:
            z = ComplexLinear(dim)(z)
            z = ModReLU()(z)
        z = ComplexLinear(self.latent_dim)(z)
        return z
    
# -- 5. Attention Bottleneck --
class AttentionBottleneck(nn.Module):
    """
    N learnable query tracks attend over the encoder output.
    
    Each query vector learns to "focus on" time-frequecy tiles
    belonging to one source. The attention is computed on signal
    magnitude (real-valued scores) while the values being aggregated
    remain complex - preserving phase for the decoder.
    
    (B, T, d) -> (B, N, T, d)
    """
    n_sources : int
    latent_dim : int
    n_heads : int = 4
    
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        B, T, d = z.shape
        
        # N learnable source identity vectors
        # Each query "asks": which time-frequency tiles belong to me?
        queries = self.param("source_queries",
                             nn.initializers.normal(0.02),
                             (self.n_sources, d))
        queries = jnp.broadcast_to(queries[None], (B, self.n_sources, d))
        
        # Keys from encoder magnitude (real-valued attention scores)
        z_mag = jnp.abs(z)
        key_proj = nn.Dense(d, name="Key_proj")(z_mag)   # (B, T, d)
        q_proj = nn.Dense(d, name="query_proj")(queries) # (B, N, d)
        
        # Scaled dot-product attention: (B, N, T)
        scale = jnp.sqrt(d).astype(jnp.float32)
        scores = jnp.einsum("bnd, btd -> bnt", q_proj, key_proj) / scale
        attn_weights = nn.softmax(scores, axis=-1)
        
        # Aggregate complex values with real attention weights
        # (B, N, T) * (B, T, d) -> (B, N, d)
        context = jnp.einsum("bnt, btd->bnd", attn_weights, z)
        
        # Brodcast context over time and add to encoder output
        context = jnp.broadcast_to(context[:, :, None, :],
                                   (B, self.n_sources, T, d))
        z_expanded = jnp.broadcast_to(z[:, None, :, :],
                                      (B, self.n_sources, T, d))
        return z_expanded + 0.5 * context   # (B, N, T, d)
    
# -- 6. Complex Decoder --
class ComplexDecoder (nn.Module):
    """
    Convert a source track into a complex ratio mask
    
    The mask is applied to the mixture spectrogram:
        separaded = mask * mixture
        
    A complex ratio mask (CRM) can adjust both amplitude and phase
    of each frequency bin - unlike a real mask which only adjusts
    amplitude. Phase adjustment is critical for GW signals where
    two sources may overlap in amplitude but differ in phase.
    
    (B, T, d) -> (B, T, F)
    """
    hidden_dims: Sequence[int]
    out_dim : int
    
    @nn.compact
    def __call__(self, track: jnp.ndarray) -> jnp.ndarray:
        z = track
        for dim in self.hidden_dims:
            z = ComplexLinear(dim)(z)
            z = ModReLU()(z)
        return ComplexLinear(self.out_dim)(z)
    
# -- 7. GW Separator --
class GWSeparator(nn.Module):
    """
    Full GW signal separation model.
    
    Forward pass:
        mixture (B, T, F) -> (B, T, d) -> bottleneck (B, N, T, F)
        -> decoder (B, N, T, F) masks -> mask * mixture -> separated (B, N, T, F)
    """
    n_sources : int
    n_freqs : int
    encoder_dims: Sequence[int]
    decoder_dims: Sequence[int]
    latent_dim : int
    n_heads : int = 4
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """

        Args:
            x : complex64(B, T, F) -> whitened mixture spectrogram

        Returns:
            complex64 (B, N, T, F) -> N separated spectrograms
        """
        # Encode mixture
        z = ComplexEncoder(
            hidden_dims = self.encoder_dims,
            latent_dim = self.latent_dim
        )(x)
        # z: (B, T, latent_dim)
        
        # Separate into N source tracks via attention
        tracks = AttentionBottleneck(
            n_sources = self.n_sources,
            latent_dim = self.latent_dim,
            n_heads = self.n_heads
        )(z)
        # tracks: (B, N, T, latent_dim)
        
        # Decode each track into a complex mask
        B, N, T, d = tracks.shape
        tracks_flat = tracks.reshape(B * N, T, d)
        
        masks_flat = ComplexDecoder(
            hidden_dims = self.decoder_dims,
            out_dim = self.n_freqs
        )(tracks_flat)
        # masks_flat: (B*N, T, F)
        
        masks = masks_flat.reshape(B, N, T, self.n_freqs)
        # masks: (B, N, T, F)
        
        # Apply complex ratio mask to mixture
        x_expanded = jnp.broadcast_to(x[:, None, :, :], masks.shape)
        masks      = jnp.tanh(masks.real) + 1j * jnp.tanh(masks.imag)
        return masks * x_expanded   # (B, N, T, F)

# -- 8. Shape check --   
if __name__ == "__main__":
    import jax.random as jr
    from load import N_FRAMES, N_FREQ

    B, N = 2, 2
    model = GWSeparator(
        n_sources=N,
        n_freqs=N_FREQ,
        encoder_dims=[256, 256],
        decoder_dims=[256, 256],
        latent_dim=128,
        n_heads=4,
    )

    key = jr.PRNGKey(0)

    # Generate complex input
    x_r = jr.normal(key, (B, N_FRAMES, N_FREQ))
    x_i = jr.normal(jr.fold_in(key, 1), (B, N_FRAMES, N_FREQ))
    x = (x_r + 1j * x_i).astype(jnp.complex64)

    # Initialize model parameters
    params = model.init(key, x)

    # Forward pass
    out = model.apply(params, x)

    # Print shapes
    print(f"Input  : {x.shape} {x.dtype}")
    print(f"Output : {out.shape} {out.dtype}")
    print(f"Expected: ({B}, {N}, {N_FRAMES}, {N_FREQ})")

    # Count parameters
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {n_params:,}")

    # Assertions
    assert out.shape == (B, N, N_FRAMES, N_FREQ)
    assert out.dtype == jnp.complex64

    print("\nAll checks passed.")