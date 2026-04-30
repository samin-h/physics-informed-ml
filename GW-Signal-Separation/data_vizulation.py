import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm

from pycbc.noise.gaussian import frequency_noise_from_psd
from pycbc.waveform import get_fd_waveform, get_waveform_filter_length_in_time as chirplen
from pycbc.waveform.generator import (FDomainDetFrameGenerator, FDomainCBCGenerator)
from pycbc.filter.matchedfilter import matched_filter
from pycbc.psd.read import from_txt

# -------- Constants -----------

SAMPLE_RATE = 4096
DURATION = 4
N_SAMPLES_TIME = SAMPLE_RATE * DURATION

F_LOW = 20.0
F_HIGH = 1024.0

SNR_MIN = 8.0
DELTA_T = 1.0 / SAMPLE_RATE
DELTA_F = 1.0 / DURATION

M_MIN = 25
M_MAX = 100
ALPHA = 3.5

DL_MIN = 1
DL_MAX = 10000

APPROXIMANT = "IMRPhenomD"


# -------- Sampling -----------

def sample_masses(rng):
    r = rng.uniform()
    exponent = 1.0 - ALPHA
    norm = (M_MAX ** exponent - M_MIN ** exponent)
    m1 = (r * norm + M_MIN ** exponent) ** (1. / exponent)
    m2 = rng.uniform(M_MIN, m1)
    return m1, m2


def sample_spin(rng):
    return rng.uniform(-0.99, 0.99)


def sample_distance(rng):
    u = rng.uniform()
    norm = DL_MAX**3 - DL_MIN**3
    return ((u * norm) + DL_MIN**3)**(1 / 3)

def sample_tc(rng):
    tc = rng.uniform(2.0, 3.5)
    return tc


def sample_sky(rng):
    iota = np.arccos(2*rng.uniform() - 1)
    psi = 2*np.pi*rng.uniform()
    ra = rng.uniform(0, 2*np.pi)
    dec = np.arcsin(rng.uniform(-1, 1))
    return iota, psi, ra, dec
        


# -------- Waveform -----------

def generate_waveform(params, psd):
    m1, m2, chi1z, chi2z, distance, tc, iota, psi, ra, dec = params

    seglen = DURATION

    delta_f = 1 / seglen
    Nt = int(SAMPLE_RATE * seglen)
    Nf = Nt // 2 + 1

    tStart = 0.0
    tTrigger = tc

    frozen = dict(
        approximant=APPROXIMANT,
        mass1=m1, mass2=m2,
        spin1z=chi1z, spin2z=chi2z,
        delta_f=delta_f,
        f_lower=F_LOW,
        f_final=F_HIGH,
        inclination=iota,
        distance=distance,
        coa_phase=0,
        ra=ra, dec=dec,
        tc=tTrigger,
        polarization=psi
    )

    generator = FDomainDetFrameGenerator(
        FDomainCBCGenerator,
        detectors=['H1'],
        epoch=tStart,
        variable_args=['ra', 'dec', 'tc', 'polarization'],
        **frozen
    )

    signal_dict = generator.generate(ra=ra, dec=dec, tc=tTrigger, polarization=psi)
    signal_fd = signal_dict['H1']  

    # Resize PSD
    psd = psd.copy()
    psd.resize(len(signal_fd))

    # # Noise (FD)
    noise_fd = frequency_noise_from_psd(psd)
    noise_fd.resize(len(signal_fd))

    data_fd = signal_fd + noise_fd

    # Template
    hp, _ = get_fd_waveform(**frozen)
    hp.resize(len(signal_fd))

    snr_ts = matched_filter(hp, data_fd, psd=psd,
                        low_frequency_cutoff=F_LOW)

    snr = float(abs(snr_ts).max())

    # Convert to time domain
    signal_td = signal_fd.to_timeseries(delta_t=DELTA_T)
    #noise_td = noise_fd.to_timeseries(delta_t=DELTA_T)

    # Pad/trim to fixed length
    signal_td.resize(N_SAMPLES_TIME)
    #noise_td.resize(N_SAMPLES_TIME)

    return signal_td.numpy(), snr


# -------- Sample -----------

def generate_sample(rng, psd):

    p1 = (
        *sample_masses(rng),
        sample_spin(rng),
        sample_spin(rng),
        sample_distance(rng),
        sample_tc(rng),
        *sample_sky(rng)
    )

    h1, snr1 = generate_waveform(p1, psd)
    
    if snr1 < SNR_MIN: 
        snr1 = max(snr1, 1e-6)
        # Target SNR
        target_snr1 = rng.uniform(6, 15)

        # Rescale distance
        p1 = list(p1)
        p1[4] *= snr1 / target_snr1   # index 4 = distance
        p1 = tuple(p1)
        
        h1, snr1 = generate_waveform(p1, psd)

    p2 = (
        *sample_masses(rng),
        sample_spin(rng),
        sample_spin(rng),
        sample_distance(rng),
        sample_tc(rng),
        *sample_sky(rng)
    )

    h2, snr2 = generate_waveform(p2, psd)
    if snr2 < SNR_MIN:
        snr2 = max(snr2, 1e-6)
        
        # Target SNR
        target_snr2 = rng.uniform(6, 15)

        # Rescale distance
        p2 = list(p2)
        p2[4] *= snr2 / target_snr2   # index 4 = distance
        p2 = tuple(p2)
        
        h2, snr2 = generate_waveform(p2, psd)
        
    # ------ Noise (independent) --------
    noise_fd = frequency_noise_from_psd(psd)
    noise_fd.resize(N_SAMPLES_TIME // 2 + 1)
    noise_td = noise_fd.to_timeseries(delta_t=DELTA_T)
    noise_td.resize(N_SAMPLES_TIME)
    noise = noise_td.numpy()

    mixture = h1 + h2 + noise

    params1 = np.array((*p1, snr1))
    params2 = np.array((*p2, snr2))

    return {
        "h1": h1,
        "h2": h2,
        "noise": noise,
        "mixture": mixture,
        "params1": params1,
        "params2": params2
    }


# -------- Shard -----------

def write_shard(idx, n_per_shard, outdir, seed, psd):

    rng = np.random.default_rng(seed)

    H1 = np.zeros((n_per_shard, N_SAMPLES_TIME))
    H2 = np.zeros((n_per_shard, N_SAMPLES_TIME))
    NOISE = np.zeros((n_per_shard, N_SAMPLES_TIME))
    MIX = np.zeros((n_per_shard, N_SAMPLES_TIME))

    PARAMS1 = np.zeros((n_per_shard, 11))
    PARAMS2 = np.zeros((n_per_shard, 11))

    for i in tqdm(range(n_per_shard), desc=f"Shard {idx}"):
        s = generate_sample(rng, psd)

        H1[i] = s["h1"]
        H2[i] = s["h2"]
        NOISE[i] = s["noise"]
        MIX[i] = s["mixture"]

        PARAMS1[i] = s["params1"]
        PARAMS2[i] = s["params2"]

    path = os.path.join(outdir, f"shard_{idx:04d}.h5")

    with h5py.File(path, "w") as f:
        kwargs = dict(compression="gzip", compression_opts=4)

        f.create_dataset("h1", data=H1, **kwargs)
        f.create_dataset("h2", data=H2, **kwargs)
        f.create_dataset("noise", data=NOISE, **kwargs)
        f.create_dataset("mixture", data=MIX, **kwargs)
        f.create_dataset("params1", data=PARAMS1, **kwargs)
        f.create_dataset("params2", data=PARAMS2, **kwargs)

    return path


# -------- Main -----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--n-shards", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="./data")
    parser.add_argument("--shard-idx", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    n_per_shard = args.n_samples // args.n_shards

    Nf = N_SAMPLES_TIME // 2 + 1
    psd = from_txt(
        "ASD_AplusDesign_O5.txt",
        length=Nf,
        delta_f=DELTA_F,
        low_freq_cutoff=F_LOW,
        is_asd_file=True
    )

    if args.shard_idx is not None:
        seed = args.base_seed + args.shard_idx
        write_shard(args.shard_idx, n_per_shard, args.outdir, seed, psd)
    else:
        for i in range(args.n_shards):
            seed = args.base_seed + i
            write_shard(i, n_per_shard, args.outdir, seed, psd)


if __name__ == "__main__":
    main()