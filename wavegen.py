#!/usr/bin/env python3
"""
wavegen.py - Modular waveform generator for SDR (USRP X310, HackRF, etc.)

Generates complex baseband IQ signals and writes them as interleaved samples:

  - int16: I0, Q0, I1, Q1, ...  (USRP-style)
  - int8 : I0, Q0, I1, Q1, ...  (HackRF sc8)

Supported waveforms:
  - noise       : Band-limited complex noise
  - cw          : Single tone (CW)
  - multitone   : Multiple equal-amplitude tones
  - two_tone    : Classic two-tone IMD test
  - am          : AM (DSB with carrier)
  - fm          : FM
  - pm          : Phase modulation
  - ask         : Amplitude shift keying
  - ook         : On-off keying
  - fsk2        : 2-FSK
  - gfsk        : GFSK (approximate)
  - bpsk        : BPSK (with optional pulse shaping)
  - qpsk        : QPSK (with optional pulse shaping)
  - qam16       : 16-QAM (with optional pulse shaping)
  - qam64       : 64-QAM (with optional pulse shaping)
  - ofdm        : Simple OFDM with QPSK subcarriers
  - chirp       : Linear FM chirp
  - fhss        : Simple FHSS (frequency-hopping CW)

Pulse shaping for PSK/QAM/ASK:
  - rectangular (no shaping)
  - simple "sinc" FIR lowpass (acts like a mild RC-like shaper)

Also supports writing SigMF metadata sidecar files (.sigmf-meta).

Usage:
  python wavegen.py --help
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from typing import Optional, Sequence

import numpy as np


# ================================================================
# Helpers
# ================================================================

def iq_to_int16_interleaved(x: np.ndarray) -> np.ndarray:
    """
    Convert complex IQ (float in [-1, 1)) to interleaved int16 array.
    Layout: I0, Q0, I1, Q1, ...
    """
    x_real = np.clip(np.real(x), -1.0, 1.0 - 1e-12)
    x_imag = np.clip(np.imag(x), -1.0, 1.0 - 1e-12)
    scale = 32767.0
    i_int16 = (x_real * scale).astype(np.int16)
    q_int16 = (x_imag * scale).astype(np.int16)

    iq_int16 = np.empty(2 * x.shape[0], dtype=np.int16)
    iq_int16[0::2] = i_int16
    iq_int16[1::2] = q_int16
    return iq_int16


def iq_to_int8_interleaved(x: np.ndarray) -> np.ndarray:
    """
    Convert complex IQ (float in [-1, 1)) to interleaved int8 array
    for HackRF (sc8 format): I0, Q0, I1, Q1, ...
    """
    x_real = np.clip(np.real(x), -1.0, 1.0 - 1e-6)
    x_imag = np.clip(np.imag(x), -1.0, 1.0 - 1e-6)

    scale = 127.0  # [-127, +127] roughly
    i_int8 = (x_real * scale).astype(np.int8)
    q_int8 = (x_imag * scale).astype(np.int8)

    iq_int8 = np.empty(2 * x.shape[0], dtype=np.int8)
    iq_int8[0::2] = i_int8
    iq_int8[1::2] = q_int8
    return iq_int8


def normalize_to_level(x: np.ndarray, level: float) -> np.ndarray:
    """
    Normalize complex waveform to requested RMS level (0 < level <= 1).
    level is RMS of |x| relative to full-scale 1.0.
    """
    if level <= 0:
        warnings.warn("level <= 0; returning zeros.")
        return np.zeros_like(x)
    if level > 1.0:
        warnings.warn("level > 1.0; clamping to 1.0.")
        level = 1.0
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    if rms == 0:
        return x
    return x * (level / rms)


def prng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def apply_burst_envelope(x: np.ndarray,
                         fs: float,
                         period: float,
                         duty: float,
                         rise: float = 0.0,
                         fall: float = 0.0) -> np.ndarray:
    """
    Apply periodic on/off gating to a waveform.
    period: seconds per burst cycle
    duty  : 0..1 fraction of period "on"
    rise/fall: optional linear ramp times (seconds)
    """
    if period <= 0:
        return x
    if duty <= 0:
        return np.zeros_like(x)
    duty = min(duty, 1.0)
    samples_per_period = int(round(period * fs))
    if samples_per_period < 1:
        return x

    on_len = int(round(samples_per_period * duty))
    on_len = max(1, min(samples_per_period, on_len))
    env_period = np.zeros(samples_per_period, dtype=np.float32)
    env_period[:on_len] = 1.0

    rise_samp = int(round(rise * fs)) if rise > 0 else 0
    fall_samp = int(round(fall * fs)) if fall > 0 else 0
    rise_samp = min(rise_samp, on_len)
    fall_samp = min(fall_samp, on_len - rise_samp)

    if rise_samp > 0:
        env_period[:rise_samp] = np.linspace(0.0, 1.0, rise_samp, endpoint=False)
    if fall_samp > 0:
        env_period[on_len - fall_samp:on_len] = np.linspace(
            1.0, 0.0, fall_samp, endpoint=False)

    reps = int(np.ceil(x.size / samples_per_period))
    env = np.tile(env_period, reps)[:x.size]
    return x * env


def apply_packetization(x: np.ndarray,
                        fs: float,
                        total_duration: float,
                        packet_len: float,
                        guard_len: float) -> np.ndarray:
    """
    Repeat a packet waveform with guard time (zeros) to fill total duration.
    """
    if packet_len <= 0:
        return x
    n_total = int(round(fs * total_duration))
    if n_total <= 0:
        return x
    pkt_len = int(round(packet_len * fs))
    if pkt_len <= 0:
        return x[:n_total] if x.size >= n_total else np.pad(x, (0, n_total - x.size))
    guard_len = max(0.0, guard_len)
    guard_samp = int(round(guard_len * fs))

    pkt = x[:pkt_len]
    if pkt.size < pkt_len:
        pkt = np.pad(pkt, (0, pkt_len - pkt.size))

    period = pkt_len + guard_samp
    if period <= 0:
        return x
    out = np.zeros(n_total, dtype=np.complex64)
    idx = 0
    while idx < n_total:
        end = min(idx + pkt_len, n_total)
        out[idx:end] = pkt[:end - idx]
        idx += period
    return out


# ================================================================
# Time / pulse shaping helpers
# ================================================================

def gen_time_vector(fs: float, duration: float) -> np.ndarray:
    n = int(round(fs * duration))
    return np.arange(n) / fs


def bits_from_rng(num_bits: int, seed: Optional[int] = None) -> np.ndarray:
    rng = prng(seed)
    return rng.integers(0, 2, size=num_bits, dtype=np.int8)


def symbols_from_bits(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    # Pack bits into integers
    if len(bits) % bits_per_symbol != 0:
        bits = np.pad(bits, (0, bits_per_symbol -
                      (len(bits) % bits_per_symbol)))
    reshaped = bits.reshape(-1, bits_per_symbol)
    weights = 2 ** np.arange(bits_per_symbol)[::-1]
    return reshaped.dot(weights)


def design_sinc_pulse(sps: int, rolloff: float, span: int) -> np.ndarray:
    """
    Simple lowpass "sinc" pulse shaping filter (NOT a perfect RC, but good
    enough for lab pulse shaping and spectral smoothing).

    sps   : samples per symbol
    rolloff: extra bandwidth factor (0.0-1.0); higher => wider filter
    span : span in symbols (each side), so total taps ~ 2*span*sps+1.
    """
    if sps < 1:
        raise ValueError("sps must be >= 1.")
    if span < 1:
        span = 1
    N = 2 * span * sps + 1
    n = np.arange(N) - (N - 1) / 2

    # Normalized cutoff (fraction of Nyquist). 1/sps is Nyquist for the symbol stream.
    Wc = (1.0 + rolloff) / (2.0 * sps)
    Wc = min(Wc, 0.49)  # don't hit Nyquist exactly

    # Simple LP: sinc in discrete time, Hann-windowed
    h = np.sinc(2 * Wc * n)
    window = np.hamming(N)
    h *= window
    h /= np.sum(h)
    return h.astype(np.float64)


def symbols_to_waveform(symbols: np.ndarray,
                        fs: float,
                        sym_rate: float,
                        duration: float,
                        pulse: str = "rect",
                        rolloff: float = 0.25,
                        span: int = 6) -> np.ndarray:
    """
    Map complex symbols to a discrete-time baseband waveform
    at sample rate fs, duration 'duration', with given pulse shaping.

    pulse = "rect" -> simple repeat (ZOH)
          = "sinc" -> lowpass sinc FIR smoothing
    """
    sps = int(fs / sym_rate)
    if sps < 1:
        raise ValueError("fs must be >= sym_rate for these simple generators.")
    up = np.repeat(symbols, sps)

    if pulse == "sinc":
        taps = design_sinc_pulse(sps, rolloff, span)
        up = np.convolve(up, taps, mode="same")

    # Trim/pad to exact duration
    n_target = int(round(fs * duration))
    if up.size > n_target:
        up = up[:n_target]
    elif up.size < n_target:
        up = np.pad(up, (0, n_target - up.size))

    return up.astype(np.complex64)


# ================================================================
# Waveform generators (all produce complex baseband at fs)
# ================================================================

def gen_noise(fs: float, duration: float, bw: float, level: float,
              seed: Optional[int] = None) -> np.ndarray:
    """
    Band-limited complex noise centered at DC with bandwidth 'bw'.
    """
    if fs <= 0:
        warnings.warn("noise: fs must be > 0; using fs=1.0.")
        fs = 1.0
    if duration <= 0:
        warnings.warn("noise: duration must be > 0; using duration=1.0.")
        duration = 1.0
    if bw <= 0:
        warnings.warn("noise: bw must be > 0; using bw=fs/2.")
        bw = fs / 2.0
    if bw > fs:
        warnings.warn("noise: bw exceeds fs; clamping to fs.")
        bw = fs
    t = gen_time_vector(fs, duration)
    n = t.size
    rng = prng(seed)
    # Complex white
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    mask = np.abs(freqs) <= (bw / 2.0)
    X_filtered = X * mask
    x_filtered = np.fft.ifft(X_filtered).astype(np.complex64)
    x_filtered = normalize_to_level(x_filtered, level)
    return x_filtered


def gen_cw(fs: float, duration: float, tone_freq: float,
           level: float) -> np.ndarray:
    """
    Complex CW tone at frequency 'tone_freq' (Hz) relative to DC.
    """
    t = gen_time_vector(fs, duration)
    x = np.exp(1j * 2 * np.pi * tone_freq * t)
    x = normalize_to_level(x, level)
    return x.astype(np.complex64)


def gen_multitone(fs: float, duration: float, freqs: Sequence[float],
                  level: float) -> np.ndarray:
    """
    Sum multiple equal-amplitude tones.
    """
    t = gen_time_vector(fs, duration)
    x = np.zeros_like(t, dtype=np.complex64)
    for f in freqs:
        x += np.exp(1j * 2 * np.pi * f * t)
    if len(freqs) > 0:
        x /= len(freqs)
    x = normalize_to_level(x, level)
    return x.astype(np.complex64)


def gen_two_tone(fs: float, duration: float,
                 center_freq: float, spacing: float,
                 level: float) -> np.ndarray:
    """
    Classic two-tone IMD test centered at center_freq, with tones at
    center_freq +/- spacing/2.
    """
    f1 = center_freq - spacing / 2.0
    f2 = center_freq + spacing / 2.0
    return gen_multitone(fs, duration, [f1, f2], level)


def gen_am(fs: float, duration: float, carrier_freq: float,
           mod_freq: float, depth: float, level: float) -> np.ndarray:
    """
    Simple AM (DSB with carrier) on a single tone.

    s(t) = [1 + depth * cos(2*pi*f_m*t)] * exp(j*2*pi*f_c*t)
    """
    t = gen_time_vector(fs, duration)
    m = np.cos(2 * np.pi * mod_freq * t)
    s = (1.0 + depth * m) * np.exp(1j * 2 * np.pi * carrier_freq * t)
    s = normalize_to_level(s, level)
    return s.astype(np.complex64)


def gen_fm(fs: float, duration: float, carrier_freq: float,
           mod_freq: float, dev: float, level: float) -> np.ndarray:
    """
    Simple FM with single-tone modulator.

    s(t) = exp(j [2*pi*f_c*t + 2*pi*dev*integral(m(t) dt)])
    where m(t) = cos(2*pi*f_m*t).
    """
    t = gen_time_vector(fs, duration)
    m = np.cos(2 * np.pi * mod_freq * t)
    dt = 1.0 / fs
    integral = np.cumsum(m) * dt
    phase = 2 * np.pi * (carrier_freq * t + dev * integral)
    s = np.exp(1j * phase)
    s = normalize_to_level(s, level)
    return s.astype(np.complex64)


def gen_pm(fs: float, duration: float, carrier_freq: float,
           mod_freq: float, phase_dev: float, level: float) -> np.ndarray:
    """
    Simple phase modulation with single-tone modulator.

    s(t) = exp(j [2*pi*f_c*t + phase_dev*cos(2*pi*f_m*t)])
    """
    t = gen_time_vector(fs, duration)
    m = np.cos(2 * np.pi * mod_freq * t)
    phase = 2 * np.pi * carrier_freq * t + phase_dev * m
    s = np.exp(1j * phase)
    s = normalize_to_level(s, level)
    return s.astype(np.complex64)


# ----------------- Digital / symbol-based waveforms ----------------- #

def gen_ask(fs: float, duration: float, sym_rate: float,
            a0: float, a1: float, level: float,
            pulse: str = "rect", rolloff: float = 0.25, span: int = 6,
            seed: Optional[int] = None) -> np.ndarray:
    """
    Binary ASK: symbol 0 -> amplitude a0, symbol 1 -> amplitude a1.
    """
    num_symbols = int(np.ceil(duration * sym_rate))
    bits = bits_from_rng(num_symbols, seed)
    amps = np.where(bits == 0, a0, a1).astype(np.float32)
    syms = amps.astype(np.complex64)
    x = symbols_to_waveform(syms, fs, sym_rate, duration, pulse, rolloff, span)
    x = normalize_to_level(x, level)
    return x.astype(np.complex64)


def gen_ook(fs: float, duration: float, sym_rate: float,
            duty: float, level: float,
            seed: Optional[int] = None) -> np.ndarray:
    """
    Simple OOK: bit 1 -> carrier present, bit 0 -> no carrier.
    duty controls probability of '1'.
    Rectangular pulses; for OOK this is usually fine.
    """
    num_symbols = int(np.ceil(duration * sym_rate))
    rng = prng(seed)
    bits = (rng.random(num_symbols) < duty).astype(np.int8)
    amps = bits.astype(np.float32)
    syms = amps.astype(np.complex64)
    x = symbols_to_waveform(syms, fs, sym_rate, duration,
                            pulse="rect", rolloff=0.25, span=6)
    x = normalize_to_level(x, level)
    return x.astype(np.complex64)


def gen_bpsk(fs: float, duration: float, sym_rate: float,
             level: float, pulse: str = "rect",
             rolloff: float = 0.25, span: int = 6,
             seed: Optional[int] = None) -> np.ndarray:
    """
    BPSK baseband, with optional pulse shaping.
    """
    num_symbols = int(np.ceil(duration * sym_rate))
    bits = bits_from_rng(num_symbols, seed)
    syms = (2 * bits - 1).astype(np.float32)  # 0->-1, 1->+1
    syms_complex = syms.astype(np.complex64)
    x = symbols_to_waveform(syms_complex, fs, sym_rate, duration,
                            pulse, rolloff, span)
    x = normalize_to_level(x, level)
    return x.astype(np.complex64)


def gen_qpsk(fs: float, duration: float, sym_rate: float,
             level: float, pulse: str = "rect",
             rolloff: float = 0.25, span: int = 6,
             seed: Optional[int] = None) -> np.ndarray:
    """
    QPSK baseband with optional pulse shaping.
    """
    num_symbols = int(np.ceil(duration * sym_rate))
    bits = bits_from_rng(2 * num_symbols, seed)
    bits = bits.reshape(-1, 2)
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }
    syms = np.array([mapping[tuple(b)] for b in bits],
                    dtype=np.complex64) / np.sqrt(2)
    x = symbols_to_waveform(syms, fs, sym_rate, duration,
                            pulse, rolloff, span)
    x = normalize_to_level(x, level)
    return x.astype(np.complex64)


def gen_qam16(fs: float, duration: float, sym_rate: float,
              level: float, pulse: str = "rect",
              rolloff: float = 0.25, span: int = 6,
              seed: Optional[int] = None) -> np.ndarray:
    """
    16-QAM baseband with optional pulse shaping.
    """
    num_symbols = int(np.ceil(duration * sym_rate))
    bits = bits_from_rng(4 * num_symbols, seed)
    symbols = symbols_from_bits(bits, 4)  # 0..15
    # Simple square 16-QAM mapping
    i_map = np.array([-3, -1, +3, +1])
    q_map = np.array([-3, -1, +3, +1])
    I = i_map[(symbols >> 2) & 0b11]
    Q = q_map[symbols & 0b11]
    syms = (I + 1j * Q).astype(np.complex64)
    syms /= np.sqrt(np.mean(np.abs(syms) ** 2))  # normalize power
    x = symbols_to_waveform(syms, fs, sym_rate, duration,
                            pulse, rolloff, span)
    x = normalize_to_level(x, level)
    return x.astype(np.complex64)


def gen_qam64(fs: float, duration: float, sym_rate: float,
              level: float, pulse: str = "rect",
              rolloff: float = 0.25, span: int = 6,
              seed: Optional[int] = None) -> np.ndarray:
    """
    64-QAM baseband with optional pulse shaping.

    Uses a standard 8x8 square constellation with PAM levels [-7,-5,-3,-1,1,3,5,7].
    """
    num_symbols = int(np.ceil(duration * sym_rate))
    bits = bits_from_rng(6 * num_symbols, seed)
    symbols = symbols_from_bits(bits, 6)  # 0..63

    pam = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
    i_idx = (symbols >> 3) & 0x7  # upper 3 bits
    q_idx = symbols & 0x7         # lower 3 bits
    I = pam[i_idx]
    Q = pam[q_idx]
    syms = (I + 1j * Q).astype(np.complex64)
    syms /= np.sqrt(np.mean(np.abs(syms) ** 2))  # normalize power

    x = symbols_to_waveform(syms, fs, sym_rate, duration,
                            pulse, rolloff, span)
    x = normalize_to_level(x, level)
    return x.astype(np.complex64)


def gen_fsk2(fs: float, duration: float, sym_rate: float,
             f0: float, f1: float, level: float,
             seed: Optional[int] = None) -> np.ndarray:
    """
    2-FSK with frequencies f0 and f1 relative to DC.
    (rectangular pulses; FSK shaping is inherently frequency-domain).
    """
    if fs <= 0:
        warnings.warn("fsk2: fs must be > 0; using fs=1.0.")
        fs = 1.0
    if duration <= 0:
        warnings.warn("fsk2: duration must be > 0; using duration=1.0.")
        duration = 1.0
    if sym_rate <= 0:
        warnings.warn("fsk2: sym_rate must be > 0; using sym_rate=fs.")
        sym_rate = fs
    if sym_rate > fs:
        warnings.warn("fsk2: sym_rate > fs; clamping sym_rate=fs.")
        sym_rate = fs
    num_symbols = int(np.ceil(duration * sym_rate))
    bits = bits_from_rng(num_symbols, seed)
    sps = int(fs / sym_rate)
    t = np.arange(num_symbols * sps) / fs
    freqs = np.where(np.repeat(bits, sps) == 0, f0, f1)
    phase = 2 * np.pi * np.cumsum(freqs) / fs
    s = np.exp(1j * phase)
    s = normalize_to_level(s, level)
    # Trim/pad
    n_target = int(round(fs * duration))
    if s.size > n_target:
        s = s[:n_target]
    elif s.size < n_target:
        s = np.pad(s, (0, n_target - s.size))
    return s.astype(np.complex64)


def gen_gfsk(fs: float, duration: float, sym_rate: float,
             h: float, bt: float, level: float,
             seed: Optional[int] = None) -> np.ndarray:
    """
    Approximate GFSK-style: smoothed frequency modulation.

    For lab stress testing, this is often sufficient as an "FSK-like" signal.
    """
    if fs <= 0:
        warnings.warn("gfsk: fs must be > 0; using fs=1.0.")
        fs = 1.0
    if duration <= 0:
        warnings.warn("gfsk: duration must be > 0; using duration=1.0.")
        duration = 1.0
    if sym_rate <= 0:
        warnings.warn("gfsk: sym_rate must be > 0; using sym_rate=fs.")
        sym_rate = fs
    if sym_rate > fs:
        warnings.warn("gfsk: sym_rate > fs; clamping sym_rate=fs.")
        sym_rate = fs
    num_symbols = int(np.ceil(duration * sym_rate))
    bits = bits_from_rng(num_symbols, seed)
    dibits = 2 * bits - 1  # +/-1
    sps = int(fs / sym_rate)
    steps = np.repeat(dibits.astype(np.float32), sps)
    window = int(max(3, bt * sps))
    if window > 1:
        kernel = np.ones(window) / window
        smooth = np.convolve(steps, kernel, mode="same")
    else:
        smooth = steps
    dev = h * sym_rate / 2.0
    freq = dev * smooth
    phase = 2 * np.pi * np.cumsum(freq) / fs
    s = np.exp(1j * phase)
    s = normalize_to_level(s, level)
    # Trim/pad
    n_target = int(round(fs * duration))
    if s.size > n_target:
        s = s[:n_target]
    elif s.size < n_target:
        s = np.pad(s, (0, n_target - s.size))
    return s.astype(np.complex64)


def gen_ofdm(fs: float, duration: float, bw: float,
             fft_size: int, cp_len: int, level: float,
             seed: Optional[int] = None) -> np.ndarray:
    """
    Simple OFDM with QPSK on inner subcarriers.

    bw <= fs. Subcarriers occupy roughly bw.
    """
    if fs <= 0:
        warnings.warn("ofdm: fs must be > 0; using fs=1.0.")
        fs = 1.0
    if duration <= 0:
        warnings.warn("ofdm: duration must be > 0; using duration=1.0.")
        duration = 1.0
    if fft_size < 8:
        warnings.warn("ofdm: fft_size too small; using fft_size=8.")
        fft_size = 8
    if cp_len < 0:
        warnings.warn("ofdm: cp_len must be >= 0; using cp_len=0.")
        cp_len = 0
    if cp_len >= fft_size:
        warnings.warn("ofdm: cp_len >= fft_size; using cp_len=fft_size-1.")
        cp_len = fft_size - 1
    rng = prng(seed)
    sym_time = fft_size / fs
    cp_time = cp_len / fs
    ofdm_sym_dur = sym_time + cp_time
    num_syms = int(np.ceil(duration / ofdm_sym_dur))

    delta_f = fs / fft_size
    if bw <= 0:
        warnings.warn("ofdm: bw must be > 0; using bw=2*delta_f.")
        bw = 2 * delta_f
    if bw > fs:
        warnings.warn("ofdm: bw exceeds fs; clamping to fs.")
        bw = fs
    num_used = int(bw / delta_f)
    if num_used % 2 == 1:
        num_used -= 1
    max_used = fft_size - 2
    if num_used > max_used:
        warnings.warn("ofdm: bw too large for fft_size; clamping used bins.")
        num_used = max_used - (max_used % 2)
    if num_used < 2:
        warnings.warn("ofdm: bw too small; using 2 subcarriers.")
        num_used = 2

    half = num_used // 2
    used_bins = np.concatenate([
        np.arange(-half, 0),
        np.arange(1, half + 1)
    ])

    symbols = []
    for _ in range(num_syms):
        bits = rng.integers(0, 2, size=(used_bins.size * 2,))
        bits = bits.reshape(-1, 2)
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 1): -1 - 1j,
            (1, 0): 1 - 1j,
        }
        qpsk_syms = np.array([mapping[tuple(b)] for b in bits],
                             dtype=np.complex64) / np.sqrt(2)
        X = np.zeros(fft_size, dtype=np.complex64)
        for idx, bin_ in enumerate(used_bins):
            X[bin_ % fft_size] = qpsk_syms[idx]
        x = np.fft.ifft(X)
        x_cp = np.concatenate([x[-cp_len:], x])
        symbols.append(x_cp)

    waveform = np.concatenate(symbols)
    n_target = int(round(fs * duration))
    if waveform.size > n_target:
        waveform = waveform[:n_target]
    elif waveform.size < n_target:
        waveform = np.pad(waveform, (0, n_target - waveform.size))

    waveform = normalize_to_level(waveform, level)
    return waveform.astype(np.complex64)


def gen_chirp(fs: float, duration: float, f_start: float,
              f_end: float, level: float) -> np.ndarray:
    """
    Linear FM chirp from f_start to f_end (relative to DC).
    """
    t = gen_time_vector(fs, duration)
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t ** 2)
    s = np.exp(1j * phase)
    s = normalize_to_level(s, level)
    return s.astype(np.complex64)


def gen_fhss(fs: float, duration: float, hop_rate: float,
             freqs: Sequence[float], level: float,
             seed: Optional[int] = None) -> np.ndarray:
    """
    Simple FHSS: hops among given frequencies, each hop a CW segment.
    """
    rng = prng(seed)
    hop_dur = 1.0 / hop_rate
    num_hops = int(np.ceil(duration / hop_dur))
    samples_per_hop = int(round(fs * hop_dur))
    t_seg = np.arange(samples_per_hop) / fs
    segments = []
    for _ in range(num_hops):
        f = float(rng.choice(freqs))
        seg = np.exp(1j * 2 * np.pi * f * t_seg)
        segments.append(seg)
    waveform = np.concatenate(segments)
    n_target = int(round(fs * duration))
    if waveform.size > n_target:
        waveform = waveform[:n_target]
    elif waveform.size < n_target:
        waveform = np.pad(waveform, (0, n_target - waveform.size))
    waveform = normalize_to_level(waveform, level)
    return waveform.astype(np.complex64)


# ================================================================
# Combo helper (for Python use, not CLI)
# ================================================================

def combine_waveforms(waveforms: Sequence[np.ndarray],
                      levels: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Combine multiple complex waveforms (same length) into one.

    levels: optional per-waveform scaling factors. If None, each waveform
            is normalized to equal RMS contribution before summing.
    """
    if len(waveforms) == 0:
        raise ValueError("No waveforms to combine.")

    min_len = min(w.size for w in waveforms)
    waves = [w[:min_len] for w in waveforms]
    waves = [w.astype(np.complex64) for w in waves]

    if levels is None:
        levels = []
        for w in waves:
            rms = np.sqrt(np.mean(np.abs(w) ** 2))
            levels.append(1.0 if rms == 0 else 1.0 / rms)
    if len(levels) != len(waves):
        raise ValueError("levels length mismatch.")

    combo = np.zeros(min_len, dtype=np.complex64)
    for w, a in zip(waves, levels):
        combo += a * w

    return combo


# ================================================================
# SigMF metadata writer
# ================================================================

def write_sigmf_meta(
    data_path: str,
    fmt: str,
    fs: float,
    num_samples: int,
    waveform: str,
    params: dict,
    center_freq: Optional[float] = None,
    description: str = "",
    author: str = "wavegen"
):
    """Simple phase modulation with single-tone modulator.

    s(t) = exp(j [2*pi*f_c*t + phase_dev*cos(2*pi*f_m*t)])
    """
    if fmt == "int16":
        datatype = "ci16_le"
    elif fmt == "int8":
        datatype = "ci8_le"
    else:
        datatype = "unknown"

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    meta = {
        "global": {
            "core:datatype": datatype,
            "core:sample_rate": fs,
            "core:description": description,
            "core:version": "1.0.0",
            "core:author": author,
            "core:hw": params.get("hw", "")
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:datetime": now
            }
        ],
        "annotations": [
            {
                "core:sample_start": 0,
                "core:sample_count": num_samples,
                "core:comment": f"{waveform} generated by wavegen",
                "sdr:waveform": waveform,
                "sdr:params": params
            }
        ]
    }

    if center_freq is not None:
        meta["captures"][0]["core:frequency"] = center_freq

    meta_path = data_path + ".sigmf-meta"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Modular waveform generator - complex baseband IQ."
    )
    parser.add_argument("--waveform", "-w", type=str, required=True,
                        choices=[
                            "noise", "cw", "multitone", "two_tone",
                            "am", "fm", "pm",
                            "ask", "ook",
                            "fsk2", "gfsk",
                            "bpsk", "qpsk", "qam16", "qam64",
                            "ofdm", "chirp", "fhss"
                        ],
                        help="Waveform type.")
    parser.add_argument("--fs", type=float, required=True,
                        help="Sample rate (Hz).")
    parser.add_argument("--duration", "-d", type=float, required=True,
                        help="Signal duration (s).")
    parser.add_argument("--level", "-L", type=float, default=0.3,
                        help="RMS level (0<level<=1), relative to full-scale.")
    parser.add_argument("--outfile", "-o", type=str, required=True,
                        help="Output file (binary IQ).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed.")
    parser.add_argument("--fmt", type=str, default="int16",
                        choices=["int16", "int8"],
                        help="Output sample format: 'int16' (USRP-style) or 'int8' (HackRF sc8).")

    # Pulse shaping (for PSK/QAM/ASK)
    parser.add_argument("--pulse", type=str, default="rect",
                        choices=["rect", "sinc"],
                        help="Pulse shape for PSK/QAM/ASK ('rect' or 'sinc').")
    parser.add_argument("--rolloff", type=float, default=0.25,
                        help="Pulse shaping rolloff factor (0-1).")
    parser.add_argument("--span", type=int, default=6,
                        help="Pulse shaping span in symbols (each side).")

    # Common modulation parameters
    parser.add_argument("--tone-freq", type=float, default=0.0,
                        help="Tone frequency / carrier offset (Hz) for CW, AM, FM, PM, etc.")
    parser.add_argument("--center-freq", type=float, default=0.0,
                        help="Center frequency for two_tone (Hz).")
    parser.add_argument("--spacing", type=float, default=1e6,
                        help="Tone spacing for two_tone (Hz).")
    parser.add_argument("--freqs", type=str, default="",
                        help="Comma-separated list of tone frequencies for multitone or FHSS.")
    parser.add_argument("--sym-rate", type=float, default=1e6,
                        help="Symbol rate for digital mods (BPSK/QPSK/QAM/FSK/etc.)")
    parser.add_argument("--mod-freq", type=float, default=1e3,
                        help="Modulating frequency for AM/FM/PM (Hz).")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="AM modulation depth (0-1).")
    parser.add_argument("--dev", type=float, default=5e3,
                        help="FM frequency deviation (Hz).")
    parser.add_argument("--phase-dev", type=float, default=1.0,
                        help="Phase deviation (rad) for PM.")
    parser.add_argument("--bw", type=float, default=10e6,
                        help="Noise or OFDM bandwidth (Hz).")
    parser.add_argument("--fft-size", type=int, default=1024,
                        help="OFDM FFT size.")
    parser.add_argument("--cp-len", type=int, default=128,
                        help="OFDM cyclic prefix length (samples).")
    parser.add_argument("--a0", type=float, default=0.2,
                        help="ASK amplitude for symbol 0.")
    parser.add_argument("--a1", type=float, default=1.0,
                        help="ASK amplitude for symbol 1.")
    parser.add_argument("--duty", type=float, default=0.5,
                        help="OOK probability of '1' (0-1).")
    parser.add_argument("--f0", type=float, default=-50e3,
                        help="FSK frequency for symbol 0 (Hz).")
    parser.add_argument("--f1", type=float, default=+50e3,
                        help="FSK frequency for symbol 1 (Hz).")
    parser.add_argument("--h", type=float, default=0.5,
                        help="GFSK modulation index.")
    parser.add_argument("--bt", type=float, default=0.5,
                        help="GFSK BT product (approx).")
    parser.add_argument("--f-start", type=float, default=-10e6,
                        help="Chirp start frequency (Hz).")
    parser.add_argument("--f-end", type=float, default=+10e6,
                        help="Chirp end frequency (Hz).")
    parser.add_argument("--hop-rate", type=float, default=1e3,
                        help="FHSS hop rate (hops per second).")

    # SigMF options
    parser.add_argument("--sigmf", action="store_true",
                        help="Write SigMF sidecar metadata (.sigmf-meta).")
    parser.add_argument("--description", type=str, default="",
                        help="Description for SigMF metadata.")
    parser.add_argument("--author", type=str, default="wavegen",
                        help="Author field for SigMF metadata.")
    parser.add_argument("--center-freq-rf", type=float, default=None,
                        help="Intended RF center frequency (Hz) for SigMF metadata.")

    # Burst / packet options
    parser.add_argument("--burst-period", type=float, default=0.0,
                        help="Burst period (s). 0 disables burst gating.")
    parser.add_argument("--burst-duty", type=float, default=0.5,
                        help="Burst duty cycle (0-1).")
    parser.add_argument("--burst-rise", type=float, default=0.0,
                        help="Burst rise time (s) for linear ramp.")
    parser.add_argument("--burst-fall", type=float, default=0.0,
                        help="Burst fall time (s) for linear ramp.")
    parser.add_argument("--packet-len", type=float, default=0.0,
                        help="Packet length (s). 0 disables packetization.")
    parser.add_argument("--guard-time", type=float, default=0.0,
                        help="Guard time (s) between packets.")

    args = parser.parse_args()
    fs = args.fs
    dur = args.duration
    lvl = args.level
    seed = args.seed

    if fs <= 0:
        warnings.warn("fs must be > 0; using fs=1.0.")
        fs = 1.0
    if dur <= 0:
        warnings.warn("duration must be > 0; using duration=1.0.")
        dur = 1.0
    if lvl <= 0:
        warnings.warn("level must be > 0; using level=0.3.")
        lvl = 0.3
    if lvl > 1.0:
        warnings.warn("level > 1.0; clamping to 1.0.")
        lvl = 1.0
    if args.sym_rate <= 0:
        warnings.warn("sym_rate must be > 0; using sym_rate=fs.")
        args.sym_rate = fs
    if args.sym_rate > fs:
        warnings.warn("sym_rate > fs; clamping sym_rate=fs.")
        args.sym_rate = fs
    if args.bw <= 0:
        warnings.warn("bw must be > 0; using bw=fs/2.")
        args.bw = fs / 2.0
    if args.bw > fs:
        warnings.warn("bw exceeds fs; clamping to fs.")
        args.bw = fs
    if args.hop_rate <= 0:
        warnings.warn("hop_rate must be > 0; using hop_rate=1.0.")
        args.hop_rate = 1.0
    if args.fft_size < 8:
        warnings.warn("fft_size too small; using fft_size=8.")
        args.fft_size = 8
    if args.cp_len < 0:
        warnings.warn("cp_len must be >= 0; using cp_len=0.")
        args.cp_len = 0
    if args.cp_len >= args.fft_size:
        warnings.warn("cp_len >= fft_size; using cp_len=fft_size-1.")
        args.cp_len = args.fft_size - 1

    if args.freqs.strip():
        freqs = [float(f.strip()) for f in args.freqs.split(",") if f.strip()]
    else:
        freqs = []

    print(f"Generating waveform '{args.waveform}'...")
    print(f"  fs       : {fs:.6g} Hz")
    print(f"  duration : {dur:.6g} s")
    print(f"  level    : {lvl:.3f}")
    print(f"  outfile  : {args.outfile}")
    print(f"  fmt      : {args.fmt}")
    if args.waveform in ("bpsk", "qpsk", "qam16", "qam64", "ask"):
        print(f"  pulse    : {args.pulse}")

    # Generate baseband complex waveform
    if args.waveform == "noise":
        x = gen_noise(fs, dur, args.bw, lvl, seed)
    elif args.waveform == "cw":
        x = gen_cw(fs, dur, args.tone_freq, lvl)
    elif args.waveform == "multitone":
        if not freqs:
            warnings.warn("multitone requires --freqs; using 0 Hz.")
            freqs = [0.0]
        x = gen_multitone(fs, dur, freqs, lvl)
    elif args.waveform == "two_tone":
        x = gen_two_tone(fs, dur, args.center_freq, args.spacing, lvl)
    elif args.waveform == "am":
        x = gen_am(fs, dur, args.tone_freq, args.mod_freq, args.depth, lvl)
    elif args.waveform == "fm":
        x = gen_fm(fs, dur, args.tone_freq, args.mod_freq, args.dev, lvl)
    elif args.waveform == "pm":
        x = gen_pm(fs, dur, args.tone_freq, args.mod_freq,
                   args.phase_dev, lvl)
    elif args.waveform == "ask":
        x = gen_ask(fs, dur, args.sym_rate,
                    args.a0, args.a1, lvl,
                    args.pulse, args.rolloff, args.span, seed)
    elif args.waveform == "ook":
        x = gen_ook(fs, dur, args.sym_rate, args.duty, lvl, seed)
    elif args.waveform == "fsk2":
        x = gen_fsk2(fs, dur, args.sym_rate, args.f0, args.f1, lvl, seed)
    elif args.waveform == "gfsk":
        x = gen_gfsk(fs, dur, args.sym_rate, args.h, args.bt, lvl, seed)
    elif args.waveform == "bpsk":
        x = gen_bpsk(fs, dur, args.sym_rate, lvl,
                     args.pulse, args.rolloff, args.span, seed)
    elif args.waveform == "qpsk":
        x = gen_qpsk(fs, dur, args.sym_rate, lvl,
                     args.pulse, args.rolloff, args.span, seed)
    elif args.waveform == "qam16":
        x = gen_qam16(fs, dur, args.sym_rate, lvl,
                      args.pulse, args.rolloff, args.span, seed)
    elif args.waveform == "qam64":
        x = gen_qam64(fs, dur, args.sym_rate, lvl,
                      args.pulse, args.rolloff, args.span, seed)
    elif args.waveform == "ofdm":
        x = gen_ofdm(fs, dur, args.bw, args.fft_size, args.cp_len, lvl, seed)
    elif args.waveform == "chirp":
        x = gen_chirp(fs, dur, args.f_start, args.f_end, lvl)
    elif args.waveform == "fhss":
        if not freqs:
            warnings.warn("fhss requires --freqs; using 0 Hz.")
            freqs = [0.0]
        x = gen_fhss(fs, dur, args.hop_rate, freqs, lvl, seed)
    else:
        raise ValueError(f"Unknown waveform: {args.waveform}")

    # Optional packetization / burst gating
    if args.packet_len > 0:
        x = apply_packetization(x, fs, dur, args.packet_len, args.guard_time)
    elif args.burst_period > 0:
        x = apply_burst_envelope(
            x, fs, args.burst_period, args.burst_duty,
            args.burst_rise, args.burst_fall)

    # Convert to chosen integer format
    if args.fmt == "int16":
        iq = iq_to_int16_interleaved(x)
        fmt_desc = "interleaved int16 IQ (I0, Q0, I1, Q1, ...)"
    elif args.fmt == "int8":
        iq = iq_to_int8_interleaved(x)
        fmt_desc = "interleaved int8 IQ (HackRF sc8: I0, Q0, I1, Q1, ...)"
    else:
        raise ValueError(f"Unknown format: {args.fmt}")

    iq.tofile(args.outfile)
    size_bytes = os.path.getsize(args.outfile)
    print(f"Done. Wrote {size_bytes} bytes to '{args.outfile}'.")
    print(f"Format: {fmt_desc}")

    # Optional SigMF metadata
    if args.sigmf:
        num_complex_samples = x.size
        params = {
            "fs": fs,
            "waveform": args.waveform,
            "sym_rate": args.sym_rate,
            "bw": args.bw,
            "pulse": args.pulse,
            "rolloff": args.rolloff,
            "span": args.span,
            "fmt": args.fmt,
            "burst_period": args.burst_period,
            "burst_duty": args.burst_duty,
            "burst_rise": args.burst_rise,
            "burst_fall": args.burst_fall,
            "packet_len": args.packet_len,
            "guard_time": args.guard_time,
            "hw": "USRP X310" if args.fmt == "int16" else "HackRF One"
        }

        if args.waveform in ("bpsk", "qpsk", "qam16", "qam64"):
            params["modulation"] = args.waveform.upper()
        elif args.waveform == "noise":
            params["modulation"] = "NOISE"
        elif args.waveform in ("am", "fm", "pm"):
            params["modulation"] = args.waveform.upper()
        else:
            params["modulation"] = args.waveform

        write_sigmf_meta(
            data_path=args.outfile,
            fmt=args.fmt,
            fs=fs,
            num_samples=num_complex_samples,
            waveform=args.waveform,
            params=params,
            center_freq=args.center_freq_rf,
            description=args.description,
            author=args.author
        )
        print(f"Wrote SigMF metadata to '{args.outfile}.sigmf-meta'.")


if __name__ == "__main__":
    main()
