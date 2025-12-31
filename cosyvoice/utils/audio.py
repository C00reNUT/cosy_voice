"""
Audio utilities for CosyVoice training.

This module provides mel spectrogram extraction functions used during training.
Originally from Matcha-TTS (third_party/Matcha-TTS/matcha/utils/audio.py).

Micromamba env: fish-speech
"""
import torch
from librosa.filters import mel as librosa_mel_fn


mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """Apply dynamic range compression to tensor."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    """Normalize spectral magnitudes."""
    return dynamic_range_compression_torch(magnitudes)


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    Compute mel spectrogram from audio waveform.

    Args:
        y: Audio waveform tensor [batch, samples]
        n_fft: FFT size
        num_mels: Number of mel frequency bins
        sampling_rate: Audio sampling rate
        hop_size: Hop size for STFT
        win_size: Window size for STFT
        fmin: Minimum frequency for mel filterbank
        fmax: Maximum frequency for mel filterbank
        center: Whether to center the STFT frames

    Returns:
        Mel spectrogram tensor
    """
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement

    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec
