"""Waveform, spectrogram, and comparison visualizations."""
import librosa
import numpy as np
import pandas as pd


def overview(path, seconds=45):
    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=seconds)
    hop = 512
    # Cap plotted points for responsiveness.
    step = max(1, len(y) // 1600)
    waveform = pd.DataFrame({"Seconds": np.arange(0, len(y), step) / sr,
                             "Amplitude": y[::step]})
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop)), ref=np.max)
    # Compress to 72 frequency rows and 250 time columns.
    fi = np.unique(np.linspace(0, spec.shape[0] - 1, 72, dtype=int))
    ti = np.unique(np.linspace(0, spec.shape[1] - 1, 250, dtype=int))
    heatmap = spec[np.ix_(fi, ti)]
    freqs = librosa.fft_frequencies(sr=sr)[fi]
    times = librosa.frames_to_time(ti, sr=sr, hop_length=hop)
    return waveform, heatmap, freqs, times
