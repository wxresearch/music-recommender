import numpy as np
import librosa


def analyze_audio(path):
    """Analyze up to the first three minutes of a local audio file."""
    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=180)
    if len(y) < sr:
        raise ValueError("Audio needs to be at least one second long.")
    if np.max(np.abs(y)) < 1e-6:
        raise ValueError("The audio is silent.")

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    onset = librosa.onset.onset_strength(y=y, sr=sr)

    beat_times = librosa.frames_to_time(beats, sr=sr)
    beat_intervals = np.diff(beat_times)
    # Beat regularity is the coefficient of variation of beat intervals.
    beat_regularity = (float(1 / (1 + np.std(beat_intervals) /
                                   max(np.mean(beat_intervals), 1e-8)))
                       if len(beat_intervals) >= 2 else 0.0)
    result = {
        "tempo": tempo,
        "beat_count": int(len(beats)),
        "beat_regularity": beat_regularity,
        "rms": float(np.mean(rms)),
        "brightness": float(np.mean(centroid)),
        "bandwidth": float(np.mean(bandwidth)),
        "rolloff": float(np.mean(rolloff)),
        "zcr": float(np.mean(zcr)),
        "onset_strength": float(np.mean(onset)),
        "duration": float(len(y) / sr),
    }
    result.update({f"mfcc_{i+1}": float(np.mean(v)) for i, v in enumerate(mfcc)})
    result.update({f"chroma_{i}": float(np.mean(v)) for i, v in enumerate(chroma)})
    return result
