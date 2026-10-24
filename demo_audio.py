"""Generate four synthetic demo WAV files (no copyrighted songs)."""
from pathlib import Path
import numpy as np
import soundfile as sf

out = Path("demo_tracks")
out.mkdir(exist_ok=True)
sr = 22050
rng = np.random.default_rng(42)
tracks = [
    ("Pulse_A", 120, 220, 0.22),
    ("Pulse_B", 124, 230, 0.25),
    ("Slow_C", 75, 330, 0.6),
    ("Pulse_D", 118, 210, 0.20),
]
for name, bpm, freq, brightness in tracks:
    duration = 18
    t = np.arange(sr * duration) / sr
    tone = 0.13 * np.sin(2 * np.pi * freq * t) + brightness * 0.07 * np.sin(2 * np.pi * freq * 3 * t)
    pulse_phase = (t * bpm / 60) % 1.0
    kick = 0.15 * np.sin(2 * np.pi * 75 * t) * np.exp(-pulse_phase * 22)
    click = 0.04 * rng.standard_normal(len(t)) * np.exp(-pulse_phase * 45)
    audio = np.clip(tone + kick + click, -1, 1).astype("float32")
    sf.write(out / f"{name}.wav", audio, sr)
print(f"Created four WAV files in {out.resolve()}")
