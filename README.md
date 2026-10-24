# Music Recommender

A local, content-based music analyzer and recommender. Upload audio recordings, extract features with **librosa**, and discover songs with similar acoustic characteristics using **scikit-learn**. A **Streamlit** interface provides analysis and playback.

## Features

- Analyze estimated tempo (BPM), RMS energy, spectral centroid, bandwidth, rolloff, zero-crossing rate, onset strength, MFCCs and chroma.
- Recommend acoustically similar tracks using standardized, weighted feature distances.
- Inspect song features, listen to uploaded recordings, and compare recommendations.
- Generate four synthetic WAV examples for testing without copyrighted music.

## Quick start (Windows PowerShell)

Python 3.11 or 3.12 recommended.

```powershell
git clone https://github.com/wxresearch/music-recommender.git
cd music-recommender
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python demo_audio.py
streamlit run app.py
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py`.

Upload the four generated WAV files from `demo_tracks/` or your own authorized audio recordings. Select any track to view its features and nearest neighbors.

## How it works

1. Decode the first 180 seconds to mono 22.05 kHz audio.
2. Extract rhythm, spectral, timbral and pitch-class summary features.
3. Standardize features across the current session's song library.
4. Reduce the impact of many correlated MFCC/chroma dimensions using group weights.
5. Rank tracks by Euclidean distance (smaller means acoustically closer, **not** a prediction of preference or a probability).

## Limitations

- Tempo may be estimated at half or double the perceived beat.
- Feature similarity is not genre, emotion, or learned listener preference.
- The tiny synthetic test library is illustrative; real-world recommendations need a larger and diverse dataset.
- Data is kept in Streamlit session memory and can be lost after a restart.
- Feature scaling changes when the library changes.
- WAV is the most reliable starting format; some compressed formats require decoding support.
- Do not publish or redistribute recordings without appropriate rights.

## Roadmap

- Visualization of waveforms, beats and spectrograms
- Persistent library and feature caching
- Listener feedback (likes and dislikes)
- Learned embeddings and improved ranking
- Public demo deployment

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `analyzer.py` | Audio feature extraction |
| `recommender.py` | Similarity ranking |
| `demo_audio.py` | Generate synthetic test WAVs |
| `requirements.txt` | Python dependencies |

