# Music Recommender

A local audio-analysis and personalized music recommendation app using Python, librosa, scikit-learn, SQLite and Streamlit.

## Features

- **Interactive analysis:** estimated tempo, RMS energy, frequency features, waveform, spectrogram, and full feature table.
- **Improved ranking:** group-weighted standardized acoustic features (MFCC, chroma, beat and spectral attributes). Distances are not probabilities.
- **Persistent library:** uploaded audio files and their features are stored locally in `library_data/`. Files are deduplicated by SHA-256 and don't require re-uploading after restart.
- **Two recommendation modes:** Find similar songs by acoustic distance, or Recommended for you using your liked/disliked examples. Personal discovery ranks unrated tracks first and never pretends already-liked songs are new discoveries.
- **Personalization:** like, dislike, or clear ratings; adjust how strongly feedback influences recommendations.
- **Manage library:** view your saved collection and delete individual songs.

## Windows Command Prompt quick start

Use Python 3.11–3.13 with compatible packages.

```cmd
git clone https://github.com/wxresearch/music-recommender.git
cd music-recommender
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python demo_audio.py
python -m streamlit run app.py
```

If you've **already cloned** this repository, instead run:

```cmd
cd music-recommender
git pull
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Your app will usually be at http://localhost:8501. Upload the four files generated in `demo_tracks/`, or upload other standard WAV/MP3/FLAC/OGG recordings you are authorized to analyze.

## How recommendations work

The analyzer reads up to 180 seconds per song and computes rhythm, spectral, pitch-class and timbre features. The ranking engine scales each feature across your current library and weights related groups to avoid overemphasizing the 13 MFCC coefficients. Euclidean distance measures acoustic similarity. **Find similar songs** ranks other tracks by ascending acoustic distance. **Recommended for you** combines acoustic closeness to your currently selected track with a profile score derived from distances to liked and disliked examples; it only returns unrated tracks when you have feedback. If your current three-song library has two liked tracks and one selected neutral track, there are no new tracks left to discover. Add more recordings to see personalized results. This is a transparent heuristic, **not** a trained AI predictor of listening preference.

## Persistence and privacy

- `library_data/library.sqlite3` holds song metadata, audio features and preferences.
- `library_data/audio/` holds the uploaded copies of your tracks.
- `library_data/` is in `.gitignore`; do **not** publish copyrighted recordings or personal data.
- Delete a song from the Library tab to remove its local audio and rating.
- This is a **single-user local application**; the SQLite library is not designed for multiuser hosted deployment.

## Limitations

- Proprietary or encrypted QQ Music formats (including `.mgg0.flac`) cannot be decoded.
- Tempo estimation may be half/double the perceived BPM.
- Audio features do not reliably infer lyrics, genres, emotions, or your tastes.
- Personalization is heuristic and improves only when the library has diverse rated tracks.
- New uploads can take time to analyze; the same recording is skipped on subsequent uploads.
- Only the first three minutes are analyzed for features; visualizations use the first 45 seconds.
- Audio playback is local, and Streamlit can require format-compatible browser playback.

## Structure

- `app.py`: user interface and workflow
- `analyzer.py`: librosa audio features
- `visualizations.py`: waveform and spectrogram
- `recommender.py`: weighted and personalized ranking
- `library.py`: SQLite data persistence
- `demo_audio.py`: generated synthetic recordings
- `requirements.txt`: dependencies

## Next research directions

Evaluate recommendations on a labeled song set, consider pretrained embeddings, add listening-history signals, and benchmark ranking quality using human judgments.

## Run recommendation tests

```cmd
python -m unittest discover -s tests
```

## Experimental pretrained audio embeddings

The default recommender continues to use librosa features and works without PyTorch.
An optional **AI audio embeddings** section is available in the Recommendations tab.
It uses the pretrained `laion/clap-htsat-unfused` CLAP model to embed the first
10 seconds of each local audio file at 48 kHz, caches vectors under
`library_data/embeddings/`, and ranks with cosine similarity. In personalized
mode, likes and dislikes affect the AI score and previously rated tracks are
excluded from discovery.

Install the optional dependencies from Windows **Command Prompt**:

```cmd
python -m pip install -r requirements-ai.txt
python -m streamlit run app.py
```

Then press **Generate AI recommendations** in the Recommendations tab.
The first run requires an internet connection to download model weights and
may consume significant disk space and memory; CPU inference can be slow.
Model availability and compatibility depend on installed PyTorch/Transformers
versions. The AI path is experimental and has not been validated on every
Windows/Python combination.

The 10-second embedding window is deliberately short for a first prototype;
future iterations should compare several segments and evaluate ranking against
human similarity judgments. A cosine value is not a match percentage.
