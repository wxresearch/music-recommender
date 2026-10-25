# Music Recommender

A Python app that analyzes audio and recommends music using acoustic features, listening preferences, and optional AI embeddings.

## Features

- Analyze tempo, rhythm, timbre, waveforms, and spectrograms.
- Find acoustically similar songs or get recommendations based on likes and dislikes.
- Save your music library and ratings locally.
- Try hybrid recommendations combining three CLAP audio segments with acoustic features.
- Rate how similar two songs sound (1–5), separately from Like/Dislike.
- Discover similar or mood-based songs, save discoveries, and export playlists.
- Explore your taste dashboard and interactive music universe.\n- Create custom playlists, track listening sessions, and screen for duplicate recordings.\n- Search your local songs by natural-language description, try Surprise Me, and inspect recommendation explanations.

## Getting started

Run in **Windows Command Prompt** (Python required):

```cmd
git clone https://github.com/wxresearch/music-recommender.git
cd music-recommender
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501 and upload WAV, MP3, FLAC, or OGG files.

To try the optional AI recommender:

```cmd
python -m pip install -r requirements-ai.txt
```

Restart the app, adjust **Acoustic feature contribution**, then choose **Generate AI recommendations**. The first run downloads the model and recalculates cached embeddings. Pairwise similarity ratings influence subsequent AI rankings.

## Updating

From your project folder:

```cmd
git pull
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Audio files and preferences are stored locally in `library_data/` and excluded from Git. Protected audio formats are not supported.


## Online music discovery

Get a free Last.fm API key at https://www.last.fm/api/account/create. In Windows CMD, set it before starting Streamlit:

```cmd
set "LASTFM_API_KEY=your-key"
python -m streamlit run app.py
```
