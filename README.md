# Music Recommender

A Python app that analyzes audio and recommends music using acoustic features, listening preferences, and optional AI embeddings.

## Features

- Analyze tempo, rhythm, timbre, waveforms, and spectrograms.
- Find acoustically similar songs or get recommendations based on likes and dislikes.
- Save your music library and ratings locally.
- Try AI-powered recommendations using pretrained CLAP audio embeddings.

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

Restart the app and choose **Generate AI recommendations**. The pretrained model downloads on first use.

## Updating

From your project folder:

```cmd
git pull
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Audio files and preferences are stored locally in `library_data/` and excluded from Git. Protected audio formats are not supported.
