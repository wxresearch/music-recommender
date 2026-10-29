# Music Recommender — AI-Powered Music Discovery and Personalized Playlists

Music Recommender is a Python and Streamlit music recommendation app that analyzes audio files and personal listening preferences to help users discover songs, understand musical similarities, and organize their music libraries. It combines **audio signal processing**, **content-based recommendations**, and optional **AI audio embeddings** to suggest music based on how songs sound and what listeners enjoy.

Rather than relying only on artist names or broad genres, the app examines acoustic features such as tempo, rhythm, and timbre. Feedback from likes, dislikes, and song-to-song similarity ratings helps refine recommendations over time.

## Features

- **Audio analysis:** Explore tempo, rhythm, timbre, waveforms, and spectrograms.
- **Personalized music recommendations:** Find acoustically similar tracks and recommendations based on likes and dislikes.
- **Optional AI recommendations:** Combine CLAP audio embeddings from multiple song segments with acoustic features.
- **Music taste learning:** Use listening feedback to improve recommendations.
- **Genre and mood tagging:** Estimate a genre and up to three mood tags for each song.
- **Playlist creation:** Build custom playlists or generate playlists with AI-assisted tools.
- **Natural-language music search:** Find tracks in your local collection using descriptive phrases.
- **Explainable recommendations:** Inspect why a song was recommended.
- **Music discovery tools:** Explore similar tracks, discover mood-based songs, and use Surprise Me mode.
- **Library management:** Save ratings, preferences, playlists, tags, and tracks locally; export and restore backups.

## Getting Started

Requires Python on Windows. In Command Prompt:

```bat
git clone https://github.com/wxresearch/music-recommender.git
cd music-recommender
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` and add WAV, MP3, FLAC, or OGG audio files.

To enable the optional AI recommender, install the additional dependencies:

```bat
python -m pip install -r requirements-ai.txt
```

Restart the app, configure the acoustic-feature contribution, and select **Generate AI recommendations**. The first run may download a model and compute local embeddings.

## Optional Online Discovery

For Last.fm-powered online music discovery, obtain an API key at https://www.last.fm/api/account/create and set it in Command Prompt before launching the app:

```bat
set "LASTFM_API_KEY=your-key"
python -m streamlit run app.py
```

## Local Data and Privacy

Music files, listening preferences, playlists, and AI tags are stored locally in `library_data/`, which is excluded from Git. Backup files can include audio files but do not include API keys. Restoring a backup merges saved records with the existing library.

## Technologies and Topics

**Python · Streamlit · Music recommendation system · Machine learning · Audio signal processing · Content-based filtering · CLAP audio embeddings · Personalized playlists · Music discovery · Genre classification · Mood detection · Natural-language search**

## Project Repository

https://github.com/wxresearch/music-recommender
