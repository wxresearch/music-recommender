# Music Recommender

A Python app that analyzes audio and recommends music using acoustic features, listening preferences, and optional AI embeddings.

## Features

- Analyze tempo, rhythm, timbre, waveforms, and spectrograms.
- Find acoustically similar songs or get recommendations based on likes and dislikes.
- Save your music library and ratings locally.
- Try hybrid recommendations combining three CLAP audio segments with acoustic features.
- Rate how similar two songs sound (1–5), separately from Like/Dislike.
- Compare CLAP-only, acoustic-only, and hybrid recommendations in the Evaluation tab.
- Discover similar or mood-based songs, save discoveries, check Spotify availability, and export playlists.
- Explore your taste dashboard, interactive song map and best-matching song passages.

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

Rate at least three song pairs, then select **Evaluation → Run evaluation** to compare how each model's similarity ordering agrees with your ratings.

## Online music discovery

Get a free Last.fm API key at https://www.last.fm/api/account/create. In Windows CMD, set it before starting Streamlit:

```cmd
set "LASTFM_API_KEY=your-key"
python -m streamlit run app.py
```

Open **Online Discovery**, check the artist and track title, and click **Find new songs**. Spotify links open searches rather than guaranteed exact tracks. Last.fm uses listening data; external tracks are not analyzed by CLAP.

For Spotify playlist export, create a Spotify developer application and register the redirect URI shown in the app. Then provide its client credentials in **Online Discovery → Export to Spotify playlist**; Spotify authorization occurs in your browser. Spotify matches tracks by artist/title, so inspect the created playlist for mismatches.

Online Discovery offers Spotify search, YouTube Music search, and Last.fm track pages. To verify exact Spotify catalog matches and filter the list, supply your Spotify developer Client ID and Secret and click **Verify tracks on Spotify**. Unverified is not proof a song is unavailable everywhere.
