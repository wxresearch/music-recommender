"""Streamlit dashboard for local acoustic music discovery."""
import hashlib
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analyzer import analyze_audio
from library import add_song, delete_song, get_feedback, set_feedback, songs
from recommender import recommendations
from visualizations import overview

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")
st.title("🎵 Music Recommender")
st.caption("Local song analysis, persistent library, and personalized recommendations.")
st.info("Use standard, unencrypted audio files you are authorized to analyze. "
        "QQ Music encrypted .mgg0/.mflac variants are not supported.")

@st.cache_data(show_spinner=False)
def analyze_cached(content: bytes, suffix: str):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        path = Path(tmp.name)
    try:
        return analyze_audio(path)
    finally:
        path.unlink(missing_ok=True)

@st.cache_data(show_spinner=False, max_entries=30)
def visualization_cached(path_text: str, mtime_ns: int):
    return overview(Path(path_text))

uploaded = st.file_uploader("Add recordings to your local library",
                            type=["wav", "mp3", "flac", "ogg"],
                            accept_multiple_files=True)
if uploaded:
    successes, duplicates, failures = 0, 0, []
    for item in uploaded:
        raw = item.getvalue()
        identifier = hashlib.sha256(raw).hexdigest()
        extension = Path(item.name).suffix.lower()
        try:
            with st.spinner(f"Analyzing {item.name}"):
                features = analyze_cached(raw, extension)
                if add_song(identifier, item.name, extension, raw, features):
                    successes += 1
                else:
                    duplicates += 1
        except Exception as exc:
            failures.append((item.name, str(exc)))
    st.success(f"Added {successes} new song(s). {duplicates} already in library.")
    for name, error in failures:
        st.error(f"{name}: {error}. Verify this is a standard, unencrypted audio file.")

library = songs()
if not library:
    st.info("No songs yet. Upload at least two recordings or run: python demo_audio.py")
    st.stop()

st.sidebar.header("Your library")
st.sidebar.write(f"{len(library)} recordings saved on this computer")
selection = st.selectbox("Choose a song to analyze", library,
                         format_func=lambda s: s["title"])
sid = selection["id"]
features = selection["features"]
st.audio(str(selection["path"]))

tab1, tab2, tab3 = st.tabs(["Analysis", "Recommendations", "Library & preferences"])
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated tempo", f"{features['tempo']:.0f} BPM")
    c2.metric("RMS energy", f"{features['rms']:.4f}")
    c3.metric("Spectral brightness", f"{features['brightness']:.0f} Hz")
    c4.metric("Duration analyzed", f"{features['duration']:.1f} seconds")
    with st.spinner("Generating visualizations"):
        waveform, spectrum, freq, times = visualization_cached(
            str(selection["path"]), selection["path"].stat().st_mtime_ns)
    st.subheader("Waveform (first 45 seconds)")
    st.line_chart(waveform, x="Seconds", y="Amplitude")
    st.subheader("Spectrogram (first 45 seconds)")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(spectrum, origin="lower", aspect="auto",
              extent=[float(times[0]), float(times[-1]),
                      float(freq[0]), float(freq[-1])],
              cmap="magma", vmin=-80, vmax=0)
    ax.set(xlabel="Time (seconds)", ylabel="Frequency (Hz)")
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Raw audio features"):
        st.dataframe(pd.DataFrame({"feature": list(features),
                                   "value": list(features.values())}),
                     hide_index=True)

with tab2:
    if len(library) < 2:
        st.warning("Add one more song to get recommendations.")
    else:
        feedback = get_feedback()
        weight = st.slider("Personalization strength", 0.0, 1.0, 0.35, 0.05,
                           help="0 uses only acoustic similarity. Larger values "
                                "favor liked songs and discourage disliked ones.")
        records = [{"id": s["id"], "title": s["title"], **s["features"]}
                   for s in library]
        results = recommendations(pd.DataFrame(records), sid, limit=10,
                                  feedback=feedback, preference_weight=weight)
        st.caption("Smaller acoustic distance means a closer match. "
                   "Rank score also incorporates your likes/dislikes; "
                   "these values are not match percentages.")
        for _, r in results.iterrows():
            item = next(s for s in library if s["id"] == r["id"])
            st.markdown(f"**{r['title']}** · {r['tempo']:.0f} BPM · "
                        f"acoustic distance {r['acoustic_distance']:.2f} · "
                        f"rank score {r['rank_score']:.2f}")
            st.audio(str(item["path"]))
            current = feedback.get(r["id"], 0)
            label = st.radio("Your preference", ["Neutral", "Like", "Dislike"],
                             horizontal=True, key=f"rating_{r['id']}",
                             index={0: 0, 1: 1, -1: 2}.get(current, 0))
            value = {"Neutral": 0, "Like": 1, "Dislike": -1}[label]
            if current != value:
                set_feedback(r["id"], value)
                feedback[r["id"]] = value
        st.caption("Ratings are saved locally. Change the selected song or slider "
                   "to refresh rankings after rating songs.")

with tab3:
    st.subheader("Saved music library")
    feedback = get_feedback()
    st.dataframe(pd.DataFrame([{
        "Song": s["title"], "Tempo (BPM)": round(s["features"]["tempo"]),
        "Duration analyzed (s)": round(s["features"]["duration"], 1),
        "Preference": {1: "Liked", -1: "Disliked", 0: "Neutral"}.get(
            feedback.get(s["id"], 0))
    } for s in library]), hide_index=True)
    st.subheader("Rate any song")
    rating_song = st.selectbox(
        "Choose a song to like or dislike",
        library,
        format_func=lambda s: s["title"],
        key="rating_song_selector",
    )
    rating_id = rating_song["id"]
    current_rating = feedback.get(rating_id, 0)
    rating_options = ["Neutral", "Like", "Dislike"]
    with st.form(key=f"rating_form_{rating_id}"):
        selected_preference = st.radio(
            "Your rating for this song",
            rating_options,
            horizontal=True,
            index={0: 0, 1: 1, -1: 2}[current_rating],
        )
        save_rating = st.form_submit_button("Save rating")
    if save_rating:
        set_feedback(rating_id, {"Neutral": 0, "Like": 1, "Dislike": -1}[selected_preference])
        st.rerun()
    st.warning("Removing a song deletes the locally stored audio copy and its rating.")
    if st.button("Delete selected song"):
        delete_song(sid)
        st.rerun()
