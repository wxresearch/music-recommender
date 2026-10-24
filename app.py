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
    if "beat_regularity" in features:
        st.caption(f"Detected beats: {features['beat_count']} · Beat regularity: {features['beat_regularity']:.2f} (0–1)")
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
        mode_label = st.radio(
            "Recommendation mode",
            ["Find similar songs", "Recommended for you"],
            horizontal=True,
            help="Similarity compares songs with your selected track; "
                 "For you uses your likes and dislikes to find unrated tracks.",
        )
        mode = "similar" if mode_label == "Find similar songs" else "personalized"
        weight = 0.35
        if mode == "personalized":
            weight = st.slider(
                "Personalization strength", 0.0, 1.0, 0.70, 0.05,
                help="Higher values prioritize musical features shared with "
                     "liked songs rather than similarity to the selected track.",
            )
            liked_count = sum(v == 1 for v in feedback.values())
            disliked_count = sum(v == -1 for v in feedback.values())
            st.caption(f"Your profile: {liked_count} liked, {disliked_count} disliked.")
            if not liked_count and not disliked_count:
                st.info("Rate a few songs first. For now, recommendations use acoustic similarity.")
        records = [{"id": s["id"], "title": s["title"], **s["features"]}
                   for s in library]
        results = recommendations(pd.DataFrame(records), sid, limit=10,
                                  feedback=feedback, preference_weight=weight,
                                  mode=mode)
        if results.empty:
            st.info("No unrated songs left to discover. Upload more tracks, or "
                    "clear a rating under Library & preferences.")
        elif mode == "similar":
            st.caption("Closest acoustic matches to the selected track. "
                       "Smaller distance is better; not a probability.")
        else:
            st.caption("Unrated tracks ranked using your likes and dislikes. "
                       "Higher preference score is better; not a probability.")
        for _, rec in results.iterrows():
            item = next(s for s in library if s["id"] == rec["id"])
            metric = (f"distance {rec['acoustic_distance']:.2f}"
                      if mode == "similar"
                      else f"preference score {rec['rank_score']:.3f}")
            st.markdown(f"**{rec['title']}** · {rec['tempo']:.0f} BPM · {metric}")
            st.audio(str(item["path"]))
            current = feedback.get(rec["id"], 0)
            st.caption("Current rating: " +
                       {1: "Liked", -1: "Disliked", 0: "Neutral"}[current])
            buttons = st.columns(3)
            for col, title, value in zip(
                buttons, ["👍 Like", "👎 Dislike", "Clear rating"], [1, -1, 0]
            ):
                if col.button(title, key=f"rec_rate_{rec['id']}_{value}",
                              disabled=current == value):
                    set_feedback(rec["id"], value)
                    st.rerun()
        st.caption("Ratings save immediately and rerank the results.")
        st.divider()
        st.subheader("AI audio embeddings (experimental)")
        st.caption("Try a pretrained CLAP model to find similar-sounding music. "
                   "First run downloads the model and analyzes each saved track. "
                   "This is optional, separate from the default feature-based recommender.")
        if st.button("Generate AI recommendations", key="run_clap"):
            try:
                from ai_embeddings import ai_recommendations
                with st.spinner("Computing or loading local audio embeddings..."):
                    ai_results = ai_recommendations(
                        library, sid, feedback=get_feedback(),
                        mode=mode, weight=weight, limit=10)
                st.session_state["ai_results"] = ai_results
                st.session_state["ai_context"] = (sid, mode, tuple(sorted(get_feedback().items())), weight)
            except ImportError:
                st.error("AI dependencies missing. In CMD run: python -m pip install -r requirements-ai.txt")
            except Exception as exc:
                st.error(f"AI analysis failed: {exc}")
        context = (sid, mode, tuple(sorted(get_feedback().items())), weight)
        if st.session_state.get("ai_context") == context:
            ai_results = st.session_state.get("ai_results", [])
            if not ai_results:
                st.info("No eligible unrated songs remain. Add tracks or change ratings.")
            for rec in ai_results:
                st.markdown(f"**{rec['title']}** · CLAP cosine similarity {rec['cosine_similarity']:.3f} "
                            f"· rank score {rec['rank_score']:.3f}")
                matched_song = next(s for s in library if s["id"] == rec["id"])
                st.audio(str(matched_song["path"]))


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
    st.write("**Current rating:** " + {1: "Liked 👍", -1: "Disliked 👎", 0: "Neutral"}[current_rating])
    rate_cols = st.columns(3)
    for col, title, value in zip(rate_cols, ["👍 Like", "👎 Dislike", "Clear rating"], [1, -1, 0]):
        if col.button(title, key=f"library_rate_{rating_id}_{value}",
                      disabled=current_rating == value):
            set_feedback(rating_id, value)
            st.rerun()
    st.warning("Removing a song deletes the locally stored audio copy and its rating.")
    if st.button("Delete selected song"):
        delete_song(sid)
        st.rerun()
