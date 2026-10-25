"""Streamlit dashboard for local acoustic music discovery."""
import hashlib
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analyzer import analyze_audio
from library import add_song, delete_song, get_feedback, set_feedback, songs, get_similarity_feedback, rate_similarity
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analysis", "Recommendations", "Library & Preferences", "Evaluation", "Online Discovery"])
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
        acoustic_weight = st.slider("Acoustic feature contribution", 0.0, 1.0, 0.25, 0.05)
        if st.button("Generate AI recommendations", key="run_clap"):
            try:
                from ai_embeddings import ai_recommendations
                with st.spinner("Computing or loading local audio embeddings..."):
                    ai_results = ai_recommendations(
                        library, sid, feedback=get_feedback(),
                        mode=mode, weight=weight, limit=10,
                        acoustic_weight=acoustic_weight, pair_ratings=get_similarity_feedback())
                st.session_state["ai_results"] = ai_results
                st.session_state["ai_context"] = (sid, mode, tuple(sorted(get_feedback().items())), weight, acoustic_weight, tuple(sorted(get_similarity_feedback().items())))
            except ImportError:
                st.error("AI dependencies missing. In CMD run: python -m pip install -r requirements-ai.txt")
            except Exception as exc:
                st.error(f"AI analysis failed: {exc}")
        context = (sid, mode, tuple(sorted(get_feedback().items())), weight, acoustic_weight, tuple(sorted(get_similarity_feedback().items())))
        if st.session_state.get("ai_context") == context:
            ai_results = st.session_state.get("ai_results", [])
            if not ai_results:
                st.info("No eligible unrated songs remain. Add tracks or change ratings.")
            for rec in ai_results:
                st.markdown(f"**{rec['title']}** · CLAP cosine similarity {rec['cosine_similarity']:.3f} "
                            f"· rank score {rec['rank_score']:.3f}")
                matched_song = next(s for s in library if s["id"] == rec["id"])
                st.audio(str(matched_song["path"]))
                pair = tuple(sorted((sid, rec["id"])))
                prior = get_similarity_feedback().get(pair, 3)
                with st.form(key="similarity_" + sid + "_" + rec["id"]):
                    judgment = st.select_slider("How similar do these songs sound?",
                                                options=[1, 2, 3, 4, 5], value=prior)
                    if st.form_submit_button("Save similarity rating"):
                        rate_similarity(sid, rec["id"], int(judgment))
                        st.rerun()


with tab3:
    st.subheader("Saved music library")
    feedback = get_feedback()
    search_text = st.text_input("Search your songs", placeholder="Type a title...", key="library_search")
    matching = [song for song in library
                if search_text.casefold().strip() in song["title"].casefold()]
    st.caption(f"Showing {len(matching)} of {len(library)} songs. Rate songs directly below.")
    for song in matching:
        current_rating = feedback.get(song["id"], 0)
        with st.container(border=True):
            title_col, like_col, dislike_col, clear_col = st.columns([5, 1.4, 1.4, 1.4], vertical_alignment="center")
            title_col.markdown(f"**{song['title']}**")
            title_col.caption({1: "👍 Liked", -1: "👎 Disliked", 0: "Not rated"}[current_rating])
            for col, label, value in (
                (like_col, "👍 Like", 1),
                (dislike_col, "👎 Dislike", -1),
                (clear_col, "Clear", 0),
            ):
                if col.button(label, key=f"library_rate_{song['id']}_{value}",
                              disabled=current_rating == value, use_container_width=True):
                    set_feedback(song["id"], value)
                    st.rerun()
    if not matching:
        st.info("No songs match your search.")
    st.warning("Removing a song deletes the locally stored audio copy and its rating.")
    if st.button("Delete selected song"):
        delete_song(sid)
        st.rerun()

with tab4:
    st.subheader("Compare recommendation methods")
    st.caption("Compare CLAP-only, acoustic-only, and hybrid similarity scores "
               "against your saved 1–5 song-pair ratings. Ratings are not used to "
               "calculate the benchmark predictions.")
    ratings = get_similarity_feedback()
    st.write(f"Rated song pairs: {len(ratings)}")
    eval_weight = st.slider("Hybrid acoustic weight for evaluation", 0.0, 1.0, 0.25, 0.05,
                            key="evaluation_weight")
    if st.button("Run evaluation", key="evaluation_run"):
        if len(ratings) < 3:
            st.warning("Rate at least 3 different song pairs first under Recommendations.")
        else:
            try:
                from ai_embeddings import embedding_for
                from evaluation import evaluate
                with st.spinner("Loading embeddings and evaluating all rated pairs..."):
                    embeddings = {song["id"]: embedding_for(song) for song in library}
                    pair_data, summary = evaluate(library, ratings, embeddings, eval_weight)
                st.session_state["evaluation_output"] = (pair_data, summary)
                st.session_state["evaluation_context"] = (
                    tuple(sorted(ratings.items())), tuple(s["id"] for s in library), eval_weight)
            except ImportError:
                st.error("Install optional AI dependencies: python -m pip install -r requirements-ai.txt")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
    current_context = (tuple(sorted(ratings.items())), tuple(s["id"] for s in library), eval_weight)
    if st.session_state.get("evaluation_context") == current_context:
        pair_data, summary = st.session_state["evaluation_output"]
        if pair_data.empty:
            st.info("No rated pairs matched songs in your current library.")
        else:
            st.subheader("Agreement with your judgments")
            st.dataframe(summary, hide_index=True, use_container_width=True)
            st.caption("Spearman correlation ranges from -1 to +1; higher means "
                       "the ordering agrees more closely with your ratings. "
                       "A blank value means there is not enough variation to calculate it. "
                       "With only a few rated pairs, results are highly uncertain.")
            st.subheader("Rated pair comparisons")
            st.dataframe(pair_data.round(3), hide_index=True, use_container_width=True)
            st.bar_chart(summary.set_index("Method")["Spearman correlation"])

with tab5:
    st.subheader("Discover songs online")
    st.caption("Find similar tracks using Last.fm listening data, then open a "
               "Spotify search to listen. These results are not CLAP audio comparisons.")
    import os
    from online_discovery import find_similar_tracks
    api_key = os.environ.get("LASTFM_API_KEY", "")
    try:
        api_key = st.secrets.get("LASTFM_API_KEY", api_key)
    except (FileNotFoundError, KeyError):
        pass
    if not api_key:
        st.info("Set LASTFM_API_KEY in Windows CMD before launching the app, "
                "or add it to Streamlit secrets. Get a free API key at "
                "https://www.last.fm/api/account/create.")
    liked_songs = [song for song in library if get_feedback().get(song["id"]) == 1]
    discovery_source = st.radio(
        "Discover based on",
        ["Currently selected song", "Any song in my library", "Liked songs only"],
        horizontal=True,
        key="discovery_source",
    )
    source_song = selection
    if discovery_source == "Any song in my library":
        st.caption(f"Showing all {len(library)} saved songs.")
        source_song = st.selectbox(
            "Choose a song", library, format_func=lambda x: x["title"],
            key="discovery_any_song")
    elif discovery_source == "Liked songs only":
        if liked_songs:
            st.caption(f"Showing {len(liked_songs)} liked songs out of {len(library)} total.")
            source_song = st.selectbox(
                "Choose a liked song", liked_songs,
                format_func=lambda x: x["title"], key="discovery_favorite")
        else:
            st.info("You haven't liked any songs yet. Switch to All songs to browse your library.")
    filename = Path(source_song["title"]).stem
    # Artist and title order differs across music libraries: both remain editable.
    guess_parts = filename.split(" - ", 1)
    suggested_track = guess_parts[0].strip()
    suggested_artist = guess_parts[1].strip() if len(guess_parts) > 1 else ""
    with st.form("lastfm_discovery_form"):
        artist_name = st.text_input("Artist", value=suggested_artist)
        track_name = st.text_input("Track title", value=suggested_track)
        count = st.slider("Number of suggestions", 5, 30, 15, 5)
        discover = st.form_submit_button("Find new songs")
    if discover:
        if not api_key:
            st.error("Add a Last.fm API key first.")
        else:
            try:
                with st.spinner("Finding similar tracks on Last.fm..."):
                    found = find_similar_tracks(artist_name, track_name, api_key, count)
                st.session_state["discovery_results"] = found
                st.session_state["discovery_query"] = (artist_name, track_name)
            except Exception as exc:
                st.error(f"Online discovery failed: {exc}")
    if st.session_state.get("discovery_query") == (artist_name, track_name):
        found = st.session_state.get("discovery_results", [])
        if not found:
            st.info("No matches even after checking tags and related artists. Try another track or verify the artist and title.")
        for i, rec in enumerate(found, 1):
            st.markdown(f"**{i}. {rec['track']} — {rec['artist']}**")
            if rec["lastfm_match"] is not None:
                st.caption(f"Last.fm similarity: {rec['lastfm_match']:.3f} · {rec['source']} (not CLAP)")
            else:
                st.caption(f"Discovered through {rec['source']} · not a CLAP-scored match")
            st.link_button("Search on Spotify", rec["spotify_url"])
            if rec["lastfm_url"].startswith("https://"):
                st.markdown(f"[See on Last.fm]({rec['lastfm_url']})")
