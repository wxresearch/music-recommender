"""Streamlit dashboard for local acoustic music discovery."""
import hashlib
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analyzer import analyze_audio
from library import add_song, delete_song, get_feedback, set_feedback, songs, get_similarity_feedback, rate_similarity, saved_discoveries, set_discovery_status, update_song_metadata, create_playlist, list_playlists, playlist_song_ids, add_to_playlist, remove_from_playlist, delete_playlist, get_ai_tags, save_ai_tags
from recommender import recommendations
from smart_features import explain_match, surprise_me, text_recommendations
from explore import parse_title, cover_for, taste_summary, music_map, mood_search, closest_section
from visualizations import overview
from intelligence import learned_taste, blended_taste_results, tag_songs, generate_playlist, cluster_descriptions, export_backup, restore_backup

def song_label(song):
    return f"{song['title']} — {song['artist']}" if song.get("artist") else song["title"]

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.7rem; max-width: 1260px;}
h1,h2,h3 {letter-spacing: -.02em;}
div.stButton>button, div.stLinkButton>a {border-radius: 12px;}
div[data-testid="stMetric"] {padding: 12px; border-radius: 14px; background:rgba(122,100,195,.08);}
</style>""", unsafe_allow_html=True)
st.title("🎵 Music Recommender")
st.caption("Local song analysis, persistent library, and personalized recommendations.")


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
                         format_func=song_label)
sid = selection["id"]
features = selection["features"]
st.audio(str(selection["path"]))

tab1, tab2, tab3, tab5, tab6, tab8 = st.tabs(["🎼 Analysis", "✨ Recommendations", "💜 Library & Preferences", "🌍 Online Discovery", "🪐 Taste & Map", "📁 Playlists"])
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
        learned = learned_taste(library, feedback) if mode == "personalized" else None
        if mode == "personalized":
            st.caption("🧠 Taste learning: " + ("trained from your likes and dislikes" if learned else
                       "needs at least 4 ratings covering both liked and disliked songs"))
        records = [{"id": s["id"], "title": s["title"], **s["features"]}
                   for s in library]
        results = recommendations(pd.DataFrame(records), sid, limit=10,
                                  feedback=feedback, preference_weight=weight,
                                  mode=mode)
        if mode == "personalized" and learned is not None:
            results, _ = blended_taste_results(library, sid, feedback)
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
                      else (f"learned score {rec['learned_score']:.3f}" if "learned_score" in rec else f"preference score {rec['rank_score']:.3f}"))
            st.markdown(f"**{rec['title']}** · {rec['tempo']:.0f} BPM · {metric}")
            st.audio(str(item["path"]))
            st.caption("Why this recommendation? " + " · ".join(explain_match(selection, item)))
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
                st.caption("Feature comparison (not an explanation of CLAP attention): " + " · ".join(explain_match(selection, matched_song)))
                pair = tuple(sorted((sid, rec["id"])))
                prior = get_similarity_feedback().get(pair, 3)
                with st.form(key="similarity_" + sid + "_" + rec["id"]):
                    judgment = st.select_slider("How similar do these songs sound?",
                                                options=[1, 2, 3, 4, 5], value=prior)
                    if st.form_submit_button("Save similarity rating"):
                        rate_similarity(sid, rec["id"], int(judgment))
                        st.rerun()


        st.divider()
        st.subheader("🔎 Describe the music you want")
        st.caption("Search your own uploaded recordings using CLAP audio/text matching. It does not search Spotify.")
        description = st.text_input("Music description", placeholder="gentle, nostalgic instrumental piano with no vocals")
        if st.button("Search songs by description"):
            if not description.strip():
                st.warning("Enter a description first.")
            else:
                try:
                    with st.spinner("Comparing your description with local audio..."):
                        st.session_state["text_matches"] = text_recommendations(library, description.strip())
                    st.session_state["text_prompt"] = description.strip()
                except ImportError:
                    st.error("Install CLAP: python -m pip install -r requirements-ai.txt")
                except Exception as exc:
                    st.error("Text search failed: " + str(exc))
        if st.session_state.get("text_prompt") == description.strip():
            for song, similarity in st.session_state.get("text_matches", []):
                st.write(song_label(song) + " · CLAP text/audio cosine: " + format(similarity, ".3f"))
                st.audio(str(song["path"]))
        st.divider()
        st.subheader("🎲 Surprise Me")
        novelty = st.slider("Adventure level", 0.0, 1.0, 0.5, 0.1,
                            help="Higher values favor more distant-sounding songs.")
        if st.button("Surprise me!"):
            st.session_state["surprise_results"] = surprise_me(library, sid, get_feedback(), novelty)
        for song in st.session_state.get("surprise_results", []):
            st.write(song_label(song))
            st.audio(str(song["path"]))


with tab3:
    st.subheader("Saved music library")
    feedback = get_feedback()
    sort_mode = st.selectbox("Sort by", ["Title", "Favorites first", "Tempo (highest first)"])
    search_text = st.text_input("Search your songs", placeholder="Type a title...", key="library_search")
    matching = [song for song in library
                if search_text.casefold().strip() in song_label(song).casefold()]
    if sort_mode == "Favorites first":
        matching.sort(key=lambda song: feedback.get(song["id"], 0), reverse=True)
    elif sort_mode == "Tempo (highest first)":
        matching.sort(key=lambda song: song["features"].get("tempo", 0), reverse=True)
    st.caption(f"Showing {len(matching)} of {len(library)} songs. Rate songs directly below.")
    for song in matching:
        current_rating = feedback.get(song["id"], 0)
        with st.container(border=True):
            title_col, like_col, dislike_col, clear_col = st.columns([5, 1.4, 1.4, 1.4], vertical_alignment="center")
            artist, title = song.get("artist", ""), song["title"]
            title_col.markdown(f"**{title}**")
            if artist:
                title_col.caption(artist)
            cover = cover_for(song)
            if cover:
                title_col.image(cover, width=90)
            title_col.caption({1: "👍 Liked", -1: "👎 Disliked", 0: "Not rated"}[current_rating])
            with title_col.expander("✏️ Edit title / artist"):
                with st.form("edit_song_" + song["id"]):
                    new_title = st.text_input("Song title", value=song["title"])
                    new_artist = st.text_input("Artist", value=song.get("artist", ""))
                    if st.form_submit_button("Save changes"):
                        try:
                            update_song_metadata(song["id"], new_title, new_artist)
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
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
        song_search = st.text_input(
            "Search your library", placeholder="Search by song title...",
            key="discovery_song_search")
        matches = [song for song in library
                   if song_search.strip().casefold() in song["title"].casefold()]
        st.caption(f"Showing {len(matches)} of {len(library)} saved songs.")
        if matches:
            source_song = st.selectbox(
                "Choose a song", matches, format_func=song_label,
                key="discovery_any_song")
        else:
            st.info("No songs match your search. Try a different title.")
    elif discovery_source == "Liked songs only":
        if liked_songs:
            st.caption(f"Showing {len(liked_songs)} liked songs out of {len(library)} total.")
            source_song = st.selectbox(
                "Choose a liked song", liked_songs,
                format_func=lambda x: x["title"], key="discovery_favorite")
        else:
            st.info("You haven't liked any songs yet. Switch to All songs to browse your library.")
    suggested_track = source_song["title"]
    suggested_artist = source_song.get("artist", "")
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
    st.subheader("Mood discovery")
    mood = st.selectbox("Mood", ["Dreamy piano", "Melancholic", "Energetic electronic", "Peaceful ambient", "Lo-fi", "Instrumental"])
    if st.button("Find songs by mood"):
        if not api_key:
            st.warning("Configure LASTFM_API_KEY first.")
        else:
            try:
                st.session_state["discovery_results"] = mood_search(mood, api_key)
                st.session_state["discovery_query"] = ("mood", mood)
            except Exception as exc:
                st.error(str(exc))
    if st.session_state.get("discovery_query") in ((artist_name, track_name), ("mood", mood)):
        found = st.session_state.get("discovery_results", [])
        if not found:
            st.info("No matches even after checking tags and related artists. Try another track or verify the artist and title.")
        statuses = saved_discoveries()
        found = [rec for rec in found if statuses.get((rec["artist"].casefold(), rec["track"].casefold()), {}).get("status") not in ("Not interested", "Already heard")]
        from online_discovery import youtube_music_search_url
        for i, rec in enumerate(found, 1):
            st.markdown(f"**{i}. {rec['track']} — {rec['artist']}**")
            if rec["lastfm_match"] is not None:
                st.caption(f"Last.fm similarity: {rec['lastfm_match']:.3f} · {rec['source']} (not CLAP)")
            else:
                st.caption(f"Discovered through {rec['source']} · not a CLAP-scored match")
            listen = st.columns(2)
            with listen[0]:
                st.link_button("🎵 Spotify", rec["spotify_url"], use_container_width=True)
            with listen[1]:
                st.link_button("▶ YouTube Music", youtube_music_search_url(rec["artist"], rec["track"]), use_container_width=True)
            actions = st.columns(3)
            for col, label in zip(actions, ["Saved", "Not interested", "Already heard"]):
                if col.button(label, key="disc_" + str(i) + "_" + label):
                    set_discovery_status(rec["artist"], rec["track"], label, rec["spotify_url"])
                    st.rerun()
            if rec["lastfm_url"].startswith("https://"):
                st.markdown(f"[See on Last.fm]({rec['lastfm_url']})")

    st.divider()
    st.subheader("Saved discoveries")
    saved_items = [v for v in saved_discoveries().values() if v["status"] == "Saved"]
    if saved_items:
        st.dataframe(pd.DataFrame(saved_items)[["artist", "track"]], hide_index=True)
        with st.expander("Export to Spotify playlist"):
            st.caption("Requires Spotify developer app credentials and account authorization.")
            cid = st.text_input("Spotify Client ID", value=os.getenv("SPOTIPY_CLIENT_ID", ""))
            secret = st.text_input("Spotify Client Secret", type="password")
            redirect = st.text_input("Spotify redirect URI", value="http://127.0.0.1:8888/callback")
            playlist_name = st.text_input("Playlist title", value="Music Recommender Discoveries")
            if st.button("Create private Spotify playlist"):
                if not cid or not secret:
                    st.warning("Enter your Spotify app credentials.")
                else:
                    try:
                        from explore import spotify_export
                        playlist_url, missing = spotify_export(
                            [(x["artist"], x["track"]) for x in saved_items],
                            cid, secret, redirect, playlist_name)
                        if playlist_url:
                            st.link_button("Open playlist", playlist_url)
                        if missing:
                            st.warning(str(len(missing)) + " tracks were not found on Spotify.")
                    except Exception as exc:
                        st.error("Spotify export failed: " + str(exc))
    else:
        st.info("Save an online discovery to collect songs for your playlist.")

with tab6:
    st.subheader("Musical taste profile")
    profile = taste_summary(library, get_feedback())
    if profile.empty:
        st.info("Rate some songs to see how their audio features differ.")
    else:
        st.dataframe(profile.round(3))
        st.caption("Averages by rating, in each feature's original units.")
    st.divider()
    st.subheader("🏷️ AI genre and mood tags")
    st.caption("CLAP estimates the closest matching labels. These are predictions, not verified genres.")
    if st.button("Analyze genre & mood", key="run_tags"):
        try:
            with st.spinner("Classifying local audio with CLAP..."):
                generated_tags = tag_songs(library)
                save_ai_tags(generated_tags)
                st.session_state["music_tags"] = generated_tags
        except Exception as exc:
            st.error("Tag analysis failed: " + str(exc))
    known_tags = get_ai_tags()
    if known_tags:
        rows = [{"Song": song_label(song), "Genre": known_tags[song["id"]]["genre"],
                 "Mood": known_tags[song["id"]]["mood"]}
                for song in library if song["id"] in known_tags]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.divider()
    st.subheader("🪐 Music Universe")
    st.caption("Each dot represents a song. Nearby dots tend to sound alike. The two hidden axes are just mathematical coordinates, not musical properties or genres.")
    use_clap = st.checkbox("Use CLAP audio embeddings", value=True)
    if st.button("✨ Build my music universe"):
        try:
            vectors = None
            if use_clap:
                from ai_embeddings import embedding_for
                with st.spinner("Mapping songs by sound..."):
                    vectors = {item["id"]: embedding_for(item) for item in library}
            st.session_state["map_points"] = music_map(library, vectors)
        except Exception as exc:
            st.error("Map failed: " + str(exc))
    if "map_points" in st.session_state:
        data = st.session_state["map_points"].copy()
        if len(data) >= 3 and set(data["id"]) == {item["id"] for item in library}:
            import plotly.express as px
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            coords = data[["x", "y"]].to_numpy()
            group_count = min(4, max(2, len(data)//8))
            labels = KMeans(n_clusters=group_count, n_init=10, random_state=5).fit_predict(
                StandardScaler().fit_transform(coords))
            cluster_names = {}
            for v in range(group_count):
                member_ids = [song_id for song_id, label in zip(data["id"], labels) if label == v]
                cluster_names[v] = f"Cluster {v+1} · " + cluster_descriptions(library, member_ids, known_tags)
            data["Sound group"] = [cluster_names[int(v)] for v in labels]
            ratings = get_feedback()
            data["Preference"] = [{1:"Liked", -1:"Disliked", 0:"Unrated"}.get(ratings.get(k,0)) for k in data["id"]]
            fig = px.scatter(
                data, x="x", y="y", color="Sound group",
                hover_name="title", custom_data=["id"],
                hover_data={"Preference": True, "x": False, "y": False, "Sound group": False},
                category_orders={"Sound group": [cluster_names[i] for i in range(group_count)]},
                color_discrete_sequence=["#78CDE8", "#B7A5E8", "#EAAE83", "#8ED5B0"],
                template="plotly_dark")
            fig.update_traces(marker=dict(size=15, opacity=0.9,
                                          line=dict(width=1.3, color="#E9ECF5")))
            fig.update_layout(
                height=550, margin=dict(l=18, r=18, t=22, b=18),
                paper_bgcolor="#171A29", plot_bgcolor="#171A29",
                font=dict(color="#ECEFFA"),
                hoverlabel=dict(bgcolor="#252A40", font_size=13),
                legend=dict(
                    title=dict(text="Sound groups", font=dict(size=12, color="#E6E8F0")),
                    x=0.985, y=0.025, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(30,34,52,0.94)",
                    bordercolor="rgba(225,230,245,0.25)", borderwidth=1,
                    font=dict(color="#F1F3FA", size=12),
                    orientation="v", itemsizing="constant"))
            fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
            fig.update_yaxes(visible=False, showgrid=False, zeroline=False,
                             scaleanchor="x", scaleratio=1)
            event = st.plotly_chart(
                fig, use_container_width=True, on_select="rerun",
                selection_mode="points", key="music_universe",
                config={"displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})
            st.caption("Sound groups use nearby map positions. If you analyze tags above, each group also shows its dominant predicted genre and mood.")
            if event.selection.points:
                chosen_id = event.selection.points[0]["customdata"][0]
                matched = next((item for item in library if item["id"] == chosen_id), None)
                if matched:
                    st.markdown("**" + matched["title"] + "**")
                    st.audio(str(matched["path"]))
                    idx = data.index[data["id"] == chosen_id][0]
                    ds = np.linalg.norm(coords - coords[idx], axis=1)
                    nearest = [data.iloc[i]["title"] for i in np.argsort(ds) if i != idx][:3]
                    st.caption("Nearby songs: " + " · ".join(nearest))
        else:
            st.info("Add more songs or rebuild the map after changing your library.")

with tab8:
    st.subheader("📁 Your playlists")
    st.caption("Create playlists from songs already saved on this computer.")
    with st.form("new_playlist"):
        playlist_name = st.text_input("New playlist name", placeholder="Late night studying")
        if st.form_submit_button("Create playlist"):
            try:
                create_playlist(playlist_name)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    playlists = list_playlists()
    if not playlists:
        st.info("Create your first playlist above.")
    else:
        pl = st.selectbox("Choose playlist", playlists, format_func=lambda p: p["name"])
        members = playlist_song_ids(pl["id"])
        member_songs = [song for song in library if song["id"] in members]
        st.caption(str(len(members)) + " songs in this playlist")
        to_add = st.selectbox("Add a song", library, format_func=song_label, key="playlist_add")
        if st.button("➕ Add to playlist"):
            try:
                add_to_playlist(pl["id"], to_add["id"])
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        for item in member_songs:
            left, right = st.columns([5, 1])
            left.write(song_label(item))
            left.audio(str(item["path"]))
            if right.button("Remove", key="playlist_rm_" + str(pl["id"]) + item["id"]):
                remove_from_playlist(pl["id"], item["id"])
                st.rerun()
        with st.expander("Delete playlist"):
            st.warning("This removes the playlist only, not the underlying songs.")
            if st.button("Delete this playlist", key="delete_playlist_button"):
                delete_playlist(pl["id"])
                st.rerun()


    st.divider()
    st.subheader("✨ Generate a playlist with AI")
    st.caption("Describe a vibe. CLAP finds local tracks and balances matching your prompt with variety.")
    with st.form("generate_playlist_form"):
        prompt = st.text_input("Describe your playlist", placeholder="Relaxing atmospheric music for late-night studying")
        name = st.text_input("Save as", value="AI Mix")
        count = st.slider("Number of tracks", 2, min(30, max(2, len(library))), min(10, max(2, len(library))))
        generate = st.form_submit_button("Generate playlist")
    if generate:
        if not prompt.strip():
            st.warning("Describe your playlist first.")
        else:
            try:
                with st.spinner("Selecting and balancing tracks..."):
                    picked = generate_playlist(library, prompt.strip(), count, feedback=get_feedback())
                st.session_state["generated_playlist_ids"] = [s["id"] for s in picked]
                st.session_state["generated_playlist_name"] = name
            except Exception as exc:
                st.error("Could not generate playlist: " + str(exc))
    chosen_ids = st.session_state.get("generated_playlist_ids", [])
    if chosen_ids:
        st.write("Suggested tracks:")
        by_id = {item["id"]: item for item in library}
        for song_id in chosen_ids:
            if song_id in by_id:
                st.write("• " + song_label(by_id[song_id]))
        if st.button("Save generated playlist"):
            try:
                playlist_name = st.session_state.get("generated_playlist_name", "AI Mix").strip() or "AI Mix"
                from sqlite3 import IntegrityError
                try:
                    create_playlist(playlist_name)
                except IntegrityError:
                    pass
                playlist = next(p for p in list_playlists() if p["name"] == playlist_name)
                for song_id in chosen_ids:
                    add_to_playlist(playlist["id"], song_id)
                st.success("Playlist saved to your local library.")
            except Exception as exc:
                st.error("Could not save playlist: " + str(exc))
    st.divider()
    st.subheader("💾 Backup and restore")
    st.caption("Download your local songs, ratings, playlists and discoveries as a ZIP. No API secrets are included.")
    if st.button("Prepare backup"):
        try:
            with st.spinner("Packaging local library..."):
                st.session_state["backup_zip"] = export_backup()
        except Exception as exc:
            st.error("Backup failed: " + str(exc))
    if "backup_zip" in st.session_state:
        st.download_button("⬇️ Download library backup", st.session_state["backup_zip"],
                           file_name="music-recommender-backup.zip", mime="application/zip")
    restore_file = st.file_uploader("Restore from a Music Recommender backup ZIP", type=["zip"], key="restore_zip")
    if restore_file is not None:
        st.caption("Restore merges missing songs and metadata without replacing existing entries.")
        confirm = st.checkbox("I want to merge this backup into my local library")
        if st.button("Restore backup", disabled=not confirm):
            try:
                with st.spinner("Verifying and restoring backup..."):
                    added = restore_backup(restore_file.getvalue())
                st.success(f"Restore complete. Imported {added} new songs; existing songs were kept.")
                st.rerun()
            except Exception as exc:
                st.error("Restore failed: " + str(exc))
