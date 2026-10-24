import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from analyzer import analyze_audio
from recommender import recommendations

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")
st.title("🎵 Music Recommender")
st.caption("Analyze local recordings and find acoustically similar tracks.")


@st.cache_data(show_spinner=False)
def analyze_cached(content: bytes, filename: str):
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        path = Path(tmp.name)
    try:
        return analyze_audio(path)
    finally:
        path.unlink(missing_ok=True)


if "library" not in st.session_state:
    st.session_state.library = {}

uploaded = st.file_uploader(
    "Upload at least two recordings",
    type=["wav", "mp3", "flac", "ogg"],
    accept_multiple_files=True,
)
if uploaded:
    for item in uploaded:
        content = item.getvalue()
        key = hashlib.sha256(content).hexdigest()
        if key not in st.session_state.library:
            try:
                with st.spinner(f"Analyzing {item.name}..."):
                    features = analyze_cached(content, item.name)
                st.session_state.library[key] = {
                    "id": key, "title": item.name,
                    "content": content, "features": features
                }
            except Exception as exc:
                st.error(f"Could not analyze {item.name}: {exc}")

if st.button("Clear library"):
    st.session_state.library = {}
    st.rerun()

library = list(st.session_state.library.values())
st.write(f"**{len(library)} songs in your current session**")
if not library:
    st.info("Upload recordings to begin, or generate demo tracks using demo_audio.py.")
    st.stop()

selected = st.selectbox("Choose a song", library, format_func=lambda x: x["title"])
feat = selected["features"]
st.audio(selected["content"])
a, b, c, d = st.columns(4)
a.metric("Estimated tempo", f"{feat['tempo']:.0f} BPM")
b.metric("RMS energy", f"{feat['rms']:.3f}")
c.metric("Brightness", f"{feat['brightness']:.0f} Hz")
d.metric("Duration analyzed", f"{feat['duration']:.0f} sec")

with st.expander("All audio features"):
    st.dataframe(pd.DataFrame([feat]).T.rename(columns={0: "Value"}), use_container_width=True)

if len(library) > 1:
    records = [{"id": s["id"], "title": s["title"], **s["features"]} for s in library]
    results = recommendations(pd.DataFrame(records), selected["id"])
    st.subheader("Recommended similar songs")
    st.caption("Ranked by standardized acoustic-feature distance. Smaller means closer; not a match percentage.")
    for _, row in results.iterrows():
        other = st.session_state.library[row["id"]]
        st.markdown(f"**{row['title']}** · distance {row['distance']:.2f} · ~{row['tempo']:.0f} BPM")
        st.audio(other["content"])
else:
    st.info("Upload one more song to activate recommendations.")
