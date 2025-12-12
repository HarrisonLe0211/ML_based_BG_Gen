import streamlit as st
import json
import pandas as pd
from model import FakeGenreClassifier

id_to_genre = json.load(open("mappings/id_to_genre.json"))
genre_keywords = json.load(open("mappings/genre_keywords.json"))
df = pd.read_csv("data/sample_tracks.csv")
classifier = FakeGenreClassifier(id_to_genre)

st.title("🎵 Mini BGM Genre Prediction Demo")

title = st.text_input("Enter a video title:", "")

if st.button("Predict"):
    pred_genre, top3 = classifier.predict(title)

    st.subheader("🔮 Top Predicted Genres")
    for g, s in top3:
        st.write(f"**{g}** — {s:.2f}")

    st.subheader("🎶 Example Tracks")
    sample = df[df["genre"] == pred_genre].sample(3, replace=True)
    for _, row in sample.iterrows():
        st.write(f"- **{row['track']}** by *{row['artist']}*")
