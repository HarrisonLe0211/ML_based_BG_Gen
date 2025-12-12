import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer

# Initialize FastAPI app
app = FastAPI(title="BGM Prediction API", version="1.0.0")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model_training" / "distilbert_genre_classifier.keras"
DATA_DIR = BASE_DIR / "model_training" / "content"
TOKENIZER_NAME = "distilbert-base-uncased"

# Global state
model = None
tokenizer = None
spotify_df = None
freesound_df = None
genre_to_id = None
id_to_genre = None

# Genre to Freesound keywords mapping
GENRE_FREESOUND_KEYWORDS = {
    "acoustic": ["guitar", "vocal", "piano", "singing", "folk"],
    "afrobeat": ["drum", "rhythm", "percussion", "beat", "dance"],
    "alt-rock": ["guitar", "rock", "electric", "band", "grunge"],
    "alternative": ["indie", "experimental", "synth", "vocal", "atmospheric"],
    "ambient": ["pad", "drone", "atmospheric", "soundscape", "meditation"],
    "anime": ["synth", "melody", "sfx", "cute", "cartoon", "japan"],
    "black-metal": ["scream", "distortion", "dark", "metal", "blast beat"],
    "bluegrass": ["banjo", "fiddle", "acoustic", "country", "folk"],
    "blues": ["guitar", "harmonica", "soul", "jazz", "vocal"],
    "brazil": ["samba", "bossa nova", "carnival", "latin", "percussion"],
    "breakbeat": ["beat", "drum break", "electronic", "loop", "bass"],
    "british": ["uk", "pop", "rock", "indie", "accent"],
    "cantopop": ["vocal", "chinese", "pop", "melody", "asia"],
    "chicago-house": ["house", "electronic", "synth", "four-on-the-floor", "dance"],
    "children": ["cartoon", "toy", "singing", "laughter", "play", "game", "story"],
    "chill": ["relax", "lounge", "smooth", "electronic", "downtempo"],
    "classical": ["piano", "violin", "orchestra", "flute", "cello", "symphony"],
    "club": ["dance", "beat", "electronic", "rave", "party"],
    "comedy": ["laughter", "squeak", "horn", "cartoon", "voice", "funny", "joke"],
    "country": ["guitar", "twang", "nashville", "folk", "western"],
}


def load_resources():
    """Load model, tokenizer, and data files on startup."""
    global model, tokenizer, spotify_df, freesound_df, genre_to_id, id_to_genre

    print("Loading model...")
    # Register custom objects for transformers models
    custom_objects = {}
    try:
        from transformers import TFDistilBertForSequenceClassification
        custom_objects['TFDistilBertForSequenceClassification'] = TFDistilBertForSequenceClassification
    except:
        pass
    
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), custom_objects=custom_objects)
        print(f"✓ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Tokenizer '{TOKENIZER_NAME}' loaded")

    print("Loading Spotify data...")
    spotify_df = pd.read_csv(DATA_DIR / "spotify.csv")
    print(f"Spotify data loaded: {len(spotify_df)} rows")

    print("Loading Freesound predictions...")
    dfs = []
    for i in range(10):
        try:
            df_temp = pd.read_csv(DATA_DIR / f"predictions_{i}.csv")
            dfs.append(df_temp)
        except FileNotFoundError:
            pass
    freesound_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"Freesound data loaded: {len(freesound_df)} rows")

    # Genre mappings (from notebook)
    # Extract unique genres from spotify data
    unique_genres = sorted(spotify_df["track_genre"].dropna().unique())
    genre_to_id = {genre: idx for idx, genre in enumerate(unique_genres)}
    id_to_genre = {idx: genre for genre, idx in genre_to_id.items()}
    print(f"Loaded {len(genre_to_id)} genres")


@app.on_event("startup")
async def startup_event():
    """Load resources on startup."""
    load_resources()


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    video_title: str


class SpotifyTrack(BaseModel):
    """Spotify track suggestion."""
    track_name: str
    artists: str
    genre: str


class FreesoundAudio(BaseModel):
    """Freesound audio suggestion."""
    fname: str
    label: str


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    video_title: str
    top_genres: list[dict]  # [{"genre": "...", "probability": 0.xx}, ...]
    spotify_suggestions: list[SpotifyTrack]
    freesound_suggestions: list[FreesoundAudio]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "BGM Prediction API is running",
        "available_endpoints": ["/predict", "/genres"],
    }


@app.get("/genres")
async def get_genres():
    """Get list of available genres."""
    if id_to_genre is None:
        return {"error": "Model not loaded"}
    return {"genres": list(id_to_genre.values()), "count": len(id_to_genre)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict background music genre from video title.

    Returns:
    - Top 3 predicted genres with probabilities
    - Example Spotify tracks from top genre
    - Example Freesound audio from top genre
    """
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}

    video_title = request.video_title.strip()
    if not video_title:
        return {"error": "Video title cannot be empty"}

    # Tokenize input
    inputs = tokenizer(
        video_title,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128,
    )

    # Get predictions
    outputs = model(inputs)
    logits = outputs[0]
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]

    # Get top 3 genres
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_genres = [
        {
            "genre": id_to_genre[idx],
            "probability": float(probabilities[idx]),
        }
        for idx in top_3_indices
    ]

    top_genre = id_to_genre[top_3_indices[0]]

    # Get Spotify suggestions (top 5 tracks from top genre)
    genre_tracks = spotify_df[spotify_df["track_genre"] == top_genre]
    spotify_suggestions = []
    if not genre_tracks.empty:
        sample_size = min(5, len(genre_tracks))
        sampled = genre_tracks.sample(n=sample_size, random_state=42)
        spotify_suggestions = [
            SpotifyTrack(
                track_name=row["track_name"],
                artists=row.get("artists", "Unknown"),
                genre=top_genre,
            )
            for _, row in sampled.iterrows()
        ]

    # Get Freesound suggestions (top 5 audio events from top genre)
    freesound_suggestions = []
    keywords = GENRE_FREESOUND_KEYWORDS.get(top_genre, [])
    if keywords and not freesound_df.empty:
        freesound_matches = freesound_df[
            freesound_df["label"].str.lower().apply(
                lambda x: any(keyword in x for keyword in keywords)
            )
        ]
        if not freesound_matches.empty:
            sample_size = min(5, len(freesound_matches))
            sampled = freesound_matches.sample(n=sample_size, random_state=42)
            freesound_suggestions = [
                FreesoundAudio(fname=row["fname"], label=row["label"])
                for _, row in sampled.iterrows()
            ]

    return PredictionResponse(
        video_title=video_title,
        top_genres=top_3_genres,
        spotify_suggestions=spotify_suggestions,
        freesound_suggestions=freesound_suggestions,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
