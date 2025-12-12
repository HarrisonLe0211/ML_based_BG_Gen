# 🎵 Background Music Generator - Full-Stack App

A full-stack web application that suggests background music genres and audio recommendations for video titles using a fine-tuned DistilBERT ML model.

## Overview

- **Backend**: FastAPI server that loads a pre-trained `.keras` model and makes predictions
- **Frontend**: React app with a clean, modern UI for submitting video titles and viewing suggestions
- **Model**: DistilBERT trained on Spotify track names → music genre classification
- **Data Sources**: 
  - Spotify track metadata (114K+ tracks, 50+ genres)
  - Freesound audio elements (94K+ audio events)

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Node.js 18+
- Pip, npm

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Start Backend (Terminal 1)

```bash
cd backend
python main.py
```

The backend will start on `http://localhost:8000`

Visit `http://localhost:8000/docs` for the Swagger API documentation.

### 4. Start Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

### 5. Use the App

Open `http://localhost:3000` and enter a video title to get suggestions!

Example titles:
- "Worlds smallest 4K headset"
- "10 satisfying gadgets you won't believe"
- "Learning JavaScript basics"
- "Meditation and relaxation guide"

## API Endpoints

### `GET /`
Health check endpoint.

```bash
curl http://localhost:8000/
```

### `GET /genres`
Get list of all available genres.

```bash
curl http://localhost:8000/genres
```

### `POST /predict`
Get music genre and audio suggestions for a video title.

**Request:**
```json
{
  "video_title": "Your video title here"
}
```

**Response:**
```json
{
  "video_title": "Your video title here",
  "top_genres": [
    {"genre": "ambient", "probability": 0.85},
    {"genre": "chill", "probability": 0.10},
    {"genre": "classical", "probability": 0.05}
  ],
  "spotify_suggestions": [
    {
      "track_name": "Peaceful Piano",
      "artists": "Pianist",
      "genre": "ambient"
    }
  ],
  "freesound_suggestions": [
    {
      "fname": "00012345.wav",
      "label": "Atmospheric pad, synthesizer"
    }
  ]
}
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"video_title": "Relaxing nature documentary"}'
```

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up --build
```

This will:
- Build and start the backend on port 8000
- Build and start the frontend on port 3000
- Both services communicate through a shared network

Access the app at `http://localhost:3000`

### Individual Docker Builds

**Backend:**
```bash
docker build -f Dockerfile.backend -t bgm-backend .
docker run -p 8000:8000 bgm-backend
```

**Frontend:**
```bash
docker build -f Dockerfile.frontend -t bgm-frontend .
docker run -p 3000:3000 bgm-frontend
```

## Project Structure

```
ML_based_BG_Gen/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── App.css          # Styles
│   │   └── main.jsx         # React entry point
│   ├── index.html           # HTML template
│   ├── vite.config.js       # Vite configuration
│   └── package.json         # Node dependencies
├── model_training/
│   ├── distilbert_genre_classifier.keras  # Trained model
│   └── content/
│       ├── spotify.csv                    # Spotify track data
│       ├── youtube_data.csv               # YouTube video titles
│       └── predictions_*.csv              # Freesound audio events
├── docker-compose.yml       # Docker multi-container setup
├── Dockerfile.backend       # Backend Docker image
└── Dockerfile.frontend      # Frontend Docker image
```

## Model Details

- **Architecture**: DistilBERT (distilled BERT) fine-tuned for sequence classification
- **Input**: Video title (text)
- **Output**: Probability distribution over 50+ music genres
- **Training Data**: 114K+ Spotify tracks mapped to music genres
- **Performance**: ~44% accuracy (varies by genre specificity)

The model can classify distinct genres (Classical, Cantopop, Brazil) with high accuracy (>80%) but struggles with overlapping genres (Alt-rock, Acoustic).

## Data Integration

### Spotify Data
Provides real-world music track examples for each predicted genre. The API returns 5 random tracks from the top predicted genre.

### Freesound Data
Provides complementary sound effects and audio events. Genre-specific keywords are used to match audio elements (e.g., "ambient" → keywords: ["pad", "drone", "atmospheric"]).

## Features

✅ Text-based music genre prediction
✅ Top 3 genre probabilities
✅ Spotify track suggestions from predicted genre
✅ Freesound audio element suggestions
✅ Clean, modern React UI
✅ Responsive design (mobile-friendly)
✅ Docker containerization
✅ CORS-enabled API
✅ Swagger API documentation

## Future Enhancements

- [ ] User authentication & history
- [ ] Filter suggestions by audio duration, BPM, etc.
- [ ] YouTube API integration to auto-fetch video metadata
- [ ] Direct audio playback from Spotify/Freesound
- [ ] Genre-based mood/energy filtering
- [ ] Multi-language support
- [ ] Advanced model with image/audio features
- [ ] Caching for repeated queries

## Troubleshooting

**Backend won't start:**
- Ensure TensorFlow and transformers are installed: `pip install -r backend/requirements.txt`
- Check if port 8000 is available

**Frontend can't connect to backend:**
- Verify backend is running on `http://localhost:8000`
- Check CORS is enabled (it is in `main.py`)
- In dev mode, frontend proxies to `http://backend:8000` via Vite

**Model loading errors:**
- Ensure `model_training/distilbert_genre_classifier.keras` exists
- Ensure `model_training/content/*.csv` data files exist

**Slow predictions:**
- DistilBERT inference takes a few seconds on CPU
- For production, use GPU or TensorFlow Serving for optimized inference

## License

MIT
