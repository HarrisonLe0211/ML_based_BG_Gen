import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [videoTitle, setVideoTitle] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const handlePredict = async (e) => {
    e.preventDefault()
    if (!videoTitle.trim()) {
      setError('Please enter a video title')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await axios.post('http://localhost:8000/predict', {
        video_title: videoTitle,
      })
      setResult(response.data)
    } catch (err) {
      setError(
        err.response?.data?.detail || 'Failed to get predictions. Make sure the backend is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1>🎵 Background Music Generator</h1>
        <p>Get music genre suggestions and audio recommendations for your video</p>
      </header>

      <main className="main">
        <form onSubmit={handlePredict} className="form">
          <input
            type="text"
            placeholder="Enter your video title..."
            value={videoTitle}
            onChange={(e) => setVideoTitle(e.target.value)}
            className="input"
            disabled={loading}
          />
          <button type="submit" className="button" disabled={loading}>
            {loading ? 'Predicting...' : 'Get Suggestions'}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="results">
            <h2>Results for: "{result.video_title}"</h2>

            {/* Top Genres Section */}
            <section className="section">
              <h3>🎬 Top Predicted Genres</h3>
              <div className="genres-grid">
                {result.top_genres.map((item, idx) => (
                  <div key={idx} className="genre-card">
                    <div className="genre-name">{item.genre}</div>
                    <div className="genre-probability">
                      {(item.probability * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* Spotify Suggestions */}
            {result.spotify_suggestions.length > 0 && (
              <section className="section">
                <h3>🎵 Spotify Recommendations</h3>
                <div className="suggestions-list">
                  {result.spotify_suggestions.map((track, idx) => (
                    <div key={idx} className="suggestion-item">
                      <div className="suggestion-icon">🎵</div>
                      <div className="suggestion-content">
                        <div className="suggestion-title">{track.track_name}</div>
                        <div className="suggestion-meta">{track.artists}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Freesound Suggestions */}
            {result.freesound_suggestions.length > 0 && (
              <section className="section">
                <h3>🎧 Freesound Audio Elements</h3>
                <div className="suggestions-list">
                  {result.freesound_suggestions.map((audio, idx) => (
                    <div key={idx} className="suggestion-item">
                      <div className="suggestion-icon">🔊</div>
                      <div className="suggestion-content">
                        <div className="suggestion-title">{audio.label}</div>
                        <div className="suggestion-meta">ID: {audio.fname}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
