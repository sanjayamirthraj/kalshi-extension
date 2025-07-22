# Kalshi Market Finder Chrome Extension

An AI-powered Chrome extension that finds relevant Kalshi prediction markets based on semantic similarity to the content you're reading.

## Features

- **Semantic Search**: Uses SentenceTransformers embeddings for intelligent market matching
- **Keyword Extraction**: Shows categorized keywords (1-word, 2-word, 3-word) extracted from articles
- **Background Processing**: Automatically processes page content when you visit websites
- **Market Deduplication**: Groups related markets by event ticker to avoid duplicates
- **Real-time Updates**: Cached results with automatic refresh

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Server

```bash
python server.py
```

The server will start on `http://localhost:8000` and fetch all Kalshi markets on startup.

### 3. Install Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" and select this project directory
4. The extension icon should appear in your browser toolbar

## Usage

1. **Start the server**: Make sure `python server.py` is running
2. **Visit any webpage** with relevant content (news articles, blogs, etc.)
3. **Click the extension icon** to see:
   - Extracted keywords organized by word count
   - Most similar Kalshi markets ranked by AI similarity scores
   - Market prices and trading information

## How It Works

1. **Content Extraction**: Extension extracts text from article titles, descriptions, and main content
2. **Keyword Analysis**: TF-IDF algorithm identifies important terms and phrases  
3. **Semantic Search**: FastAPI server uses SentenceTransformers to find semantically similar markets
4. **Smart Ranking**: Markets are ranked by cosine similarity scores from the AI model
5. **Deduplication**: Only shows one market per event to avoid duplicate results

## API Endpoints

- `POST /similarity` - Search for similar markets
- `GET /health` - Check server status
- `POST /refresh-cache` - Manually refresh market cache

## Technical Stack

- **Frontend**: Chrome Extension (JavaScript)
- **Backend**: FastAPI (Python)
- **AI Model**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Data Source**: Kalshi API with pagination support