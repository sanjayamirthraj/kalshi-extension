# Kalshi Market Finder Chrome Extension

AI-powered Chrome extension that finds relevant Kalshi prediction markets based on webpage content 

## Local Setup

### Prerequisites

- Python 3.8+
- Chrome browser
- Git

### 1. Clone the Repository

```bash
git clone <repo-url>
cd kalshi-extension
```

### 2. Set Up the FastAPI Backend

#### Install Python Dependencies

```bash
pip install fastapi uvicorn requests sentence-transformers transformers torch scikit-learn python-multipart
```

#### Start the Backend Server

```bash
cs server && python server.py
```

The server will start on `http://localhost:8000`. You should see:
```
Server starting on http://localhost:8000
Models loaded successfully
```

**Note**: The first run will download AI models (~1GB total), which may take a few minutes.

### 3. Install the Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top-right corner)
3. Click **"Load unpacked"**
4. Select the `kalshi-extension` folder
5. The extension icon should appear in your Chrome toolbar

### 4. Usage

1. **Automatic Processing**: Navigate to any webpage - the extension automatically processes content in the background
2. **View Results**: Click the extension icon to see:
   - **Named Entities**: Key people, organizations, locations detected
   - **Related Markets**: Relevant Kalshi prediction markets with similarity scores
3. **Open Markets**: Click any market to open it on Kalshi

## Architecture

### Backend (`server.py`)
- **FastAPI server** handling AI processing
- **Sentence Transformers** for semantic similarity (`paraphrase-multilingual-mpnet-base-v2`)
- **RoBERTa-Large NER** for entity extraction (`Jean-Baptiste/roberta-large-ner-english`)
- **Market caching** with 1-hour persistence
- **GPU acceleration** support

### Frontend Components
- **`background.js`**: Background processing and API communication
- **`content.js`**: Page content extraction
- **`popup.js`**: Extension UI logic
- **`popup.html`**: Modern UI with shadcn-inspired styling
- **`manifest.json`**: Extension configuration

### Data Flow
1. Content script extracts page content
2. Background script processes content using AI models
3. Results cached for instant display
4. Popup shows processed keywords and relevant markets

## API Endpoints

- `GET /health` - Server health check
- `POST /keywords` - Extract keywords and entities from text
- `POST /similarity` - Find similar markets using semantic search

## Configuration

### Model Configuration
Models are automatically downloaded on first run:
- **Similarity Model**: `paraphrase-multilingual-mpnet-base-v2` (420M parameters)
- **NER Model**: `Jean-Baptiste/roberta-large-ner-english` (355M parameters)

### Cache Settings
- **Market Cache**: 1 hour (configurable in `server.py`)
- **Background Results**: 50 pages max (configurable in `background.js`)

## Development

### Adding New Features
1. Backend changes: Modify `server.py`
2. Extension logic: Update `background.js` or `popup.js`
3. UI changes: Edit `popup.html` for styling

### Debugging
- Backend logs: Check console where `python server.py` is running
- Extension logs: Open Chrome DevTools → Console while extension popup is open
- Background script logs: Go to `chrome://extensions/` → Extension details → Inspect views: background page

## Troubleshooting

### Common Issues

**Extension shows "Server not running" error:**
- Ensure `python server.py` is running on localhost:8000
- Check firewall settings

**No markets found:**
- Verify the page has meaningful content
- Check if Kalshi API is accessible
- Look for backend server logs

**Extension not loading:**
- Refresh the extension at `chrome://extensions/`
- Check manifest.json syntax
- Ensure all files are present

### Performance Tips
- Server downloads models on first run (~1GB)
- Use GPU if available for faster processing
- Markets are cached for 1 hour to reduce API calls

## Credits

Created by [@sanjayamirthraj](https://twitter.com/sanjayamirthraj) & [@pranav_jad](https://twitter.com/pranav_jad)

## License

MIT
