# Kalshi Market Finder Chrome Extension

A Chrome extension that analyzes the current page content and finds the most relevant Kalshi prediction markets using the Kalshi API.

## Features

- **Page Content Analysis**: Extracts text content from the current webpage including title, description, and main content
- **Market Matching**: Uses keyword similarity to find relevant Kalshi markets
- **Easy Access**: Simple popup interface showing top 5 related markets
- **Direct Links**: Click on any market to open it directly on Kalshi.com

## Installation

1. **Download the extension files** to a local folder (all files should be in the same directory)

2. **Open Chrome Extensions page**:
   - Go to `chrome://extensions/`
   - Or click the three dots menu → More tools → Extensions

3. **Enable Developer Mode**:
   - Toggle the "Developer mode" switch in the top right corner

4. **Load the extension**:
   - Click "Load unpacked"
   - Select the folder containing the extension files
   - The extension should now appear in your extensions list

5. **Pin the extension** (optional):
   - Click the puzzle piece icon in the toolbar
   - Click the pin icon next to "Kalshi Market Finder"

## Usage

1. Navigate to any webpage
2. Click the Kalshi Market Finder extension icon in your toolbar
3. The extension will analyze the page content and show related Kalshi markets
4. Click on any market to open it on Kalshi.com

## How It Works

1. **Content Extraction**: The content script extracts the page title, meta description, and main text content
2. **API Call**: The background script fetches all available markets from the Kalshi API
3. **Similarity Matching**: A keyword-based algorithm compares page content with market titles and descriptions
4. **Results Display**: Top matching markets are displayed with similarity scores

## Files Structure

- `manifest.json` - Extension configuration and permissions
- `content.js` - Content script for extracting page data
- `background.js` - Service worker handling API calls and market matching
- `popup.html` - Extension popup interface
- `popup.js` - Popup functionality and UI handling
- `README.md` - This documentation file

## API Usage

The extension uses the public Kalshi API endpoint:
- `https://api.elections.kalshi.com/trade-api/v2/markets`

No authentication is required for accessing market data.

## Privacy

This extension:
- Only reads content from pages you visit when you click the extension
- Sends page content to the Kalshi API for market matching
- Does not store any personal data
- Does not track your browsing history

## Troubleshooting

If the extension isn't working:

1. **Check permissions**: Make sure the extension has permission to access the current site
2. **Refresh the page**: Try refreshing the page and clicking the extension again
3. **Check console**: Open Developer Tools (F12) and check for any error messages
4. **Reload extension**: Go to `chrome://extensions/` and click the reload button for the extension

## Development

To modify the extension:

1. Make changes to the relevant files
2. Go to `chrome://extensions/`
3. Click the reload button for the extension
4. Test your changes

## Limitations

- The similarity algorithm is basic keyword matching (could be enhanced with NLP)
- Limited to 1000 characters of page content to avoid processing overhead
- Requires active internet connection to fetch Kalshi markets
- Only works on pages that can be read by the content script
