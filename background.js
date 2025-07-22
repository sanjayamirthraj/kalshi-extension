// Background service worker
const KALSHI_API_URL = 'https://api.elections.kalshi.com/trade-api/v2/markets';

// Function to fetch all markets from Kalshi API
async function fetchKalshiMarkets() {
  try {
    const response = await fetch(KALSHI_API_URL);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data.markets || [];
  } catch (error) {
    console.error('Error fetching Kalshi markets:', error);
    throw error;
  }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'fetchMarkets') {
    fetchKalshiMarkets()
      .then(markets => {
        sendResponse({ success: true, markets });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    
    // Return true to indicate we will send a response asynchronously
    return true;
  }
});
