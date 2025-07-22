// Background service worker
const KALSHI_API_URL = 'https://api.elections.kalshi.com/trade-api/v2/markets?limit=1000&status=open';

// Cache for markets and processed results
let marketsCache = null;
let marketsCacheTime = 0;
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Cache for processed page results
const pageResultsCache = new Map();
const MAX_CACHE_SIZE = 50;

// Function to fetch all markets from Kalshi API
async function fetchKalshiMarkets() {
  try {
    // Return cached markets if still fresh
    if (marketsCache && (Date.now() - marketsCacheTime) < CACHE_DURATION) {
      return marketsCache;
    }

    console.log('Fetching fresh markets from Kalshi API...');
    const response = await fetch(KALSHI_API_URL);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    
    // Cache the results
    marketsCache = data.markets || [];
    marketsCacheTime = Date.now();
    
    return marketsCache;
  } catch (error) {
    console.error('Error fetching Kalshi markets:', error);
    throw error;
  }
}

// Function to process page content in background
async function processPageContentInBackground(pageContent) {
  try {
    console.log('Processing page content in background:', pageContent.title);
    
    // Check if we already processed this URL recently
    const cacheKey = pageContent.url;
    if (pageResultsCache.has(cacheKey)) {
      console.log('Using cached results for:', pageContent.url);
      return;
    }
    
    // Fetch markets
    const markets = await fetchKalshiMarkets();
    
    // We'll store the page content and let popup do the keyword processing
    // since it has the keyword extraction logic
    pageResultsCache.set(cacheKey, {
      pageContent: pageContent,
      markets: markets,
      timestamp: Date.now()
    });
    
    // Limit cache size
    if (pageResultsCache.size > MAX_CACHE_SIZE) {
      const firstKey = pageResultsCache.keys().next().value;
      pageResultsCache.delete(firstKey);
    }
    
    console.log('Background processing completed for:', pageContent.title);
    
  } catch (error) {
    console.error('Error processing page content in background:', error);
  }
}

// Listen for messages from content script and popup
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
  
  if (request.action === 'processPageContent') {
    // Process in background without blocking
    processPageContentInBackground(request.pageContent);
    // Don't send response as this is fire-and-forget
  }
  
  if (request.action === 'getCachedResults') {
    const cached = pageResultsCache.get(request.url);
    sendResponse(cached || null);
  }
});
