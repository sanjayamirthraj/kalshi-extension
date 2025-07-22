// Background service worker
const SIMILARITY_API_URL = 'http://localhost:8000';

// Cache for processed page results
const pageResultsCache = new Map();
const MAX_CACHE_SIZE = 50;

// Function to search similar markets using FastAPI server
async function searchSimilarMarkets(query, maxResults = 10) {
  try {
    console.log('Searching similar markets for query:', query);
    
    const response = await fetch(`${SIMILARITY_API_URL}/similarity`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query,
        max_results: maxResults
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log(`Found ${data.results.length} similar markets from ${data.total_markets_searched} total markets`);
    
    return {
      results: data.results,
      totalSearched: data.total_markets_searched,
      query: data.query
    };
  } catch (error) {
    console.error('Error searching similar markets:', error);
    throw error;
  }
}

// Function to extract keywords using FastAPI server
async function extractKeywords(text, maxKeywords = 15) {
  try {
    console.log('Extracting keywords using FastAPI server...');
    
    const response = await fetch(`${SIMILARITY_API_URL}/keywords`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text,
        max_keywords: maxKeywords
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log(`Extracted ${data.one_word.length + data.two_word.length + data.three_word.length} keywords`);
    
    return data;
  } catch (error) {
    console.error('Error extracting keywords:', error);
    throw error;
  }
}

// Check if FastAPI server is running
async function checkServerHealth() {
  try {
    const response = await fetch(`${SIMILARITY_API_URL}/health`);
    if (!response.ok) {
      throw new Error(`Server health check failed: ${response.status}`);
    }
    const health = await response.json();
    console.log('Server health:', health);
    return health.status === 'healthy';
  } catch (error) {
    console.error('Server health check failed:', error);
    return false;
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
    
    // Check if server is healthy
    const isServerHealthy = await checkServerHealth();
    if (!isServerHealthy) {
      console.warn('FastAPI server is not available, skipping background processing');
      return;
    }
    
    // Create query from page content
    const query = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.trim();
    
    // Search for similar markets
    const similarMarkets = await searchSimilarMarkets(query, 10);
    
    // Cache the results
    pageResultsCache.set(cacheKey, {
      pageContent: pageContent,
      similarMarkets: similarMarkets,
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
  if (request.action === 'searchSimilarMarkets') {
    searchSimilarMarkets(request.query, request.maxResults || 10)
      .then(results => {
        sendResponse({ success: true, results });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    
    // Return true to indicate we will send a response asynchronously
    return true;
  }
  
  if (request.action === 'extractKeywords') {
    extractKeywords(request.text, request.maxKeywords || 15)
      .then(keywords => {
        sendResponse({ success: true, keywords });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    
    return true;
  }
  
  if (request.action === 'checkServerHealth') {
    checkServerHealth()
      .then(isHealthy => {
        sendResponse({ success: true, isHealthy });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    
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
