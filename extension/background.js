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

// Function to analyze sentiment using FastAPI server
async function analyzeSentiment(text) {
  try {
    console.log('Analyzing sentiment using FastAPI server...');
    const response = await fetch(`${SIMILARITY_API_URL}/sentiment`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text })
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    console.log('Sentiment analysis result:', data);
    return data;
  } catch (error) {
    console.error('Error analyzing sentiment:', error);
    throw error;
  }
}

// Function to compare article and market sentiment using FastAPI server
async function compareSentiment(articleText, yesPrice, noPrice) {
  try {
    const response = await fetch(`${SIMILARITY_API_URL}/compare_sentiment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: articleText,
        market_yes_price: yesPrice,
        market_no_price: noPrice
      })
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error comparing sentiment:', error);
    return { error: error.message };
  }
}

// Function to get real signal analysis using FastAPI server
async function getSignal(articleText, marketText) {
  try {
    console.log('Getting signal analysis from FastAPI server...', {
      url: `${SIMILARITY_API_URL}/get_signal`,
      articleTextLength: articleText?.length,
      marketTextLength: marketText?.length
    });
    
    const requestBody = {
      article_text: articleText,
      market_text: marketText
    };
    console.log('Request body:', requestBody);
    
    const response = await fetch(`${SIMILARITY_API_URL}/get_signal`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    console.log('Fetch response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('HTTP error response:', errorText);
      throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
    }

    const data = await response.json();
    console.log('Signal analysis result:', data);
    
    return data;
  } catch (error) {
    console.error('Error getting signal:', error);
    console.error('Error stack:', error.stack);
    throw error;
  }
}

// Function to get mock signal analysis using FastAPI server
async function getMockSignal(articleText, marketText) {
  try {
    console.log('Getting mock signal analysis from FastAPI server...');
    
    const response = await fetch(`${SIMILARITY_API_URL}/mock_get_signal`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        article_text: articleText,
        market_text: marketText
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('Mock signal analysis result:', data);
    
    return data;
  } catch (error) {
    console.error('Error getting mock signal:', error);
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
  
  if (request.action === 'getBackgroundResults') {
    // Get pre-computed results from background processing
    const cached = pageResultsCache.get(request.url);
    if (cached && cached.keywords && cached.similarMarkets) {
      sendResponse({
        success: true,
        keywords: cached.keywords,
        markets: cached.similarMarkets.results,
        cached: true
      });
    } else {
      sendResponse({
        success: false,
        error: 'No background results available',
        cached: false
      });
    }
  }

  if (request.action === 'analyzeSentiment') {
    analyzeSentiment(request.text)
      .then(result => {
        sendResponse({ success: true, result });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    return true; // Indicates async response
  }

  if (request.action === 'compareSentiment') {
    compareSentiment(request.text, request.yesPrice, request.noPrice)
      .then(result => {
        sendResponse({ success: true, result });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    return true; // Indicates async response
  }

  if (request.action === 'getSignal') {
    console.log('Background script received getSignal message', {
      articleTextLength: request.articleText?.length,
      marketTextLength: request.marketText?.length
    });
    
    getSignal(request.articleText, request.marketText)
      .then(analysis => {
        console.log('Background script getSignal success', analysis);
        sendResponse({ success: true, analysis });
      })
      .catch(error => {
        console.error('Background script getSignal error', error);
        sendResponse({ success: false, error: error.message });
      });
    return true; // Indicates async response
  }

  if (request.action === 'getMockSignal') {
    getMockSignal(request.articleText, request.marketText)
      .then(analysis => {
        sendResponse({ success: true, analysis });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    return true; // Indicates async response
  }
});
