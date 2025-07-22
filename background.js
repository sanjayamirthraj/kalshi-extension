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

// Function to calculate text similarity using keyword matching
function calculateSimilarity(pageContent, market) {
  const pageText = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.toLowerCase();
  const marketText = `${market.title} ${market.subtitle} ${market.yes_sub_title} ${market.no_sub_title}`.toLowerCase();
  
  // Extract keywords from both texts
  const pageWords = pageText.match(/\b\w+\b/g) || [];
  const marketWords = marketText.match(/\b\w+\b/g) || [];
  
  // Remove common words that don't add much meaning
  const commonWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'will', 'be', 'is', 'are', 'was', 'were'];
  const filteredPageWords = pageWords.filter(word => word.length > 2 && !commonWords.includes(word));
  const filteredMarketWords = marketWords.filter(word => word.length > 2 && !commonWords.includes(word));
  
  // Count matching words
  let matches = 0;
  const marketWordSet = new Set(filteredMarketWords);
  
  for (const word of filteredPageWords) {
    if (marketWordSet.has(word)) {
      matches++;
    }
  }
  
  // Calculate similarity score
  const totalUniqueWords = new Set([...filteredPageWords, ...filteredMarketWords]).size;
  return totalUniqueWords > 0 ? matches / totalUniqueWords : 0;
}

// Function to find the most relevant markets
async function findRelevantMarkets(pageContent, maxResults = 5) {
  try {
    const markets = await fetchKalshiMarkets();
    
    // Calculate similarity scores for all markets
    const scoredMarkets = markets.map(market => ({
      ...market,
      similarity: calculateSimilarity(pageContent, market)
    }));
    
    // Sort by similarity score (highest first) and return top results
    return scoredMarkets
      .filter(market => market.similarity > 0)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, maxResults);
  } catch (error) {
    console.error('Error finding relevant markets:', error);
    throw error;
  }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'findMarkets') {
    findRelevantMarkets(request.pageContent)
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
