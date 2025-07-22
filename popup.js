// Popup script

// Simple keyword extraction without heavy ML models
let modelLoaded = true; // Always true for simple extraction

// Simple TF-IDF style keyword extraction
function calculateTFIDF(text) {
  // Common stop words
  const stopWords = new Set([
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'an', 'a', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
    'must', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our',
    'your', 'my', 'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'him', 'them',
    'us', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'than', 'then',
    'now', 'here', 'there', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
    'further', 'once', 'same', 'any', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'only', 'own', 'so', 'very', 'too', 'also', 'just', 'being',
    'during', 'before', 'after', 'above', 'below', 'between', 'through', 'into'
  ]);

  // Clean and tokenize text
  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 2 && !stopWords.has(word));

  // Calculate word frequencies
  const wordFreq = {};
  words.forEach(word => {
    wordFreq[word] = (wordFreq[word] || 0) + 1;
  });

  // Calculate TF scores and extract meaningful phrases
  const candidates = new Set();
  
  // Add single words
  Object.keys(wordFreq).forEach(word => candidates.add(word));
  
  // Add bigrams
  for (let i = 0; i < words.length - 1; i++) {
    const bigram = `${words[i]} ${words[i + 1]}`;
    candidates.add(bigram);
  }
  
  // Add trigrams
  for (let i = 0; i < words.length - 2; i++) {
    const trigram = `${words[i]} ${words[i + 1]} ${words[i + 2]}`;
    candidates.add(trigram);
  }

  // Score candidates
  const scored = [];
  candidates.forEach(candidate => {
    const candidateWords = candidate.split(' ');
    let score = 0;
    
    candidateWords.forEach(word => {
      if (wordFreq[word]) {
        score += Math.log(wordFreq[word] + 1); // TF with log scaling
      }
    });
    
    // Boost score for longer phrases and capitalized words in original text
    if (candidateWords.length > 1) score *= 1.5;
    if (text.includes(candidate.charAt(0).toUpperCase() + candidate.slice(1))) {
      score *= 1.3;
    }
    
    scored.push({ term: candidate, score });
  });

  return scored.sort((a, b) => b.score - a.score);
}

// Extract keywords using simple TF-IDF approach
async function extractKeywords(text, topK = 50) {
  try {
    console.log('Extracting keywords using TF-IDF approach...');
    
    // Use the TF-IDF calculation
    const scoredTerms = calculateTFIDF(text);
    
    return filteredTerms.slice(0, topK).map(item => item.term);
    
  } catch (error) {
    console.error('Error extracting keywords:', error);
    // Simple fallback
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 3);
    
    const stopWords = new Set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']);
    return words.filter(word => !stopWords.has(word)).slice(0, topK);
  }
}

// Function to calculate keyword-based relevance score
async function calculateKeywordRelevance(pageContent, market) {
  try {
    // Prepare text content
    const pageText = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.trim();
    const marketText = `${market.title} ${market.subtitle || ''} ${market.yes_sub_title || ''} ${market.no_sub_title || ''}`.trim();
    
    if (!pageText || !marketText) {
      return 0;
    }
    
    // Extract keywords from page content
    const pageKeywords = await extractKeywords(pageText, 15);
    
    if (pageKeywords.length === 0) {
      return 0;
    }
    
    // Clean market text for matching
    const cleanMarketText = marketText.toLowerCase();
    
    // Calculate keyword matches
    let matchScore = 0;
    let totalKeywordScore = 0;
    
    pageKeywords.forEach((keyword, index) => {
      const keywordWeight = 1 / (index + 1); // Higher weight for top keywords
      totalKeywordScore += keywordWeight;
      
      if (cleanMarketText.includes(keyword.toLowerCase())) {
        matchScore += keywordWeight;
      }
    });
    
    // Calculate relevance score (0-1 range)
    const relevance = totalKeywordScore > 0 ? matchScore / totalKeywordScore : 0;
    
    return relevance;
    
  } catch (error) {
    console.error('Error calculating keyword relevance:', error);
    return 0;
  }
}

// Function to find the most relevant markets
async function findRelevantMarkets(pageContent, markets, maxResults = 5) {
  try {
    if (!markets || markets.length === 0) {
      console.log('No markets found');
      return [];
    }
    
    // Deduplicate markets by event_ticker - keep the first occurrence
    const uniqueMarkets = [];
    const seenEventTickers = new Set();
    
    for (const market of markets) {
      if (!seenEventTickers.has(market.event_ticker)) {
        uniqueMarkets.push(market);
        seenEventTickers.add(market.event_ticker);
      }
    }
    
    console.log(`Deduplicated ${markets.length} markets to ${uniqueMarkets.length} unique events`);
    console.log(`Calculating similarities for ${uniqueMarkets.length} markets...`);
    
    // No model initialization needed for simple keyword extraction
    
    // Calculate keyword relevance scores for all unique markets
    const batchSize = 10; // Process in larger batches since keyword matching is faster
    const scoredMarkets = [];
    
    for (let i = 0; i < uniqueMarkets.length; i += batchSize) {
      const batch = uniqueMarkets.slice(i, i + batchSize);
      console.log(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(uniqueMarkets.length/batchSize)}`);
      
      const batchPromises = batch.map(async (market) => {
        const relevance = await calculateKeywordRelevance(pageContent, market);
        return {
          ...market,
          similarity: relevance
        };
      });
      
      const batchResults = await Promise.all(batchPromises);
      scoredMarkets.push(...batchResults);
    }
    
    console.log('Keyword relevance calculation completed');
    
    // Sort by similarity score (highest first) and return top results
    const results = scoredMarkets
      .filter(market => market.similarity > 0)
      .filter(market => !market.title?.includes('What will say during') && !market.title?.includes('Who will say during'))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, maxResults);
    
    console.log(`Returning ${results.length} relevant markets`);
    return results;
  } catch (error) {
    console.error('Error finding relevant markets:', error);
    throw error;
  }
}

// Function to extract page content (fallback if content script isn't available)
function extractPageContentScript() {
  // Get the main text content from the page
  const title = document.title || '';
  const description = document.querySelector('meta[name="description"]')?.getAttribute('content') || '';
  
  // Extract main content from common elements
  const contentSelectors = [
    'article',
    'main',
    '[role="main"]',
    '.content',
    '#content',
    '.post',
    '.article'
  ];
  
  let mainContent = '';
  for (const selector of contentSelectors) {
    const element = document.querySelector(selector);
    if (element) {
      mainContent = element.innerText;
      break;
    }
  }
  
  // Fallback to body content if no main content found
  if (!mainContent) {
    mainContent = document.body.innerText;
  }
  
  // Clean up the content - take first 1000 characters to avoid too much text
  mainContent = mainContent.replace(/\s+/g, ' ').trim().substring(0, 1000);
  
  return {
    title,
    description,
    content: mainContent,
    url: window.location.href
  };
}

// Function to get page content with fallback
async function getPageContent(tab) {
  try {
    // First try to get content from the content script
    const pageContent = await chrome.tabs.sendMessage(tab.id, { action: 'getPageContent' });
    if (pageContent) {
      return pageContent;
    }
  } catch (error) {
    console.log('Content script not available, using fallback method:', error);
  }
  
  try {
    // Fallback: inject script to extract content
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      function: extractPageContentScript
    });
    
    if (results && results[0] && results[0].result) {
      return results[0].result;
    }
  } catch (error) {
    console.error('Failed to extract page content using fallback:', error);
  }
  
  throw new Error('Could not extract page content. Please refresh the page and try again.');
}

document.addEventListener('DOMContentLoaded', async () => {
  const marketList = document.getElementById('market-list');
  const statusMessage = document.getElementById('status-message');
  
  try {
    statusMessage.textContent = 'Getting page content...';
    
    // Get the current active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab) {
      throw new Error('No active tab found');
    }
    
    statusMessage.textContent = 'Extracting content...';
    
    // Get page content with fallback handling
    const pageContent = await getPageContent(tab);
    
    console.log('Page content extracted:', pageContent);
    statusMessage.textContent = 'Fetching markets...';
    
    // Get markets from background script
    const marketResponse = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { action: 'fetchMarkets' },
        (result) => {
          if (chrome.runtime.lastError) {
            console.error('Chrome runtime error:', chrome.runtime.lastError);
            resolve({ success: false, error: 'Extension communication error' });
            return;
          }
          resolve(result);
        }
      );
    });
    
    if (!marketResponse.success) {
      throw new Error(marketResponse.error);
    }
    
    statusMessage.textContent = 'Extracting keywords...';
    
    // Extract and display keywords first
    const pageText = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.trim();
    const extractedKeywords = await extractKeywords(pageText);
    displayKeywords(extractedKeywords);
    
    statusMessage.textContent = 'Finding related markets...';
    
    // Find relevant markets using keyword matching
    const relevantMarkets = await findRelevantMarkets(pageContent, marketResponse.markets);
    
    statusMessage.textContent = '';
    displayMarkets(relevantMarkets);
    
  } catch (error) {
    console.error('Error in popup:', error);
    displayError(error.message || 'Failed to analyze page content');
  }
});

function displayKeywords(keywords) {
  const keywordsSection = document.getElementById('keywords-section');
  const keywordsList = document.getElementById('keywords-list');
  
  if (keywords.length === 0) {
    keywordsSection.style.display = 'none';
    return;
  }
  
  const keywordTags = keywords.map(keyword => 
    `<span class="keyword-tag">${keyword}</span>`
  ).join('');
  
  keywordsList.innerHTML = keywordTags;
  keywordsSection.style.display = 'block';
}

function displayMarkets(markets) {
  const marketList = document.getElementById('market-list');
  
  if (markets.length === 0) {
    marketList.innerHTML = '<li>No related markets found</li>';
    return;
  }
  
  const marketItems = markets.map(market => {
    const marketUrl = `https://kalshi.com/events/${market.event_ticker}/${market.ticker}`;
    const similarity = Math.round(market.similarity * 100);
    
    return `
      <li>
        <a href="${marketUrl}" target="_blank">
          <strong>${market.title}</strong>
          <br>
          <small>Match: ${similarity}%</small>
        </a>
      </li>
    `;
  }).join('');
  
  marketList.innerHTML = marketItems;
}

function displayError(message) {
  const marketList = document.getElementById('market-list');
  const statusMessage = document.getElementById('status-message');
  statusMessage.textContent = '';
  marketList.innerHTML = `<li style="color: red;">${message}</li>`;
}
