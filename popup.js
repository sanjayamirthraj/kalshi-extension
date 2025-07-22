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
    
    // Filter out very short or very common terms
    const filteredTerms = scoredTerms.filter(item => {
      const term = item.term;
      return term.length > 2 && 
             !term.match(/^\d+$/) && // Not just numbers
             item.score > 0.5; // Minimum score threshold
    });
    
    // Categorize keywords by word count
    const oneWord = [];
    const twoWord = [];
    const threeWord = [];
    
    filteredTerms.forEach((item, index) => {
      const wordCount = item.term.split(' ').length;
      const keywordObj = {
        term: item.term,
        score: item.score,
        rank: index + 1
      };
      
      if (wordCount === 1) {
        oneWord.push(keywordObj);
      } else if (wordCount === 2) {
        twoWord.push(keywordObj);
      } else if (wordCount === 3) {
        threeWord.push(keywordObj);
      }
    });
    
    // Return categorized keywords (take top 10 from each category)
    return {
      oneWord: oneWord.slice(0, 10),
      twoWord: twoWord.slice(0, 10),
      threeWord: threeWord.slice(0, 10),
      all: filteredTerms.slice(0, topK).map((item, index) => ({
        term: item.term,
        score: item.score,
        rank: index + 1
      }))
    };
    
  } catch (error) {
    console.error('Error extracting keywords:', error);
    // Simple fallback
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 3);
    
    const stopWords = new Set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']);
    const fallbackKeywords = words.filter(word => !stopWords.has(word)).slice(0, topK).map((term, index) => ({
      term,
      score: 1.0,
      rank: index + 1
    }));
    
    return {
      oneWord: fallbackKeywords,
      twoWord: [],
      threeWord: [],
      all: fallbackKeywords
    };
  }
}

// Function to calculate keyword-based relevance score with n-gram priority
async function calculateKeywordRelevance(pageContent, market) {
  try {
    // Prepare text content
    const pageText = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.trim();
    const marketText = `${market.title} ${market.subtitle || ''} ${market.yes_sub_title || ''} ${market.no_sub_title || ''}`.trim();
    
    if (!pageText || !marketText) {
      return 0;
    }
    
    // Extract keywords from page content
    const keywordCategories = await extractKeywords(pageText, 15);
    
    if (!keywordCategories || (!keywordCategories.oneWord && !keywordCategories.twoWord && !keywordCategories.threeWord)) {
      return 0;
    }
    
    // Clean market text for matching
    const cleanMarketText = marketText.toLowerCase();
    
    // Define priority weights: 1-word gets highest priority, then 2-word, then 3-word
    const priorityWeights = {
      oneWord: 3.0,   // Highest priority
      twoWord: 2.0,   // Medium priority  
      threeWord: 1.0  // Lowest priority
    };
    
    let totalMatchScore = 0;
    let totalPossibleScore = 0;
    
    // Process each category with different priority weights
    ['oneWord', 'twoWord', 'threeWord'].forEach(category => {
      const keywords = keywordCategories[category] || [];
      const priorityWeight = priorityWeights[category];
      
      keywords.forEach((keywordObj) => {
        const keyword = keywordObj.term || keywordObj;
        const baseScore = keywordObj.score || (1 / (keywordObj.rank || 1));
        const weightedScore = baseScore * priorityWeight;
        
        totalPossibleScore += weightedScore;
        
        if (cleanMarketText.includes(keyword.toLowerCase())) {
          totalMatchScore += weightedScore;
        }
      });
    });
    
    // Calculate relevance score (0-1 range)
    const relevance = totalPossibleScore > 0 ? totalMatchScore / totalPossibleScore : 0;
    
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
    
    statusMessage.textContent = 'Checking for background results...';
    
    // First, try to get pre-computed background results
    const backgroundResults = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { 
          action: 'getBackgroundResults',
          url: tab.url
        },
        (result) => {
          if (chrome.runtime.lastError) {
            console.error('Chrome runtime error:', chrome.runtime.lastError);
            resolve({ success: false });
            return;
          }
          resolve(result);
        }
      );
    });
    
    if (backgroundResults.success && backgroundResults.cached) {
      // Use pre-computed results - instant display!
      console.log('Using pre-computed background results!');
      statusMessage.textContent = 'Loading cached results...';
      
      displayKeywords(backgroundResults.keywords);
      displayMarkets(backgroundResults.markets);
      statusMessage.textContent = '';
      return;
    }
    
    // Fallback: No background results available, process fresh
    console.log('No background results available, processing fresh...');
    
    statusMessage.textContent = 'Checking server connection...';
    
    // Check if FastAPI server is running
    const serverHealthResponse = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { action: 'checkServerHealth' },
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
    
    if (!serverHealthResponse.success || !serverHealthResponse.isHealthy) {
      throw new Error('FastAPI server is not running. Please start the server with: python server.py');
    }
    
    statusMessage.textContent = 'Extracting content...';
    
    // Get page content with fallback handling
    const pageContent = await getPageContent(tab);
    
    console.log('Page content extracted:', pageContent);
    
    statusMessage.textContent = 'Extracting keywords using AI...';
    
    // Extract keywords using FastAPI server
    const pageText = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.trim();
    const keywordResponse = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { 
          action: 'extractKeywords',
          text: pageText,
          maxKeywords: 15
        },
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
    
    if (!keywordResponse.success) {
      console.warn('Keyword extraction failed, using fallback:', keywordResponse.error);
      // Fallback to simple keyword extraction
      const fallbackKeywords = await extractKeywords(pageText);
      displayKeywords(fallbackKeywords);
    } else {
      displayKeywords(keywordResponse.keywords);
    }
    
    statusMessage.textContent = 'Finding similar markets using AI...';
    
    // Use FastAPI server for semantic similarity search
    const similarityResponse = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { 
          action: 'searchSimilarMarkets',
          query: pageText,
          maxResults: 10
        },
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
    
    if (!similarityResponse.success) {
      throw new Error(similarityResponse.error);
    }
    
    const relevantMarkets = similarityResponse.results.results || [];
    
    statusMessage.textContent = '';
    displayMarkets(relevantMarkets);
    
  } catch (error) {
    console.error('Error in popup:', error);
    displayError(error.message || 'Failed to analyze page content');
  }
});

function displayKeywords(keywordCategories) {
  const keywordsSection = document.getElementById('keywords-section');
  const entitiesList = document.getElementById('entities-list');
  const keywordsList = document.getElementById('keywords-list');
  
  // Helper function to create keyword tags
  function createKeywordTags(keywords, tagClass = 'keyword-tag') {
    return keywords.map(keywordObj => {
      const keyword = keywordObj.term || keywordObj;
      const rank = keywordObj.rank || '';
      const type = keywordObj.type || '';
      const score = keywordObj.score || 0;
      const titleText = `Score: ${score.toFixed(2)}${type ? `, Type: ${type}` : ''}`;
      
      return `<span class="${tagClass}" title="${titleText}">${keyword}${type && type !== 'keyword' ? ` (${type})` : ''}</span>`;
    }).join('');
  }
  
  // Get entities and keywords from server response
  const entities = keywordCategories.entities || [];
  const keywords = keywordCategories.keywords || [];
  
  // Always show the section if we have any data
  keywordsSection.style.display = 'block';
  
  // Display entities (always shown, even if empty)
  if (entities.length > 0) {
    entitiesList.innerHTML = createKeywordTags(entities, 'entity-tag');
    document.getElementById('entities-section').style.display = 'block';
  } else {
    entitiesList.innerHTML = '<span style="color: #999; font-style: italic;">No named entities detected</span>';
    document.getElementById('entities-section').style.display = 'block';
  }
  
  // Display additional keywords
  if (keywords.length > 0) {
    keywordsList.innerHTML = createKeywordTags(keywords, 'keyword-tag-alt');
    document.getElementById('keywords-section-extra').style.display = 'block';
  } else {
    document.getElementById('keywords-section-extra').style.display = 'none';
  }
}

function displayMarkets(markets) {
  const marketList = document.getElementById('market-list');
  
  if (markets.length === 0) {
    marketList.innerHTML = '<li>No related markets found</li>';
    return;
  }
  
  const marketItems = markets.map(market => {
    const marketUrl = `https://kalshi.com/events/${market.event_ticker}/${market.ticker}`;
    const similarity = Math.round((market.similarity_score || market.similarity || 0) * 100);
    
    return `
      <li>
        <a href="${marketUrl}" target="_blank">
          <strong>${market.title}</strong>
          ${market.subtitle ? `<br><small style="color: #666;">${market.subtitle}</small>` : ''}
          <br>
          <small style="color: #007bff;">Similarity: ${similarity}%</small>
          ${market.last_price ? ` • Last: $${(market.last_price / 100).toFixed(2)}` : ''}
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
