// Popup script

// Import Transformers.js for keyword extraction
import * as transformers from './transformers.js';

// Initialize keyword extraction model
let keywordPipeline = null;
let modelLoaded = false;

// Load the keyword extraction model
async function initializeKeywordModel() {
  if (keywordPipeline) return keywordPipeline;
  
  try {
    console.log('Loading keyword extraction model...');
    const { pipeline, env } = transformers;
    
    // Configure environment for Chrome extension
    env.allowRemoteModels = true;
    env.remoteHost = 'https://huggingface.co/';
    
    // Use NER model for keyword/entity extraction
    keywordPipeline = await pipeline('token-classification', 'Xenova/bert-base-NER', {
      quantized: false,
    });
    
    modelLoaded = true;
    console.log('Keyword extraction model loaded successfully');
    return keywordPipeline;
  } catch (error) {
    console.error('Failed to load keyword extraction model:', error);
    modelLoaded = false;
    return null;
  }
}

// Extract keywords from text using Hugging Face NER model
async function extractKeywords(text, topK = 15) {
  if (!keywordPipeline) {
    await initializeKeywordModel();
  }
  
  if (!keywordPipeline) {
    throw new Error('Keyword extraction model not available');
  }
  
  try {
    // Truncate text to avoid memory issues
    const truncatedText = text.substring(0, 512);
    
    // Use NER model to extract named entities as keywords
    const entities = await keywordPipeline(truncatedText);
    
    // Process entities and extract unique keywords
    const keywords = new Set();
    const entityGroups = {};
    
    // Group consecutive tokens of the same entity
    entities.forEach(entity => {
      if (entity.score > 0.5) { // Only high-confidence entities
        const cleanWord = entity.word.replace(/^##/, ''); // Remove BERT subword markers
        
        if (entity.entity.startsWith('B-')) {
          // Beginning of entity
          const entityType = entity.entity.substring(2);
          if (!entityGroups[entity.start]) {
            entityGroups[entity.start] = { words: [cleanWord], type: entityType };
          }
        } else if (entity.entity.startsWith('I-')) {
          // Inside entity - find the group to append to
          const entityType = entity.entity.substring(2);
          const groupKey = Object.keys(entityGroups).find(key => {
            const group = entityGroups[key];
            return group.type === entityType && Math.abs(parseInt(key) - entity.start) < 10;
          });
          if (groupKey && entityGroups[groupKey]) {
            entityGroups[groupKey].words.push(cleanWord);
          }
        }
      }
    });
    
    // Convert entity groups to keywords
    Object.values(entityGroups).forEach(group => {
      if (group.words.length > 0) {
        const keyword = group.words.join(' ').toLowerCase().trim();
        if (keyword.length > 2) {
          keywords.add(keyword);
        }
      }
    });
    
    // Fallback: extract important single words if no entities found
    if (keywords.size === 0) {
      const words = truncatedText.toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(word => word.length > 3);
      
      const stopWords = new Set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'will', 'would', 'should', 'could']);
      
      // Get word frequencies
      const wordFreq = {};
      words.forEach(word => {
        if (!stopWords.has(word)) {
          wordFreq[word] = (wordFreq[word] || 0) + 1;
        }
      });
      
      // Add top frequent words as keywords
      Object.entries(wordFreq)
        .sort(([,a], [,b]) => b - a)
        .slice(0, Math.min(10, topK))
        .forEach(([word]) => keywords.add(word));
    }
    
    return Array.from(keywords).slice(0, topK);
    
  } catch (error) {
    console.error('Error extracting keywords:', error);
    // Fallback to simple word extraction
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
    
    // Initialize keyword extraction model first
    if (!modelLoaded) {
      console.log('Initializing keyword extraction model...');
      await initializeKeywordModel();
    }
    
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
