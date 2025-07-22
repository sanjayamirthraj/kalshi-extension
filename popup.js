// Popup script

// Import Transformers.js for embeddings
import * as transformers from './transformers.js';

// Initialize embedding model
let embeddingPipeline = null;
let modelLoaded = false;

// Load the sentence transformer model
async function initializeEmbeddingModel() {
  if (embeddingPipeline) return embeddingPipeline;
  
  try {
    console.log('Loading embedding model...');
    const { pipeline, env } = transformers;
    
    // Configure environment for Chrome extension
    env.allowRemoteModels = true;
    env.remoteHost = 'https://huggingface.co/';
    
    // Use a lightweight sentence transformer model
    embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      quantized: false,
    });
    
    modelLoaded = true;
    console.log('Embedding model loaded successfully');
    return embeddingPipeline;
  } catch (error) {
    console.error('Failed to load embedding model:', error);
    modelLoaded = false;
    return null;
  }
}

// Calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  if (vecA.length !== vecB.length) {
    throw new Error('Vectors must have the same length');
  }
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

// Generate text embedding
async function getTextEmbedding(text) {
  if (!embeddingPipeline) {
    await initializeEmbeddingModel();
  }
  
  if (!embeddingPipeline) {
    throw new Error('Embedding model not available');
  }
  
  try {
    // Truncate text to avoid memory issues
    const truncatedText = text.substring(0, 512);
    
    // Get embeddings with mean pooling
    const result = await embeddingPipeline(truncatedText, {
      pooling: 'mean',
      normalize: true,
    });
    
    // Convert to regular array
    return Array.from(result.data);
  } catch (error) {
    console.error('Error generating embedding:', error);
    throw error;
  }
}

// Function to calculate text similarity using vector embeddings
async function calculateSimilarity(pageContent, market) {
  try {
    // Prepare text content
    const pageText = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.trim();
    const marketText = `${market.title} ${market.subtitle || ''} ${market.yes_sub_title || ''} ${market.no_sub_title || ''}`.trim();
    
    if (!pageText || !marketText) {
      return 0;
    }
    
    // Generate embeddings for both texts
    const pageEmbedding = await getTextEmbedding(pageText);
    const marketEmbedding = await getTextEmbedding(marketText);
    
    // Calculate cosine similarity
    const similarity = cosineSimilarity(pageEmbedding, marketEmbedding);
    
    // Normalize to 0-1 range and ensure non-negative
    return Math.max(0, (similarity + 1) / 2); // Convert from [-1, 1] to [0, 1]
    
  } catch (error) {
    console.error('Error calculating embedding similarity:', error);
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
    
    console.log(`Calculating similarities for ${markets.length} markets...`);
    
    // Initialize embedding model first
    if (!modelLoaded) {
      console.log('Initializing embedding model...');
      await initializeEmbeddingModel();
    }
    
    // Calculate similarity scores for all markets using embeddings
    const batchSize = 5; // Process in smaller batches to avoid overwhelming the model
    const scoredMarkets = [];
    
    for (let i = 0; i < markets.length; i += batchSize) {
      const batch = markets.slice(i, i + batchSize);
      console.log(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(markets.length/batchSize)}`);
      
      const batchPromises = batch.map(async (market) => {
        const similarity = await calculateSimilarity(pageContent, market);
        return {
          ...market,
          similarity
        };
      });
      
      const batchResults = await Promise.all(batchPromises);
      scoredMarkets.push(...batchResults);
    }
    
    console.log('Similarity calculation completed');
    
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
    
    statusMessage.textContent = 'Finding related markets...';
    
    // Find relevant markets using AI similarity
    const relevantMarkets = await findRelevantMarkets(pageContent, marketResponse.markets);
    
    statusMessage.textContent = '';
    displayMarkets(relevantMarkets);
    
  } catch (error) {
    console.error('Error in popup:', error);
    displayError(error.message || 'Failed to analyze page content');
  }
});

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
