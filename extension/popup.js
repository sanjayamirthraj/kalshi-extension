// Popup script

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
  
  // Clean up the content
  mainContent = mainContent.replace(/\s+/g, ' ').trim();
  
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
    
    statusMessage.textContent = 'Finding similar markets using AI...';
    
    // Prepare page text for similarity search (truncate to 2000 characters)
    const fullPageText = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.trim();
    const pageText = fullPageText.substring(0, 2000);
    
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
    
    // Store page content globally for AI analysis
    globalPageContent = pageContent;
    
    statusMessage.style.display = 'none';
    displayMarkets(relevantMarkets);
    
  } catch (error) {
    console.error('Error in popup:', error);
    displayError(error.message || 'Failed to analyze page content');
  }
});



// Store markets data globally for dropdown analysis
let globalMarketsData = [];
let globalPageContent = null;

// Function to load AI analysis for a market dropdown
async function loadAIAnalysis(index, dropdown, market) {
  // Show loading state
  dropdown.innerHTML = `
    <div style="display: flex; align-items: center; gap: 8px; color: #6b7280;">
      <div style="width: 16px; height: 16px; border: 2px solid #e5e7eb; border-top: 2px solid #3b82f6; border-radius: 50%; animation: spin 1s linear infinite;"></div>
      <span>Analyzing article relevance...</span>
    </div>
  `;
  
  // Add loading animation CSS if not already present
  if (!document.getElementById('loading-animation-styles')) {
    const style = document.createElement('style');
    style.id = 'loading-animation-styles';
    style.textContent = `
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(style);
  }
  
  try {
    // Prepare article text from page content (truncate to 10k characters for get_signal)
    let articleText = '';
    if (globalPageContent) {
      const fullArticleText = `${globalPageContent.title} ${globalPageContent.description} ${globalPageContent.content}`.trim();
      articleText = fullArticleText.substring(0, 10000);
    }

    console.log('articleText: ', articleText);
    
    // Prepare market text
    const marketText = `${market.title || ''} ${market.sub_title || ''}`.trim();
    
    // Call get_signal endpoint via background script
    console.log('Popup sending getSignal message', {
      articleTextLength: articleText?.length,
      marketTextLength: marketText?.length
    });
    
    const response = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        {
          action: 'getSignal',
          articleText: articleText,
          marketText: marketText
        },
        (result) => {
          if (chrome.runtime.lastError) {
            console.error('Chrome runtime error:', chrome.runtime.lastError);
            console.error('Full error details:', chrome.runtime.lastError.message);
            resolve({ success: false, error: 'Extension communication error' });
            return;
          }
          console.log('Popup received getSignal response', result);
          resolve(result);
        }
      );
    });
    
    if (!response.success) {
      throw new Error(response.error || 'Failed to get AI analysis');
    }
    
    // Format and display results
    const analysis = response.analysis;
    displayAIAnalysis(dropdown, analysis);
    
  } catch (error) {
    console.error('Error loading AI analysis:', error);
    dropdown.innerHTML = `
      <div style="color: #dc3545; font-size: 12px;">
        <strong>Error:</strong> ${error.message || 'Failed to load AI analysis'}
      </div>
    `;
  }
}

// Function to display AI analysis results
function displayAIAnalysis(dropdown, analysis) {
  const { recommendation, score, top_sentences } = analysis;
  
  // Get recommendation badge color
  let badgeColor = '#6c757d'; // gray for neutral
  if (recommendation === 'BUY') {
    badgeColor = '#28a745'; // green
  } else if (recommendation === 'SELL') {
    badgeColor = '#dc3545'; // red
  }
  
  // Build sentences HTML
  let sentencesHtml = '';
  if (top_sentences && top_sentences.length > 0) {
    sentencesHtml = `
      <div style="margin-top: 12px;">
        <div style="font-weight: 600; font-size: 12px; color: #374151; margin-bottom: 8px;">
          Supporting Evidence:
        </div>
        ${top_sentences.map(sentence => `
          <div style="margin-bottom: 8px; padding: 8px; background: #f8f9fa; border-radius: 4px; border-left: 3px solid ${sentence.sentiment_label === 'POSITIVE' ? '#28a745' : sentence.sentiment_label === 'NEGATIVE' ? '#dc3545' : '#6c757d'};">
            <div style="font-size: 11px; color: #495057; line-height: 1.4; margin-bottom: 4px;">
              "${sentence.sentence}"
            </div>
            <div style="font-size: 10px; color: #6c757d; display: flex; justify-content: space-between;">
              <span>Relevance: ${(sentence.similarity_score * 100).toFixed(1)}%</span>
              <span>${sentence.sentiment_label}: ${(sentence.sentiment_score * 100).toFixed(1)}%</span>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }
  
  dropdown.innerHTML = `
    <div>
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
          <div style="
            background: ${badgeColor}; 
            color: white; 
            padding: 4px 8px; 
            border-radius: 4px; 
            font-size: 11px; 
            font-weight: bold;
          ">
            ${recommendation}
          </div>
          <div style="font-size: 12px; color: #6b7280;">
            Confidence: ${score.toFixed(1)}%
          </div>
        </div>
      </div>
      ${sentencesHtml}
    </div>
  `;
}

function displayMarkets(markets) {
  const marketList = document.getElementById('market-list');
  
  // Store markets data for dropdown functionality
  globalMarketsData = markets;
  
  if (markets.length === 0) {
    marketList.innerHTML = '<li>No related markets found</li>';
    return;
  }
  
  // Add CSS for hover effects and dropdown functionality
  if (!document.getElementById('market-hover-styles')) {
    const style = document.createElement('style');
    style.id = 'market-hover-styles';
    style.textContent = `
      .market-row {
        cursor: pointer;
        transition: background-color 0.2s ease;
      }
      .market-row:hover {
        background-color: #f5f5f5 !important;
      }
      .dropdown-arrow {
        cursor: pointer;
        transition: transform 0.2s ease;
        color: #6b7280;
        font-size: 12px;
        margin-right: 8px;
        padding: 4px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
      }
      .dropdown-arrow:hover {
        background-color: #e5e7eb;
      }
      .dropdown-arrow.expanded {
        transform: rotate(90deg);
      }
      .market-dropdown {
        display: none;
        background: #f9fafb;
        padding: 12px 16px;
        font-size: 12px;
        color: #6b7280;
        line-height: 1.4;
        border-bottom: 1px solid #eee;
      }
      .market-dropdown.visible {
        display: block;
      }
      .market-main-content {
        flex: 1;
        display: flex;
        align-items: center;
      }
    `;
    document.head.appendChild(style);
  }
  
  const marketItems = markets.map((market, index) => {
    const marketUrl = `https://kalshi.com/events/${market.event_ticker}/${market.ticker}`;
    
    // Extract series ticker (everything before the first "-")
    const seriesTicker = market.ticker.split('-')[0];
    const iconUrl = `https://kalshi.com/_next/image?url=https%3A%2F%2Fd1lvyva3zy5u58.cloudfront.net%2Fseries-images-webp%2F${seriesTicker}.webp%3Fsize%3Dsm&w=256&q=80`;
    
    // Calculate price delta and determine colors/icons
    let priceDisplay = '';
    if (market.last_price) {
      const lastPrice = (market.last_price / 100).toFixed(2);
      let deltaDisplay = '';
      
      if (market.previous_price) {
        const delta = market.last_price - market.previous_price;
        const deltaValue = (delta / 100).toFixed(2);
        
        if (delta > 0) {
          // Positive delta: green color and up triangle
          deltaDisplay = `<div style="color: #28a745; font-size: 12px; font-weight: bold;">▲ +$${deltaValue}</div>`;
          priceDisplay = `<div style="color: #28a745; font-size: 14px; font-weight: bold;">$${lastPrice}</div>${deltaDisplay}`;
        } else if (delta < 0) {
          // Negative delta: red color and down triangle
          deltaDisplay = `<div style="color: #dc3545; font-size: 12px; font-weight: bold;">▼ -$${Math.abs(parseFloat(deltaValue)).toFixed(2)}</div>`;
          priceDisplay = `<div style="color: #dc3545; font-size: 14px; font-weight: bold;">$${lastPrice}</div>${deltaDisplay}`;
        } else {
          // No change: gray color
          priceDisplay = `<div style="color: #6c757d; font-size: 14px; font-weight: bold;">$${lastPrice}</div>`;
        }
      } else {
        // No previous price available: gray color
        priceDisplay = `<div style="color: #6c757d; font-size: 14px; font-weight: bold;">$${lastPrice}</div>`;
      }
    }
    
    if (priceDisplay === '') {
      return { html: '', url: '' };
    }
    
    return {
      html: `
        <div class="market-container">
          <div class="market-row" data-market-index="${index}" style="
            display: flex; 
            align-items: center; 
            padding: 12px;
            border-bottom: 1px solid #eee;
            background-color: white;
            position: relative;
          ">
            <div class="dropdown-arrow" data-dropdown-index="${index}">
              ▶
            </div>
            <div class="market-main-content" data-market-url="${marketUrl}">
              <img src="${iconUrl}" 
                   alt="${seriesTicker}" 
                   style="
                     width: 32px; 
                     height: 32px; 
                     margin-right: 12px; 
                     border-radius: 4px;
                     flex-shrink: 0;
                   " 
                   onerror="this.style.display='none'; this.nextElementSibling.style.display='inline-block';"
              />
              <div style="
                width: 32px; 
                height: 32px; 
                margin-right: 12px; 
                background-color: #f0f0f0; 
                border-radius: 4px; 
                display: none; 
                align-items: center; 
                justify-content: center; 
                font-size: 12px; 
                color: #666;
                flex-shrink: 0;
              ">?</div>
              <div style="flex: 1; min-width: 0;">
                <div style="font-weight: normal; font-size: 13px; line-height: 1.3; margin-bottom: 2px;">
                  ${market.title}
                </div>
                ${market.sub_title ? `<div style="font-size: 11px; color: #888; line-height: 1.2; font-family: monospace;">
                  ${market.sub_title}
                </div>` : ''}
              </div>
              <div style="
                text-align: right; 
                margin-left: 12px;
                flex-shrink: 0;
              ">
                ${priceDisplay}
              </div>
            </div>
          </div>
          <div class="market-dropdown" data-dropdown-content="${index}">
            AI analysis here
          </div>
        </div>
      `,
      url: marketUrl
    };
  }).filter(item => item.html !== '');

  const marketUrls = marketItems.map(item => item.url);
  const marketHtml = marketItems.map(item => item.html).join('');
  
  // Update the container to remove list styling
  marketList.style.listStyle = 'none';
  marketList.style.padding = '0';
  marketList.style.margin = '0';
  marketList.innerHTML = marketHtml;

  // Add click event listeners for dropdown arrows
  const dropdownArrows = marketList.querySelectorAll('.dropdown-arrow');
  dropdownArrows.forEach((arrow) => {
    arrow.addEventListener('click', async (e) => {
      e.stopPropagation();
      const index = arrow.getAttribute('data-dropdown-index');
      const dropdown = marketList.querySelector(`[data-dropdown-content="${index}"]`);
      
      // Toggle dropdown visibility
      const wasVisible = dropdown.classList.contains('visible');
      dropdown.classList.toggle('visible');
      arrow.classList.toggle('expanded');
      
      // If dropdown is now visible and doesn't have analysis yet, fetch it
      if (!wasVisible && dropdown.classList.contains('visible')) {
        await loadAIAnalysis(index, dropdown, globalMarketsData[index]);
      }
    });
  });

  // Add click event listeners for market content (excluding dropdown arrow)
  const marketMainContent = marketList.querySelectorAll('.market-main-content');
  marketMainContent.forEach((content) => {
    content.addEventListener('click', () => {
      const marketUrl = content.getAttribute('data-market-url');
      window.open(marketUrl, '_blank');
    });
  });
}

function displayError(message) {
  const marketList = document.getElementById('market-list');
  const statusMessage = document.getElementById('status-message');
  statusMessage.textContent = '';
  marketList.innerHTML = `<li style="color: red;">${message}</li>`;}