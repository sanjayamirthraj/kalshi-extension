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
  // Display a test message in the popup UI
  const marketList = document.getElementById('market-list');
  if (marketList) {
    marketList.innerHTML = '<li class="market-item"><div class="market-link"><div class="empty-state">Hello Armaan</div></div></li>';
  }
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
    
    // Prepare page text for similarity search
    const pageText = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.trim();
    
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
    
    statusMessage.style.display = 'none';
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
  
  // Add CSS for hover effects
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
        <div class="market-row" data-market-index="${index}" style="
          display: flex; 
          align-items: center; 
          padding: 12px; 
          border-bottom: 1px solid #eee;
          text-decoration: none;
          color: inherit;
          background-color: white;
        ">
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

  // Add click event listeners to each market row
  const marketRows = marketList.querySelectorAll('.market-row');
  marketRows.forEach((row, index) => {
    row.addEventListener('click', () => {
      window.open(marketUrls[index], '_blank');
    });
  });
}

function displayError(message) {
  const marketList = document.getElementById('market-list');
  const statusMessage = document.getElementById('status-message');
  statusMessage.textContent = '';
  marketList.innerHTML = `<li style="color: red;">${message}</li>`;
}
