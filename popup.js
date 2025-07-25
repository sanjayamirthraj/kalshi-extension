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

  // Get page content
  let pageContent;
  try {
    // Try to get content from the content script
    pageContent = await new Promise((resolve) => {
      chrome.runtime.sendMessage({ action: 'getPageContent' }, (result) => {
        resolve(result);
      });
    });
  } catch (error) {
    pageContent = null;
  }

  if (pageContent && pageContent.title && pageContent.content) {
    const pageText = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.trim();
    
    // Show sentiment section
    const sentimentSection = document.getElementById('sentiment-section');
    if (sentimentSection) {
      sentimentSection.style.display = 'block';
    }
    
    // Request sentiment analysis from background
    chrome.runtime.sendMessage(
      { action: 'analyzeSentiment', text: pageText },
      (response) => {
        if (response && response.success && response.result) {
          displaySentiment(response.result);
        } else {
          displaySentimentError();
        }
      }
    );
  }
  
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
    
    // Get sentiment analysis and generate recommendations for each market
    if (pageContent && pageContent.title && pageContent.content) {
      const pageText = `${pageContent.title} ${pageContent.description} ${pageContent.content}`.trim();
      generateMarketRecommendations(pageText, relevantMarkets);
    }
    
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
  
  const marketItems = markets.map((market, index) => {
    const marketUrl = `https://kalshi.com/events/${market.event_ticker}/${market.ticker}`;
    
    // Extract series ticker (everything before the first "-")
    const seriesTicker = market.ticker.split('-')[0];
    const iconUrl = `https://kalshi.com/_next/image?url=https%3A%2F%2Fd1lvyva3zy5u58.cloudfront.net%2Fseries-images-webp%2F${seriesTicker}.webp%3Fsize%3Dsm&w=256&q=80`;
    
    // Calculate price display
    let priceDisplay = '';
    if (market.last_price) {
      const lastPrice = (market.last_price / 100).toFixed(2);
      let deltaDisplay = '';
      
      if (market.previous_price) {
        const delta = market.last_price - market.previous_price;
        const deltaValue = (delta / 100).toFixed(2);
        
        if (delta > 0) {
          deltaDisplay = `<div style="color: #28a745; font-size: 12px; font-weight: bold;">▲ +$${deltaValue}</div>`;
          priceDisplay = `<div style="color: #28a745; font-size: 14px; font-weight: bold;">$${lastPrice}</div>${deltaDisplay}`;
        } else if (delta < 0) {
          deltaDisplay = `<div style="color: #dc3545; font-size: 12px; font-weight: bold;">▼ -$${Math.abs(parseFloat(deltaValue)).toFixed(2)}</div>`;
          priceDisplay = `<div style="color: #dc3545; font-size: 14px; font-weight: bold;">$${lastPrice}</div>${deltaDisplay}`;
        } else {
          priceDisplay = `<div style="color: #6c757d; font-size: 14px; font-weight: bold;">$${lastPrice}</div>`;
        }
      } else {
        priceDisplay = `<div style="color: #6c757d; font-size: 14px; font-weight: bold;">$${lastPrice}</div>`;
      }
    }
    
    if (priceDisplay === '') {
      return '';
    }
    
    return `
      <div class="market-item" style="margin-bottom: 8px;">
        <div class="market-row" data-market-index="${index}" onclick="window.open('${marketUrl}', '_blank')" style="
          display: flex; 
          align-items: center; 
          padding: 12px; 
          border: 1px solid #eee;
          border-radius: 8px;
          text-decoration: none;
          color: inherit;
          background-color: white;
          cursor: pointer;
          transition: all 0.15s ease;
        " onmouseover="this.style.borderColor='#07C285'; this.style.boxShadow='0 4px 6px -1px rgb(0 0 0 / 0.1)'; this.style.transform='translateY(-1px)'" onmouseout="this.style.borderColor='#eee'; this.style.boxShadow='none'; this.style.transform='translateY(0)'">
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
            <button class="toggle-recommendation" data-market-index="${index}" id="toggle-${index}">
              <span class="recommendation-arrow" id="arrow-${index}">▼</span>
              AI Recommendation
            </button>
          </div>
          <div style="
            text-align: right; 
            margin-left: 12px;
            flex-shrink: 0;
          ">
            ${priceDisplay}
          </div>
        </div>
        <div class="market-recommendation-dropdown" id="dropdown-${index}">
          <div class="recommendation-content">
            <div class="recommendation-header">
              <div id="recommendation-badge-${index}" class="recommendation-badge recommend-neutral">Loading...</div>
            </div>
            <div class="recommendation-text" id="recommendation-text-${index}">
              Analyzing content for trading insights...
            </div>
          </div>
        </div>
      </div>
    `;
  }).filter(item => item !== '');

  // Update the container
  marketList.style.listStyle = 'none';
  marketList.style.padding = '0';
  marketList.style.margin = '0';
  marketList.innerHTML = marketItems.join('');
  
  // Add event listeners for recommendation toggles
  const toggleButtons = marketList.querySelectorAll('.toggle-recommendation');
  toggleButtons.forEach(button => {
    button.addEventListener('click', (event) => {
      event.stopPropagation();
      const marketIndex = button.getAttribute('data-market-index');
      toggleRecommendation(marketIndex);
    });
  });
}

function displayError(message) {
  const marketList = document.getElementById('market-list');
  const statusMessage = document.getElementById('status-message');
  statusMessage.textContent = '';
  marketList.innerHTML = `<li style="color: red;">${message}</li>`;
}

function displaySentiment(sentimentResult) {
  const sentimentScore = document.getElementById('sentiment-score');
  const sentimentLabel = document.getElementById('sentiment-label');
  
  if (!sentimentScore || !sentimentLabel) return;
  
  const sentiment = sentimentResult.sentiment;
  const score = sentimentResult.score;
  
  // Determine sentiment class and icon
  let sentimentClass = 'sentiment-neutral';
  let icon = '😐';
  let displayText = 'NEUTRAL';
  
  if (sentiment === 'POSITIVE') {
    sentimentClass = 'sentiment-positive';
    icon = '😊';
    displayText = 'POSITIVE';
  } else if (sentiment === 'NEGATIVE') {
    sentimentClass = 'sentiment-negative';
    icon = '😞';
    displayText = 'NEGATIVE';
  }
  
  // Update the display
  sentimentScore.className = `sentiment-score ${sentimentClass}`;
  sentimentLabel.innerHTML = `${icon} ${displayText} (${(score * 100).toFixed(1)}%)`;
}

function displaySentimentError() {
  const sentimentScore = document.getElementById('sentiment-score');
  const sentimentLabel = document.getElementById('sentiment-label');
  
  if (!sentimentScore || !sentimentLabel) return;
  
  sentimentScore.className = 'sentiment-score sentiment-neutral';
  sentimentLabel.innerHTML = '⚠️ Error analyzing sentiment';
}

function toggleRecommendation(marketIndex) {
  const dropdown = document.getElementById(`dropdown-${marketIndex}`);
  const arrow = document.getElementById(`arrow-${marketIndex}`);
  
  if (!dropdown || !arrow) return;
  
  if (dropdown.style.display === 'none' || dropdown.style.display === '') {
    dropdown.style.display = 'block';
    arrow.classList.add('expanded');
  } else {
    dropdown.style.display = 'none';
    arrow.classList.remove('expanded');
  }
}

function generateMarketRecommendations(pageText, markets) {
  if (!markets || markets.length === 0) return;
  
  // Process each market individually
  markets.forEach((market, index) => {
    generateSingleMarketRecommendation(pageText, market, index);
  });
}

function generateSingleMarketRecommendation(pageText, market, marketIndex) {
  // Calculate market prices
  let yesPrice = 0.5;
  let noPrice = 0.5;
  
  if (market.yes_ask !== undefined && market.yes_ask !== null) {
    yesPrice = market.yes_ask / 100;
    noPrice = market.no_ask !== undefined && market.no_ask !== null ? market.no_ask / 100 : (1 - yesPrice);
  } else if (market.last_price !== undefined && market.last_price !== null) {
    yesPrice = market.last_price / 100;
    noPrice = 1 - yesPrice;
  }
  
  // Get sentiment analysis and comparison
  chrome.runtime.sendMessage(
    { action: 'analyzeSentiment', text: pageText },
    (sentimentResponse) => {
      if (!sentimentResponse || !sentimentResponse.success) {
        updateRecommendationDisplay(marketIndex, 'NEUTRAL', 'Unable to analyze content');
        return;
      }
      
      chrome.runtime.sendMessage(
        {
          action: 'compareSentiment',
          text: pageText,
          yesPrice: yesPrice,
          noPrice: noPrice
        },
        (comparisonResponse) => {
          if (!comparisonResponse || !comparisonResponse.success) {
            updateRecommendationDisplay(marketIndex, 'NEUTRAL', 'Unable to generate recommendation');
            return;
          }
          
          const sentiment = sentimentResponse.result;
          const comparison = comparisonResponse.result;
          const delta = comparison.delta || 0;
          
          let recommendation = 'NEUTRAL';
          let supportingText = '';
          
          if (delta > 0.1) {
            recommendation = 'YES';
            supportingText = extractSupportingText(pageText, sentiment, 'positive');
          } else if (delta < -0.1) {
            recommendation = 'NO';
            supportingText = extractSupportingText(pageText, sentiment, 'negative');
          } else {
            recommendation = 'NEUTRAL';
            supportingText = extractSupportingText(pageText, sentiment, 'neutral');
          }
          
          updateRecommendationDisplay(marketIndex, recommendation, supportingText);
        }
      );
    }
  );
}

function extractSupportingText(pageText, sentiment, direction) {
  // Split text into sentences
  const sentences = pageText.split(/[.!?]+/).filter(s => s.trim().length > 10);
  
  // Keywords that suggest positive/negative sentiment
  const positiveKeywords = [
    'success', 'growth', 'increase', 'rising', 'boost', 'improve', 'gain', 'positive', 'upward', 
    'strengthen', 'advance', 'progress', 'benefit', 'win', 'winning', 'achieve', 'excellent',
    'strong', 'bullish', 'optimistic', 'favorable', 'good news', 'breakthrough', 'surge', 'climb'
  ];
  
  const negativeKeywords = [
    'decline', 'fall', 'drop', 'decrease', 'loss', 'fail', 'negative', 'down', 'weak',
    'struggle', 'crisis', 'problem', 'concern', 'worry', 'risk', 'threat', 'bearish',
    'pessimistic', 'unfavorable', 'bad news', 'setback', 'crash', 'collapse', 'plunge', 'tumble'
  ];
  
  const neutralKeywords = [
    'stable', 'steady', 'unchanged', 'maintain', 'continue', 'remain', 'consistent', 'balanced',
    'neutral', 'mixed', 'uncertain', 'unclear', 'waiting', 'pending', 'monitor', 'watch'
  ];
  
  let targetKeywords;
  let actionText;
  
  if (direction === 'positive') {
    targetKeywords = positiveKeywords;
    actionText = 'YES';
  } else if (direction === 'negative') {
    targetKeywords = negativeKeywords;
    actionText = 'NO';
  } else {
    targetKeywords = neutralKeywords;
    actionText = 'HOLD';
  }
  
  // Find sentences with relevant keywords
  const relevantSentences = sentences.filter(sentence => {
    const lowerSentence = sentence.toLowerCase();
    return targetKeywords.some(keyword => lowerSentence.includes(keyword));
  });
  
  if (relevantSentences.length > 0) {
    // Return the first relevant sentence, trimmed to reasonable length
    let supportingText = relevantSentences[0].trim();
    if (supportingText.length > 150) {
      supportingText = supportingText.substring(0, 147) + '...';
    }
    
    if (direction === 'positive') {
      return `"${supportingText}" - This positive development suggests market conditions may favor YES positions.`;
    } else if (direction === 'negative') {
      return `"${supportingText}" - This concerning development suggests market conditions may favor NO positions.`;
    } else {
      return `"${supportingText}" - This neutral or mixed signal suggests a HOLD position may be most appropriate.`;
    }
  }
  
  // Enhanced fallback with more specific contextual text
  if (direction === 'positive') {
    if (sentiment.sentiment === 'POSITIVE') {
      return `The content expresses optimism with ${(sentiment.score * 100).toFixed(1)}% confidence. This positive sentiment suggests developments that could drive the market outcome toward YES.`;
    } else {
      return 'Despite mixed sentiment, the analysis indicates conditions that may favor a positive market outcome, suggesting YES positions could be beneficial.';
    }
  } else if (direction === 'negative') {
    if (sentiment.sentiment === 'NEGATIVE') {
      return `The content expresses pessimism with ${(sentiment.score * 100).toFixed(1)}% confidence. This negative sentiment suggests developments that could drive the market outcome toward NO.`;
    } else {
      return 'Despite mixed sentiment, the analysis indicates concerning factors that may favor a negative market outcome, suggesting NO positions could be beneficial.';
    }
  } else {
    // Neutral/HOLD case
    return `The content sentiment (${sentiment.sentiment.toLowerCase()}, ${(sentiment.score * 100).toFixed(1)}% confidence) aligns closely with current market pricing. No strong directional bias detected - holding current positions recommended.`;
  }
}

function updateRecommendationDisplay(marketIndex, recommendation, supportingText) {
  const badge = document.getElementById(`recommendation-badge-${marketIndex}`);
  const text = document.getElementById(`recommendation-text-${marketIndex}`);
  
  if (!badge || !text) return;
  
  // Update badge
  let badgeClass = 'recommend-neutral';
  let badgeText = 'HOLD';
  
  if (recommendation === 'YES') {
    badgeClass = 'recommend-yes';
    badgeText = 'BUY YES';
  } else if (recommendation === 'NO') {
    badgeClass = 'recommend-no';
    badgeText = 'BUY NO';
  }
  
  badge.className = `recommendation-badge ${badgeClass}`;
  badge.textContent = badgeText;
  
  // Update supporting text
  text.textContent = supportingText;
}
