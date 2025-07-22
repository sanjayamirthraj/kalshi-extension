// Popup script
document.addEventListener('DOMContentLoaded', async () => {
  const marketList = document.getElementById('market-list');
  
  try {
    // Get the current active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Get page content from the content script
    const response = await chrome.tabs.sendMessage(tab.id, { action: 'getPageContent' });
    
    if (!response) {
      throw new Error('Could not extract page content');
    }
    
    // Send page content to background script to find relevant markets
    chrome.runtime.sendMessage(
      { action: 'findMarkets', pageContent: response },
      (result) => {
        if (result.success) {
          displayMarkets(result.markets);
        } else {
          displayError(`Error: ${result.error}`);
        }
      }
    );
    
  } catch (error) {
    console.error('Error in popup:', error);
    displayError('Failed to analyze page content');
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
  marketList.innerHTML = `<li style="color: red;">${message}</li>`;
}
