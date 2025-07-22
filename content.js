console.log('Kalshi Market Finder content script loaded');

// Extract page content when page loads
function extractPageContent() {
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

// Send page content to background script for processing
function sendPageContentToBackground() {
  const pageContent = extractPageContent();
  
  // Only process if there's meaningful content
  if (pageContent.title && pageContent.content && pageContent.content.length > 50) {
    chrome.runtime.sendMessage({
      action: 'processPageContent',
      pageContent: pageContent
    });
  }
}

// Process page content when DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', sendPageContentToBackground);
} else {
  // DOM is already loaded
  sendPageContentToBackground();
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getPageContent') {
    sendResponse(extractPageContent());
  }
});