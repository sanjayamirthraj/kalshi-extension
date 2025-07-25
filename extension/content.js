console.log('Kalshi Market Finder content script loaded');

// Extract page content when page loads using Mozilla's Readability library
function extractPageContent() {
  try {
    // Clone the document to avoid modifying the original
    const documentClone = document.cloneNode(true);
    
    // Use Readability to extract the main article content
    const reader = new Readability(documentClone, {
      charThreshold: 100,
      maxElemsToParse: 0,
      nbTopCandidates: 10,
      debug: false,
    });
    const article = reader.parse();
    console.log('readability article: ', article);
    
    if (article) {
      // Use Readability's extracted content
      const title = article.title || document.title || '';
      const description = document.querySelector('meta[name="description"]')?.getAttribute('content') || '';
      
      // Get the text content from Readability's parsed content
      // Remove HTML tags and clean up whitespace
      let mainContent = article.textContent || article.content || '';
      mainContent = mainContent.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim();
      
      return {
        title,
        description,
        content: mainContent,
        url: window.location.href,
        excerpt: article.excerpt || '',
        byline: article.byline || '',
        readabilityParsed: true
      };
    }
  } catch (error) {
    console.warn('Readability parsing failed, falling back to manual extraction:', error);
  }
  
  // Fallback to original method if Readability fails
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
    url: window.location.href,
    readabilityParsed: false
  };
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getPageContent') {
    sendResponse(extractPageContent());
  }
});