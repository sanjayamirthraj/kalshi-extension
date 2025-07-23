from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import re
from collections import Counter
import logging
import json
import os
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kalshi Market Similarity Search", version="1.0.0")

# Global variables
similarity_model = None
keyword_model = None
ner_pipeline = None
markets_cache = []
embeddings_cache = []
last_cache_update = 0
CACHE_DURATION = 3600  # 1 hour in seconds (3600 seconds)
CACHE_FILE = "kalshi_markets_cache.pkl"

# Global sentiment analysis pipeline
sentiment_pipeline = None

def get_sentiment_pipeline():
    global sentiment_pipeline
    if sentiment_pipeline is None:
        sentiment_pipeline = pipeline('sentiment-analysis')
    return sentiment_pipeline

def get_optimism_score(label, score):
    if label.upper() == 'POSITIVE':
        return score
    elif label.upper() == 'NEGATIVE':
        return 1 - score
    else:
        return 0.5  # Neutral or unknown

# Pydantic models
class SimilarityRequest(BaseModel):
    query: str
    max_results: int = 10

class KeywordRequest(BaseModel):
    text: str
    max_keywords: int = 15

class KeywordResponse(BaseModel):
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, Any]]  # Additional keywords from text analysis

class MarketResponse(BaseModel):
    ticker: str
    event_ticker: str
    title: str
    subtitle: Optional[str] = None
    similarity_score: float
    market_type: str
    status: str
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    no_bid: Optional[int] = None
    no_ask: Optional[int] = None
    last_price: Optional[int] = None
    previous_price: Optional[int] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None

class SimilarityResponse(BaseModel):
    query: str
    results: List[MarketResponse]
    total_markets_searched: int

# Initialize the models
def initialize_models():
    global similarity_model, keyword_model, ner_pipeline
    
    if similarity_model is None:
        logger.info("Loading high-end similarity model (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)...")
        # Use the best multilingual model for semantic similarity
        similarity_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        logger.info("High-end similarity model loaded successfully")
    
    if keyword_model is None:
        logger.info("Loading keyword scoring model (sentence-transformers/all-mpnet-base-v2)...")
        # Use high-quality model for keyword importance scoring
        keyword_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        logger.info("Keyword scoring model loaded successfully")
    
    if ner_pipeline is None:
        logger.info("Loading state-of-the-art NER pipeline...")
        # Use the best available NER model
        try:
            logger.info("Attempting to load spaCy transformer model (en_core_web_trf)...")
            ner_pipeline = pipeline("ner", 
                                   model="Jean-Baptiste/roberta-large-ner-english",  # State-of-the-art NER
                                   tokenizer="Jean-Baptiste/roberta-large-ner-english",
                                   aggregation_strategy="simple",
                                   device=0 if torch.cuda.is_available() else -1)
            logger.info("RoBERTa-large NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load RoBERTa-large NER, falling back to BERT-large: {e}")
            try:
                ner_pipeline = pipeline("ner", 
                                       model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                       tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
                                       aggregation_strategy="simple",
                                       device=0 if torch.cuda.is_available() else -1)
                logger.info("BERT-large NER model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load BERT-large, using base model: {e}")
                ner_pipeline = pipeline("ner", 
                                       model="dbmdz/bert-base-cased-finetuned-conll03-english",
                                       aggregation_strategy="simple")
        logger.info("NER pipeline loaded successfully")

# Fetch all Kalshi markets with pagination
async def fetch_all_kalshi_markets():
    """Fetch all markets from Kalshi API with pagination"""
    logger.info("Fetching all Kalshi markets...")
    
    all_markets = []
    cursor = None
    base_url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                params = {
                    "limit": 1000,
                    "status": "open"
                }
                if cursor:
                    params["cursor"] = cursor
                
                response = await client.get(base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                markets = data.get("markets", [])
                all_markets.extend(markets)
                
                logger.info(f"Fetched {len(markets)} markets, total so far: {len(all_markets)}")
                
                # Check if there's more data
                cursor = data.get("cursor")
                if not cursor:
                    break
                    
            except httpx.HTTPError as e:
                logger.error(f"HTTP error fetching markets: {e}")
                break
            except Exception as e:
                logger.error(f"Error fetching markets: {e}")
                break
    
    logger.info(f"Total markets fetched: {len(all_markets)}")
    return all_markets

# Deduplicate markets by event_ticker
def deduplicate_markets(markets):
    """Keep only the first occurrence of each event_ticker"""
    seen_events = set()
    unique_markets = []
    
    for market in markets:
        event_ticker = market.get("event_ticker")
        if event_ticker and event_ticker not in seen_events:
            unique_markets.append(market)
            seen_events.add(event_ticker)
    
    logger.info(f"Deduplicated {len(markets)} markets to {len(unique_markets)} unique events")
    return unique_markets

# Advanced keyword extraction using NER and semantic analysis
def extract_keywords_advanced(text, max_keywords=15):
    """Extract keywords using NER entities and semantic importance"""
    
    # Initialize models
    initialize_models()
    
    # Clean text
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Extract named entities (always show these)
    entities = []
    try:
        # Process text in chunks if too long
        text_chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        
        for chunk in text_chunks:
            ner_results = ner_pipeline(chunk)
            
            for entity in ner_results:
                # Lower confidence threshold to show more entities
                if entity['score'] > 0.7:  
                    entity_text = entity['word'].strip()
                    if len(entity_text) > 1:  # Allow shorter entities
                        entities.append({
                            'term': entity_text,
                            'type': entity['entity_group'],
                            'score': float(entity['score']),
                            'rank': len(entities) + 1
                        })
    except Exception as e:
        logger.warning(f"NER extraction failed: {e}")
    
    # Extract additional semantic keywords using TF-IDF for context
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'an', 'a', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
        'must', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our',
        'your', 'my', 'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'him', 'them',
        'us', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'than', 'then',
        'now', 'here', 'there', 'up', 'down', 'out', 'off', 'over', 'under', 'again'
    }
    
    # Simple keyword extraction for additional context
    words = cleaned_text.lower().split()
    words = [w for w in words if len(w) > 3 and w not in stop_words and w.isalpha()]
    
    # Get most frequent meaningful words
    word_freq = Counter(words)
    keywords = []
    
    for rank, (word, freq) in enumerate(word_freq.most_common(max_keywords), 1):
        # Skip if already found as entity
        if not any(word.lower() in entity['term'].lower() for entity in entities):
            keywords.append({
                'term': word,
                'score': float(freq),
                'rank': rank,
                'type': 'keyword'
            })
    
    # Remove duplicates and limit entities
    seen_entities = set()
    unique_entities = []
    
    for entity in entities:
        entity_lower = entity['term'].lower()
        if entity_lower not in seen_entities:
            unique_entities.append(entity)
            seen_entities.add(entity_lower)
    
    return {
        'entities': unique_entities[:max_keywords],
        'keywords': keywords[:max_keywords]
    }

# Load cache from disk
def load_cache_from_disk():
    """Load markets cache from disk if it exists and is fresh"""
    global markets_cache, embeddings_cache, last_cache_update
    
    if not os.path.exists(CACHE_FILE):
        logger.info("No cache file found")
        return False
    
    try:
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check if cache is still valid
        import time
        current_time = time.time()
        cache_age = current_time - cache_data['timestamp']
        
        if cache_age < CACHE_DURATION:
            markets_cache = cache_data['markets']
            embeddings_cache = cache_data['embeddings']
            last_cache_update = cache_data['timestamp']
            
            cache_age_minutes = cache_age / 60
            logger.info(f"Loaded fresh cache from disk ({cache_age_minutes:.1f} minutes old, {len(markets_cache)} markets)")
            return True
        else:
            logger.info(f"Cache file is too old ({cache_age/60:.1f} minutes), will refresh")
            return False
            
    except Exception as e:
        logger.warning(f"Failed to load cache from disk: {e}")
        return False

# Save cache to disk
def save_cache_to_disk():
    """Save current markets cache to disk"""
    try:
        cache_data = {
            'markets': markets_cache,
            'embeddings': embeddings_cache,
            'timestamp': last_cache_update
        }
        
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Cache saved to disk ({len(markets_cache)} markets)")
        
    except Exception as e:
        logger.warning(f"Failed to save cache to disk: {e}")

# Update markets cache and embeddings
async def update_markets_cache():
    global markets_cache, embeddings_cache, last_cache_update
    
    import time
    current_time = time.time()
    
    # Check if cache is still fresh (1 hour)
    if current_time - last_cache_update < CACHE_DURATION and markets_cache:
        cache_age_minutes = (current_time - last_cache_update) / 60
        logger.info(f"Using cached markets data (cached {cache_age_minutes:.1f} minutes ago)")
        return
    
    # Try to load from disk first if memory cache is empty
    if not markets_cache:
        if load_cache_from_disk():
            return
    
    logger.info("Updating markets cache...")
    
    # Fetch fresh markets
    all_markets = await fetch_all_kalshi_markets()
    
    if not all_markets:
        logger.warning("No markets fetched")
        return
    
    # Deduplicate by event_ticker
    unique_markets = deduplicate_markets(all_markets)
    
    # Prepare text for embeddings
    market_texts = []
    for market in unique_markets:
        title = market.get("title", "")
        subtitle = market.get("subtitle", "")
        yes_sub_title = market.get("yes_sub_title", "")
        no_sub_title = market.get("no_sub_title", "")
        
        # Combine all text fields
        combined_text = f"{title} {subtitle} {yes_sub_title} {no_sub_title}".strip()
        market_texts.append(combined_text)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    initialize_models()
    embeddings = similarity_model.encode(market_texts, convert_to_tensor=False)
    
    # Update cache
    markets_cache = unique_markets
    embeddings_cache = embeddings
    last_cache_update = current_time
    
    # Save cache to disk
    save_cache_to_disk()
    
    logger.info(f"Cache updated with {len(unique_markets)} unique markets. Next update in {CACHE_DURATION/60:.0f} minutes.")

@app.on_event("startup")
async def startup_event():
    """Initialize models and cache on startup"""
    logger.info("Starting up FastAPI server...")
    initialize_models()
    
    # Try to load cache from disk first, then fetch if needed
    logger.info("Loading markets cache...")
    if not load_cache_from_disk():
        logger.info("No valid cache found, fetching fresh data...")
        await update_markets_cache()
    else:
        logger.info("Using cached markets data from disk")

@app.get("/")
async def root():
    return {"message": "Kalshi Market Similarity Search API", "status": "running"}

@app.get("/health")
async def health_check():
    import time
    current_time = time.time()
    cache_age_minutes = (current_time - last_cache_update) / 60 if last_cache_update > 0 else 0
    minutes_until_refresh = max(0, (CACHE_DURATION - (current_time - last_cache_update)) / 60) if last_cache_update > 0 else 0
    
    return {
        "status": "healthy",
        "cached_markets": len(markets_cache),
        "cache_age_minutes": round(cache_age_minutes, 1),
        "minutes_until_refresh": round(minutes_until_refresh, 1),
        "similarity_model_loaded": similarity_model is not None,
        "keyword_model_loaded": keyword_model is not None,
        "ner_model_loaded": ner_pipeline is not None
    }

@app.post("/similarity", response_model=SimilarityResponse)
async def similarity_search(request: SimilarityRequest):
    """Find similar Kalshi markets based on semantic similarity"""
    
    try:
        # Update cache if needed
        await update_markets_cache()
        
        if not markets_cache or len(embeddings_cache) == 0:
            raise HTTPException(status_code=503, detail="Markets data not available")
        
        # Generate embedding for the query
        initialize_models()
        query_embedding = similarity_model.encode([request.query], convert_to_tensor=False)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, embeddings_cache)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:request.max_results]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarities
                market = markets_cache[idx]
                
                result = MarketResponse(
                    ticker=market.get("ticker", ""),
                    event_ticker=market.get("event_ticker", ""),
                    title=market.get("title", ""),
                    subtitle=market.get("subtitle"),
                    similarity_score=float(similarities[idx]),
                    market_type=market.get("market_type", ""),
                    status=market.get("status", ""),
                    yes_bid=market.get("yes_bid"),
                    yes_ask=market.get("yes_ask"),
                    no_bid=market.get("no_bid"),
                    no_ask=market.get("no_ask"),
                    last_price=market.get("last_price"),
                    volume=market.get("volume"),
                    open_interest=market.get("open_interest"),
                    previous_price=market.get("previous_price")
                )
                results.append(result)
        
        return SimilarityResponse(
            query=request.query,
            results=results,
            total_markets_searched=len(markets_cache)
        )
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/keywords", response_model=KeywordResponse)
async def extract_keywords(request: KeywordRequest):
    """Extract keywords from text using NER and advanced TF-IDF"""
    
    try:
        logger.info(f"Extracting keywords from text of length: {len(request.text)}")
        
        # Extract keywords using advanced method
        keywords = extract_keywords_advanced(request.text, request.max_keywords)
        
        return KeywordResponse(
            entities=keywords['entities'],
            keywords=keywords['keywords']
        )
        
    except Exception as e:
        logger.error(f"Error in keyword extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh-cache")
async def refresh_cache():
    """Manually refresh the markets cache"""
    global last_cache_update
    last_cache_update = 0  # Force cache refresh
    
    await update_markets_cache()
    
    return {
        "message": "Cache refreshed successfully",
        "total_markets": len(markets_cache)
    }

@app.post("/sentiment")
async def sentiment(request: Request):
    data = await request.json()
    text = data.get('text', '')
    logger.info(f"/sentiment called. Text length: {len(text)}")
    # Limit to first 3000 characters for now (simple implementation)
    limited_text = text[:3000]
    # TODO: Add chunking/aggregation for longer texts in the future
    try:
        pipe = get_sentiment_pipeline()
        result = pipe(limited_text)
        # result is a list of dicts, take the first
        sentiment_label = result[0]['label']
        sentiment_score = float(result[0]['score'])
        logger.info(f"Sentiment result: {sentiment_label} (score: {sentiment_score:.2f})")
        response = {
            'sentiment': sentiment_label,
            'score': sentiment_score,
            'input_text': limited_text
        }
    except Exception as e:
        logger.error(f"Error in /sentiment: {e}", exc_info=True)
        response = {
            'sentiment': 'neutral',
            'score': 0.5,
            'input_text': limited_text,
            'error': str(e)
        }
    return JSONResponse(content=response)

@app.post("/compare_sentiment")
async def compare_sentiment(request: Request):
    data = await request.json()
    text = data.get('text', '')
    market_yes_price = data.get('market_yes_price', None)
    market_no_price = data.get('market_no_price', None)
    logger.info(f"/compare_sentiment called. Market YES price: {market_yes_price}, Market NO price: {market_no_price}, Text length: {len(text)}")
    if market_yes_price is None or market_no_price is None:
        logger.warning("market_yes_price or market_no_price missing in request.")
        return JSONResponse(status_code=400, content={"error": "market_yes_price and market_no_price are required and should be floats between 0 and 1."})
    limited_text = text[:3000]
    try:
        pipe = get_sentiment_pipeline()
        result = pipe(limited_text)
        sentiment_label = result[0]['label']
        sentiment_score = float(result[0]['score'])
        article_optimism = get_optimism_score(sentiment_label, sentiment_score)
        market_optimism = float(market_yes_price)  # For now, use YES price as optimism
        delta = article_optimism - market_optimism
        logger.info(f"Article optimism: {article_optimism:.2f}, Market YES optimism: {market_optimism:.2f}, Market NO price: {market_no_price}, Delta: {delta:.2f}")
        response = {
            'article_sentiment_label': sentiment_label,
            'article_sentiment_score': sentiment_score,
            'article_optimism': article_optimism,
            'market_yes_price': float(market_yes_price),
            'market_no_price': float(market_no_price),
            'market_optimism': market_optimism,
            'delta': delta,
            'input_text': limited_text
        }
    except Exception as e:
        logger.error(f"Error in /compare_sentiment: {e}", exc_info=True)
        response = {
            'error': str(e),
            'article_optimism': None,
            'market_yes_price': market_yes_price,
            'market_no_price': market_no_price,
            'market_optimism': None,
            'delta': None,
            'input_text': limited_text
        }
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")