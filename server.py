from fastapi import FastAPI, HTTPException
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
CACHE_DURATION = 300  # 5 minutes in seconds

# Pydantic models
class SimilarityRequest(BaseModel):
    query: str
    max_results: int = 10

class KeywordRequest(BaseModel):
    text: str
    max_keywords: int = 15

class KeywordResponse(BaseModel):
    one_word: List[Dict[str, Any]]
    two_word: List[Dict[str, Any]]
    three_word: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]

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

# Advanced keyword extraction using NER and TF-IDF
def extract_keywords_advanced(text, max_keywords=15):
    """Extract keywords using NER entities and TF-IDF scoring"""
    
    # Initialize models
    initialize_models()
    
    # Clean text
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Extract named entities
    entities = []
    try:
        ner_results = ner_pipeline(text[:512])  # Limit text length for NER
        
        for entity in ner_results:
            if entity['score'] > 0.8:  # High confidence only
                entity_text = entity['word'].lower().strip()
                if len(entity_text) > 2:
                    entities.append({
                        'term': entity_text,
                        'type': entity['entity_group'],
                        'score': float(entity['score']),
                        'rank': len(entities) + 1
                    })
    except Exception as e:
        logger.warning(f"NER extraction failed: {e}")
    
    # TF-IDF based keyword extraction
    stop_words = {
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
    }
    
    # Tokenize and clean
    words = text.lower().split()
    words = [w for w in words if len(w) > 2 and w not in stop_words and w.isalpha()]
    
    # Generate n-grams
    candidates = []
    
    # 1-grams
    word_freq = Counter(words)
    for word, freq in word_freq.items():
        candidates.append({
            'term': word,
            'freq': freq,
            'ngram_type': 1,
            'position_score': 1.0 - (text.lower().find(word) / len(text))
        })
    
    # 2-grams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_freq = Counter(bigrams)
    for bigram, freq in bigram_freq.items():
        candidates.append({
            'term': bigram,
            'freq': freq,
            'ngram_type': 2,
            'position_score': 1.0 - (text.lower().find(bigram) / len(text))
        })
    
    # 3-grams
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
    trigram_freq = Counter(trigrams)
    for trigram, freq in trigram_freq.items():
        candidates.append({
            'term': trigram,
            'freq': freq,
            'ngram_type': 3,
            'position_score': 1.0 - (text.lower().find(trigram) / len(text))
        })
    
    # Score candidates
    for candidate in candidates:
        # TF score with log scaling
        tf_score = np.log(candidate['freq'] + 1)
        
        # Position bonus (earlier = better)
        position_bonus = candidate['position_score']
        
        # N-gram bonus (longer phrases get slight bonus)
        ngram_bonus = 1.0 + (candidate['ngram_type'] - 1) * 0.1
        
        # Final score
        candidate['score'] = tf_score * (1 + position_bonus) * ngram_bonus
    
    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Categorize results
    one_word = []
    two_word = []
    three_word = []
    
    rank = 1
    for candidate in candidates:
        if candidate['ngram_type'] == 1 and len(one_word) < max_keywords:
            one_word.append({
                'term': candidate['term'],
                'score': float(candidate['score']),
                'rank': rank
            })
        elif candidate['ngram_type'] == 2 and len(two_word) < max_keywords:
            two_word.append({
                'term': candidate['term'],
                'score': float(candidate['score']),
                'rank': rank
            })
        elif candidate['ngram_type'] == 3 and len(three_word) < max_keywords:
            three_word.append({
                'term': candidate['term'],
                'score': float(candidate['score']),
                'rank': rank
            })
        rank += 1
    
    return {
        'one_word': one_word[:max_keywords],
        'two_word': two_word[:max_keywords],
        'three_word': three_word[:max_keywords],
        'entities': entities[:max_keywords]
    }

# Update markets cache and embeddings
async def update_markets_cache():
    global markets_cache, embeddings_cache, last_cache_update
    
    import time
    current_time = time.time()
    
    # Check if cache is still fresh
    if current_time - last_cache_update < CACHE_DURATION and markets_cache:
        logger.info("Using cached markets data")
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
    
    logger.info(f"Cache updated with {len(unique_markets)} unique markets")

@app.on_event("startup")
async def startup_event():
    """Initialize models and cache on startup"""
    logger.info("Starting up FastAPI server...")
    initialize_models()
    await update_markets_cache()

@app.get("/")
async def root():
    return {"message": "Kalshi Market Similarity Search API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cached_markets": len(markets_cache),
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
                    open_interest=market.get("open_interest")
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
            one_word=keywords['one_word'],
            two_word=keywords['two_word'],
            three_word=keywords['three_word'],
            entities=keywords['entities']
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")