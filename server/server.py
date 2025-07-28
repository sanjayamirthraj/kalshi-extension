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
from models import *

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

def get_optimism_score(label, score, market_sentiment_label, market_sentiment_score):
    # if market sentiment is positive then positive sentences are optimistic
    # if market sentiment is negative then negative sentences are optimistic
    if market_sentiment_label.upper() == 'POSITIVE':
        if label.upper() == 'POSITIVE':
            return score
        elif label.upper() == 'NEGATIVE':
            return 1 - score
    elif market_sentiment_label.upper() == 'NEGATIVE':
        if label.upper() == 'POSITIVE':
            return 1 - score
        elif label.upper() == 'NEGATIVE':
            return score
    else:
        return 0.5  # Neutral or unknown

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex"""
    # Use regex to split on sentence endings, but be more careful about abbreviations
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Only keep sentences that are substantial (more than 10 characters and have meaningful content)
        if len(sentence) > 10 and not sentence.isspace():
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def calculate_signal_score(sentence_analyses: List[SentenceAnalysis], market_sentiment_label: str, market_sentiment_score: float) -> float:
    """Calculate a signal score 0-100 based on similarity and sentiment of top sentences"""
    if not sentence_analyses:
        return 50.0  # Neutral if no sentences
    
    total_score = 0.0
    weight_sum = 0.0
    
    for analysis in sentence_analyses:
        # Weight by similarity (higher similarity = more important)
        similarity_weight = analysis.similarity_score
        
        # Convert sentiment to optimism score
        optimism = get_optimism_score(analysis.sentiment_label, analysis.sentiment_score, market_sentiment_label, market_sentiment_score)
        
        # Calculate weighted contribution
        # High similarity + positive sentiment = high score
        # Low similarity or negative sentiment = low score
        contribution = similarity_weight * optimism * 100
        
        total_score += contribution * similarity_weight
        weight_sum += similarity_weight
    
    # Calculate weighted average
    if weight_sum > 0:
        final_score = total_score / weight_sum
    else:
        final_score = 50.0
    
    # Ensure score is between 0 and 100
    return max(0.0, min(100.0, final_score))

def get_recommendation_from_score(score: float) -> str:
    """Convert score to BUY/NEUTRAL/SELL recommendation"""
    if score >= 70:
        return "BUY"
    elif score <= 30:
        return "SELL"
    else:
        return "NEUTRAL"

def truncate_text_for_model(text: str, max_chars: int = 2000) -> str:
    """
    Truncate text to a safe length for transformer models.
    Using character-based truncation as a proxy for token limits.
    Most models have ~512 token limit, and roughly 4 chars per token,
    so 2000 chars should be well within limits.
    """
    if len(text) <= max_chars:
        return text
    
    # Truncate and try to end at a sentence boundary
    truncated = text[:max_chars]
    
    # Find the last sentence ending
    last_period = truncated.rfind('.')
    last_exclamation = truncated.rfind('!')
    last_question = truncated.rfind('?')
    
    last_sentence_end = max(last_period, last_exclamation, last_question)
    
    # If we found a sentence ending and it's not too early, use it
    if last_sentence_end > max_chars * 0.7:  # At least 70% of the text
        return truncated[:last_sentence_end + 1]
    else:
        # Otherwise just truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # At least 80% of the text
            return truncated[:last_space]
        else:
            return truncated

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

@app.post("/get_signal", response_model=GetSignalResponse)
async def get_signal(request: GetSignalRequest):
    """Analyze article text against market text to provide trading signal"""
    
    try:
        logger.info(f"Getting signal for article length: {len(request.article_text)}, market text length: {len(request.market_text)}")
        
        # Initialize models
        initialize_models()
        print("request.article_text: ", request.article_text)
        # Step 1: Split article into sentences
        sentences = split_into_sentences(request.article_text)
        logger.info(f"Split article into {len(sentences)} sentences")
        
        if not sentences:
            logger.warning("No sentences found in article")
            return GetSignalResponse(
                recommendation="NEUTRAL",
                score=50.0,
                top_sentences=[],
                market_text=request.market_text
            )
        
        # Step 2: Embed each sentence (truncate each sentence to avoid token limit)
        truncated_sentences = [truncate_text_for_model(sentence) for sentence in sentences]
        
        # Log if any sentences were truncated
        truncated_count = sum(1 for i, sentence in enumerate(sentences) if len(sentence) != len(truncated_sentences[i]))
        if truncated_count > 0:
            logger.info(f"Truncated {truncated_count} out of {len(sentences)} sentences to fit token limit")
        
        sentence_embeddings = similarity_model.encode(truncated_sentences, convert_to_tensor=False)
        
        # Step 3: Embed market text (truncate to avoid token limit)
        truncated_market_text = truncate_text_for_model(request.market_text)
        if len(request.market_text) != len(truncated_market_text):
            logger.info(f"Truncated market text from {len(request.market_text)} to {len(truncated_market_text)} characters")
        
        market_embedding = similarity_model.encode([truncated_market_text], convert_to_tensor=False)
        
        # Step 4: Calculate similarities between each sentence and market text
        similarities = cosine_similarity(sentence_embeddings, market_embedding).flatten()
        
        # Step 5: Get top 3 sentences by similarity
        top_indices = np.argsort(similarities)[::-1][:3]
        
        # Step 6: Analyze sentiment of top sentences
        sentiment_pipe = get_sentiment_pipeline()
        top_sentence_analyses = []
        
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarities
                sentence = sentences[idx]  # Original sentence for display
                truncated_sentence = truncated_sentences[idx]  # Truncated for processing
                similarity_score = float(similarities[idx])
                
                # Get sentiment (use truncated version to avoid token limit)
                sentiment_result = sentiment_pipe(truncated_sentence)
                sentiment_label = sentiment_result[0]['label']
                sentiment_score = float(sentiment_result[0]['score'])
                
                analysis = SentenceAnalysis(
                    sentence=sentence,  # Use original sentence for display
                    similarity_score=similarity_score,
                    sentiment_label=sentiment_label,
                    sentiment_score=sentiment_score
                )
                top_sentence_analyses.append(analysis)
                
                logger.info(f"Top sentence similarity: {similarity_score:.3f}, sentiment: {sentiment_label} ({sentiment_score:.3f})")
        # get sentiment of market text
        market_sentiment_result = sentiment_pipe(request.market_text)
        market_sentiment_label = market_sentiment_result[0]['label']
        market_sentiment_score = float(market_sentiment_result[0]['score'])
        # Step 7: Calculate signal score (0-100)
        signal_score = calculate_signal_score(top_sentence_analyses, market_sentiment_label, market_sentiment_score)
        
        # Step 8: Get recommendation based on score
        recommendation = get_recommendation_from_score(signal_score)
        
        logger.info(f"Signal analysis complete. Score: {signal_score:.1f}, Recommendation: {recommendation}")
        
        return GetSignalResponse(
            recommendation=recommendation,
            score=signal_score,
            top_sentences=top_sentence_analyses,
            market_text=request.market_text
        )
        
    except Exception as e:
        logger.error(f"Error in get_signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mock_get_signal", response_model=GetSignalResponse)
async def mock_get_signal(request: GetSignalRequest):
    """Mock version of get_signal endpoint for frontend development and testing"""
    
    logger.info(f"Mock signal analysis for article length: {len(request.article_text)}, market text length: {len(request.market_text)}")
    
    # Create mock sentence analyses with realistic data
    mock_sentences = [
        SentenceAnalysis(
            sentence="The company reported strong quarterly earnings that exceeded analyst expectations by 15%.",
            similarity_score=0.87,
            sentiment_label="POSITIVE",
            sentiment_score=0.92
        ),
        SentenceAnalysis(
            sentence="Market conditions remain favorable with increased consumer demand and positive economic indicators.",
            similarity_score=0.73,
            sentiment_label="POSITIVE",
            sentiment_score=0.85
        ),
        SentenceAnalysis(
            sentence="However, some analysts express concerns about potential regulatory challenges in the coming quarter.",
            similarity_score=0.61,
            sentiment_label="NEGATIVE",
            sentiment_score=0.78
        )
    ]
    
    # Mock score calculation (realistic based on the mock data above)
    mock_score = 74.5
    mock_recommendation = "BUY"
    
    return GetSignalResponse(
        recommendation=mock_recommendation,
        score=mock_score,
        top_sentences=mock_sentences,
        market_text=request.market_text
    )

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