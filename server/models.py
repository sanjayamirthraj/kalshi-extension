# Pydantic models
from typing import List
from pydantic import BaseModel
from typing import Dict, Any, Optional

class SimilarityRequest(BaseModel):
    query: str
    max_results: int = 10

class KeywordRequest(BaseModel):
    text: str
    max_keywords: int = 15

class KeywordResponse(BaseModel):
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, Any]]  # Additional keywords from text analysis

class GetSignalRequest(BaseModel):
    article_text: str
    market_text: str

class SentenceAnalysis(BaseModel):
    sentence: str
    similarity_score: float
    sentiment_label: str
    sentiment_score: float

class GetSignalResponse(BaseModel):
    recommendation: str  # BUY, SELL, NEUTRAL
    score: float  # 0-100
    top_sentences: List[SentenceAnalysis]
    market_text: str

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