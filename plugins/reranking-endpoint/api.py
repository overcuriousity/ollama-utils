import asyncio
import httpx
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ollama Cross-Encoder Reranker API")

class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = 3

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None

class RerankResponse(BaseModel):
    results: List[RerankResult]

async def get_embedding(
    client: httpx.AsyncClient,
    model: str,
    text: str
) -> Optional[List[float]]:
    """Get embedding from Ollama."""
    url = "http://localhost:11434/api/embeddings"
    
    try:
        response = await client.post(
            url,
            json={"model": model, "prompt": text},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json().get("embedding")
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

async def score_document_cross_encoder_workaround(
    client: httpx.AsyncClient,
    model: str,
    query: str,
    doc: str,
    index: int
) -> dict:
    """
    Workaround for using cross-encoder reranker models with Ollama.
    
    Works with: BGE-reranker, Qwen3-Reranker, and other cross-encoder models.
    Based on: https://medium.com/@rosgluk/reranking-documents-with-ollama-and-qwen3-reranker-model-in-go-6dc9c2fb5f0b
    
    The Problem: Cross-encoder models have a classification head that outputs relevance scores.
    Ollama only exposes the embedding API, not the classification head.
    
    The Workaround: When using concatenated query+doc embeddings with cross-encoders,
    LOWER magnitude = MORE relevant. We invert the scores so that
    higher values = more relevant (standard convention).
    
    Steps:
    1. Concatenate query and document in cross-encoder format
    2. Get embedding of the concatenated text
    3. Calculate magnitude (lower = more relevant for cross-encoders)
    4. Invert and normalize to 0-1 (higher = more relevant)
    """
    
    # Format as cross-encoder input
    # The format matters - reranker models expect specific patterns
    combined = f"Query: {query}\n\nDocument: {doc}\n\nRelevance:"
    
    # Get embedding
    embedding = await get_embedding(client, model, combined)
    
    if embedding is None:
        logger.warning(f"Failed to get embedding for document {index}")
        return {
            "index": index,
            "relevance_score": 0.0,
            "document": doc
        }
    
    # Calculate magnitude (L2 norm) of the embedding vector
    vec = np.array(embedding)
    magnitude = float(np.linalg.norm(vec))
    
    # CRITICAL DISCOVERY: For cross-encoder rerankers via Ollama embeddings:
    # LOWER magnitude = MORE relevant document
    # Observed range: ~15-25 (lower = better)
    # This pattern applies to BGE, Qwen3-Reranker, and similar cross-encoder models
    
    # Invert and normalize to 0-1 where higher score = more relevant
    # Adjusted bounds based on empirical observations
    typical_good_magnitude = 15.0   # Highly relevant documents
    typical_poor_magnitude = 25.0   # Irrelevant documents
    
    # Linear interpolation (inverted)
    # magnitude 15 → score ~0.9
    # magnitude 25 → score ~0.0
    score = (typical_poor_magnitude - magnitude) / (typical_poor_magnitude - typical_good_magnitude)
    
    # Clamp to 0-1 range
    score = min(max(score, 0.0), 1.0)
    
    logger.debug(f"Doc {index}: magnitude={magnitude:.2f}, score={score:.4f}")
    logger.info(f"Raw magnitude: {magnitude:.2f}")

    return {
        "index": index,
        "relevance_score": score,
        "document": doc
    }

@app.on_event("startup")
async def check_ollama():
    """Verify Ollama is accessible on startup."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            response.raise_for_status()
            logger.info("✓ Successfully connected to Ollama")
            logger.warning("⚠️  Using workaround: Ollama doesn't expose cross-encoder classification heads")
            logger.warning("⚠️  Using concatenation + magnitude method instead")
            logger.info("💡 Works with: BGE-reranker, Qwen3-Reranker, etc.")
    except Exception as e:
        logger.error(f"✗ Cannot connect to Ollama: {e}")

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents using cross-encoder models via Ollama workaround.
    
    Supports: BGE-reranker, Qwen3-Reranker, and other cross-encoder models.
    
    NOTE: This uses a workaround (magnitude of concatenated embeddings)
    because Ollama doesn't expose the cross-encoder classification head.
    For best accuracy, use sentence-transformers or dedicated reranker APIs.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    logger.info(f"Reranking {len(request.documents)} documents (workaround method)")
    logger.info(f"Query: {request.query[:100]}...")
    
    async with httpx.AsyncClient() as client:
        # Score all documents concurrently
        tasks = [
            score_document_cross_encoder_workaround(
                client, request.model, request.query, doc, i
            )
            for i, doc in enumerate(request.documents)
        ]
        results = await asyncio.gather(*tasks)
        
        # Sort by score DESCENDING (higher score = more relevant)
        # Scores are now inverted, so higher = better
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Log scores
        top_scores = [f"{r['relevance_score']:.4f}" for r in results[:request.top_n]]
        logger.info(f"Top {len(top_scores)} scores: {top_scores}")
        
        return {"results": results[:request.top_n]}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ollama-cross-encoder-reranker",
        "supported_models": "BGE-reranker, Qwen3-Reranker, etc.",
        "method": "concatenation + magnitude workaround",
        "note": "Ollama doesn't expose classification heads - using embedding magnitude"
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("Ollama Cross-Encoder Reranker API")
    logger.info("=" * 60)
    logger.info("Supports: BGE-reranker, Qwen3-Reranker, etc.")
    logger.info("Method: Concatenation + magnitude workaround")
    logger.info("Why: Ollama doesn't expose cross-encoder classification heads")
    logger.info("Starting on: http://0.0.0.0:8080")
    logger.info("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
