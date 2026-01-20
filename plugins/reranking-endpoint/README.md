# Ollama Cross-Encoder Reranker Workaround

> **⚠️ Important:** This is a **workaround/hack**, not a proper solution. It exploits an undocumented behavior of embedding magnitudes and should be used with caution.

A FastAPI service that provides document reranking using Ollama's embedding endpoint. This exists because Ollama does not natively support a `/api/rerank` endpoint for cross-encoder reranker models.

## The Problem

Cross-encoder reranker models (like BGE-reranker, Qwen3-Reranker, etc.) are designed to score query-document pairs for relevance. However:

- **Ollama has no `/api/rerank` endpoint** - reranker models can't be used as intended
- **`/api/embeddings`** - returns embeddings, not the classification head scores
- **`/api/generate`** - reranker models can't generate text (they output uniform scores like 0.5)

**Root Cause:** Cross-encoder models have a classification head that outputs relevance scores. Ollama only exposes the embedding layer, not the classification layer.

## The Workaround

This service uses a magnitude-based approach:

1. Concatenates query and document in cross-encoder format: `"Query: {query}\n\nDocument: {doc}\n\nRelevance:"`
2. Gets embedding vector from Ollama's `/api/embeddings` endpoint
3. Calculates the L2 norm (magnitude) of the embedding vector
4. **Key discovery:** For cross-encoder models, **lower magnitude = more relevant**
5. Inverts and normalizes to 0-1 range where higher score = more relevant

### Why This Works (Sort Of)

When a cross-encoder model processes a query-document pair through the embedding endpoint, the embedding's magnitude appears to correlate inversely with relevance. This pattern has been observed in:
- BGE-reranker models (BGE-reranker-v2-m3, etc.)
- Qwen3-Reranker models (Qwen3-Reranker-4B, etc.)
- Potentially other cross-encoder architectures

**However, this is:**
- **Not documented behavior** - exploiting accidental correlation
- **Not guaranteed across all models** - each model may have different magnitude ranges
- **Not the intended use** - bypasses the classification head
- **Less accurate** - proper cross-encoder scoring would be significantly better

But it's currently the only way to use cross-encoder reranker models with Ollama.

## Limitations

### ⚠️ Critical Limitations

1. **Bypasses Classification Head**
   - Cross-encoder models have a specialized classification layer for scoring
   - Ollama only exposes the embedding layer, not the classification head
   - We're using embedding magnitudes as a proxy, not the actual relevance scores
   - **This is fundamentally wrong** - we're using the wrong layer of the model

2. **Model-Specific Behavior**
   - Magnitude ranges differ between models:
     - BGE-reranker-v2-m3: ~15-28 (lower = more relevant)
     - Qwen3-Reranker: similar pattern observed
     - Other models: unknown, requires testing
   - Correlation direction may theoretically vary (though inverse correlation seems common)
   - Requires manual calibration per model family

3. **No Theoretical Foundation**
   - Exploits accidental correlation, not designed functionality
   - No documentation or guarantees from model creators
   - Could break with model updates or quantization changes
   - No mathematical proof this approach is valid

4. **Significantly Less Accurate**
   - Proper cross-encoder classification head scoring would be far more accurate
   - sentence-transformers library uses the models correctly (30-50% better accuracy expected)
   - This workaround is a compromise for Ollama's GPU scheduling benefits
   - **Not suitable for production** without extensive validation

5. **Embedding Dimension Dependency**
   - Magnitude scales with dimensionality (384 vs 768 vs 1024)
   - Models with different dimensions need different calibration
   - Quantization (Q4 vs Q5 vs Q8) may affect magnitude distributions

6. **Performance Overhead**
   - Requires one API call per document (40 docs = 40 calls)
   - Slower than native reranking API would be
   - Concurrent processing helps but still suboptimal
   - No batching support in Ollama's embedding API

## When To Use This

✅ **Use if:**
- You need Ollama's GPU scheduling for multiple models
- VRAM is constrained and you can't run separate services
- You're okay with **significantly reduced accuracy** vs proper cross-encoder usage
- You can tolerate model-specific calibration and testing
- You understand you're using the **wrong layer** of the model
- This is for experimentation, not production

❌ **Don't use if:**
- You need reliable, production-grade reranking
- You need cross-model consistency
- You have VRAM for sentence-transformers (~200MB for reranker only)
- Accuracy is critical for your use case
- You need guaranteed correctness
- You're deploying to production without extensive validation

### Recommended Alternative

For production use, run sentence-transformers separately:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('BAAI/bge-reranker-v2-m3')
scores = model.predict([(query, doc) for doc in documents])
```
This uses the classification head correctly and provides proper relevance scores.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ollama-reranker-workaround.git
cd ollama-reranker-workaround

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with a cross-encoder reranker model
# Examples:
ollama pull qllama/bge-reranker-v2-m3
# or
ollama pull dengcao/qwen3-reranker-4b
```

## Usage

### Start the Service

```bash
python api.py
```

The service runs on `http://0.0.0.0:8080`

### API Request

```bash
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qllama/bge-reranker-v2-m3:latest",
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of artificial intelligence.",
      "The weather today is sunny.",
      "Neural networks are used in deep learning."
    ],
    "top_n": 2
  }'
```

### Response

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9234,
      "document": "Machine learning is a subset of artificial intelligence."
    },
    {
      "index": 2,
      "relevance_score": 0.7845,
      "document": "Neural networks are used in deep learning."
    }
  ]
}
```

## Configuration & Tunables

### Model Calibration

The most critical parameters are in `score_document_cross_encoder_workaround()`:

```python
# Magnitude bounds (model-specific!)
typical_good_magnitude = 15.0   # Highly relevant documents
typical_poor_magnitude = 25.0   # Irrelevant documents

# For cross-encoder models (BGE, Qwen3-Reranker):
# Observed range: ~15-28
# Lower magnitude = more relevant (inverse correlation)
# MUST be calibrated per model family!
```

### How to Calibrate for a New Model

1. **Enable magnitude logging:**
   ```python
   logger.info(f"Raw magnitude: {magnitude:.2f}")
   ```

2. **Test with known relevant/irrelevant documents:**
   ```python
   # Send queries with obviously relevant and irrelevant docs
   # Observe magnitude ranges in logs
   ```

3. **Determine correlation direction:**
   - If relevant docs have **lower** magnitudes → set `invert = True`
   - If relevant docs have **higher** magnitudes → set `invert = False`

4. **Set bounds:**
   ```python
   # Find 90th percentile of relevant doc magnitudes
   typical_good_magnitude = <observed_value>
   
   # Find 10th percentile of irrelevant doc magnitudes  
   typical_poor_magnitude = <observed_value>
   ```

### Prompt Format Tuning

The concatenation format may affect results:

```python
# Current format (works for BGE-reranker-v2-m3)
combined = f"Query: {query}\n\nDocument: {doc}\n\nRelevance:"

# Alternative formats to try:
combined = f"{query} [SEP] {doc}"
combined = f"query: {query} document: {doc}"
combined = f"<query>{query}</query><document>{doc}</document>"
```

Test different formats and check if score distributions improve.

### Concurrency Settings

```python
# In the rerank() endpoint
# Process all documents concurrently (default)
tasks = [score_document(...) for doc in documents]
results = await asyncio.gather(*tasks)

# Or batch for rate limiting:
batch_size = 10
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    # process batch...
```

## Technical Details

### Magnitude Calculation

```python
import numpy as np

# Get embedding from Ollama
embedding = await get_embedding(client, model, combined_text)

# Calculate L2 norm (Euclidean length)
vec = np.array(embedding)
magnitude = float(np.linalg.norm(vec))
# magnitude = sqrt(sum(x_i^2 for all dimensions))
```

### Score Normalization

```python
# Linear interpolation (inverted for BGE-reranker-v2-m3)
score = (typical_poor_magnitude - magnitude) / (typical_poor_magnitude - typical_good_magnitude)

# Clamp to [0, 1]
score = min(max(score, 0.0), 1.0)
```

### Example Magnitude Distributions

From real queries to **BGE-reranker-v2-m3** (your results may vary with other models):

```
Query: "Was ist eine Catalog Node ID?"

Highly relevant docs: magnitude ~15.30 - 15.98 → score 0.95-0.97
Moderately relevant:   magnitude ~17.00 - 19.00 → score 0.70-0.85
Weakly relevant:       magnitude ~20.00 - 24.00 → score 0.20-0.50
Irrelevant:           magnitude ~25.00 - 28.00 → score 0.00-0.10
```

**Note:** Qwen3-Reranker and other cross-encoder models will have different ranges. Always calibrate!

## Alternatives

### 1. Use sentence-transformers (Recommended for Production)

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')
scores = model.predict([(query, doc) for doc in documents])
```

**Pros:** Accurate, reliable, proper implementation  
**Cons:** ~200MB VRAM/RAM, separate from Ollama

### 2. Request Ollama Feature

Open an issue on [Ollama's GitHub](https://github.com/ollama/ollama) requesting native `/api/rerank` support.

### 3. Use API Services

Services like Cohere, Jina AI, or Voyage AI offer reranking APIs.

## Requirements

```
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
pydantic>=2.0.0
numpy>=1.24.0
```

## Contributing

This is a workaround for a missing feature. Contributions welcome for:
- Calibration configs for additional models
- Auto-calibration logic
- Alternative prompt formats
- Better normalization strategies

But remember: **The best contribution would be native Ollama support.**

## License

MIT

## Disclaimer

This is an **experimental workaround** that exploits undocumented behavior and **uses the wrong layer of cross-encoder models**. It is:

- **Using embeddings instead of classification scores** - fundamentally incorrect approach
- Not endorsed by Ollama, BAAI, Alibaba (Qwen), or any model creator
- Not guaranteed to work across models, versions, or quantization levels
- Not suitable for production use without extensive testing and validation
- A temporary hack until Ollama adds native `/api/rerank` support
- Significantly less accurate than proper cross-encoder usage

**Use at your own risk and always validate results against ground truth.**

For production systems, use sentence-transformers or dedicated reranking APIs that access the classification head properly.
