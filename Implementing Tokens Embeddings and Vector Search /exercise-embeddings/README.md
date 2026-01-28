# Lesson 7: Embeddings Implementation

## Overview

In this lesson, you'll build a semantic search system for product reviews and customer FAQs using embeddings. You'll learn how to convert text into vectors, calculate similarity, and find semantically related content.

## Learning Objectives

By completing this exercise, you will be able to:

- Generate embeddings using OpenAI's API
- Store embeddings with metadata for search
- Calculate cosine similarity between vectors
- Perform semantic search to find similar reviews
- Cluster feedback to identify common themes
- Build practical customer service applications with embeddings

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (Vocareum key provided in course)
- Basic understanding of vectors and similarity concepts
- Completion of Lessons 5-6 recommended

## Setup

### Install Dependencies

```bash
# Required
pip install openai numpy

# Optional (for advanced clustering)
pip install scikit-learn
```

### Set Your API Key

```bash
export OPENAI_API_KEY="your-key-here"
```

## Exercise Structure

### Starter Code
`exercises/starter/review_embedding_system.py`

Implement the `ReviewEmbeddingSystem` class with functions for:
- Creating embeddings
- Storing embeddings with metadata
- Calculating similarity
- Finding similar reviews
- Clustering feedback

### Solution Code
`exercises/solution/review_embedding_system.py`

Complete implementation with 5 demonstration functions and sample dataset.

## Running the Exercise

```bash
python exercises/solution/review_embedding_system.py

# Includes 5 demos:
# 1. Create and view embeddings
# 2. Semantic search for reviews
# 3. Similarity calculations
# 4. Clustering by theme
# 5. Practical use cases
```

## Key Concepts

### What Are Embeddings?

Embeddings are numerical representations of text as vectors (lists of numbers). Similar meanings result in similar vectors.

**Example:**
- "Great product!" → [0.23, -0.45, 0.12, ..., 0.67] (1536 numbers)
- "Excellent item!" → [0.25, -0.43, 0.14, ..., 0.65] (similar values!)
- "Terrible quality" → [-0.31, 0.52, -0.18, ..., -0.71] (different!)

### Why Use Embeddings?

**Traditional keyword search:**
- Query: "laptop freezes"
- Matches: Documents containing exactly "laptop" AND "freezes"
- Misses: "computer crashes", "PC hangs", "system unresponsive"

**Semantic search with embeddings:**
- Query: "laptop freezes" → vector
- Finds: Similar vectors (same meaning, different words!)
- Matches: "computer crashes", "PC hangs", "system unresponsive"

### OpenAI's Embedding Models

| Model | Dimensions | Cost per 1K tokens | Use Case |
|-------|------------|-------------------|----------|
| text-embedding-3-small | 1536 | $0.00002 | General purpose, cost-effective |
| text-embedding-3-large | 3072 | $0.00013 | Higher quality, more expensive |
| text-embedding-ada-002 | 1536 | $0.0001 | Older model (deprecated) |

**Recommendation**: Use `text-embedding-3-small` for most applications - it's 5x cheaper than ada-002 with better performance!

## Implementation Guide

### 1. Creating Embeddings

```python
def create_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Generate embedding vector for text."""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding  # List of 1536 floats
```

**Best practices:**
- Clean text before embedding (remove excessive whitespace, etc.)
- Keep texts under 8,191 tokens
- Batch multiple texts in one API call to save time
- Cache embeddings (they don't change for the same text)

### 2. Calculating Similarity

**Cosine Similarity** measures the angle between two vectors:
- **1.0**: Identical meaning
- **0.8-0.9**: Very similar
- **0.6-0.8**: Somewhat related
- **< 0.6**: Not very related

```python
import numpy as np

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    return dot_product / (magnitude1 * magnitude2)
```

### 3. Semantic Search

```python
def find_similar_reviews(query: str, top_k: int = 5) -> List[dict]:
    """Find reviews most similar to query."""
    # 1. Embed the query
    query_embedding = create_embedding(query)

    # 2. Calculate similarity to all reviews
    similarities = []
    for review in self.reviews:
        similarity = calculate_similarity(query_embedding, review['embedding'])
        similarities.append((review, similarity))

    # 3. Sort by similarity and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
```

### 4. Clustering Feedback

Group similar reviews to identify common themes:

```python
from sklearn.cluster import KMeans

def cluster_feedback(feedback_list: List[str], num_clusters: int = 3):
    """Cluster feedback into themes."""
    # 1. Create embeddings for all feedback
    embeddings = [create_embedding(text) for text in feedback_list]

    # 2. Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # 3. Group feedback by cluster
    return group_by_cluster(feedback_list, clusters)
```

## Practical Applications

### 1. Customer Support Ticket Routing

**Scenario**: Automatically route support tickets to the right team

```python
# Define team specialties
teams = {
    "billing": "payment issues, refunds, invoices, charges",
    "technical": "bugs, errors, crashes, performance",
    "shipping": "delivery, tracking, lost packages"
}

# Find best team for ticket
ticket = "I was charged twice for my order"
ticket_embedding = create_embedding(ticket)

best_team = None
best_score = 0
for team, specialty in teams.items():
    specialty_embedding = create_embedding(specialty)
    score = calculate_similarity(ticket_embedding, specialty_embedding)
    if score > best_score:
        best_score = score
        best_team = team

# Result: "billing" team
```

### 2. FAQ Matching

**Scenario**: Find relevant FAQs for customer questions

```python
faqs = [
    {"q": "How do I return an item?", "a": "Visit returns page..."},
    {"q": "Where is my order?", "a": "Check tracking page..."},
    # ... more FAQs
]

# Customer asks in different words
customer_question = "Can I send back a product I don't want?"

# Find best matching FAQ
matches = find_similar_questions(customer_question, faqs)
# Returns: "How do I return an item?" (high similarity!)
```

### 3. Review Analysis

**Scenario**: Identify common complaints in product reviews

```python
# Cluster 100 negative reviews
negative_reviews = get_reviews_with_rating_below(3)
clusters = cluster_feedback(negative_reviews, num_clusters=5)

# Cluster themes might be:
# - Cluster 0: Shipping/delivery issues
# - Cluster 1: Product quality problems
# - Cluster 2: Size/fit complaints
# - Cluster 3: Price concerns
# - Cluster 4: Customer service issues
```

### 4. Duplicate Detection

**Scenario**: Find duplicate or near-duplicate tickets

```python
def find_duplicates(new_ticket: str, existing_tickets: List[str], threshold: float = 0.9):
    """Find existing tickets very similar to new one."""
    new_embedding = create_embedding(new_ticket)

    duplicates = []
    for ticket in existing_tickets:
        similarity = calculate_similarity(new_embedding, ticket['embedding'])
        if similarity > threshold:  # Very similar!
            duplicates.append(ticket)

    return duplicates
```

## Sample Dataset

The exercise includes 10 diverse product reviews:

```python
{
    "review_id": "R001",
    "product": "Wireless Headphones",
    "rating": 5,
    "text": "Excellent sound quality! Best headphones I've ever owned.",
    "date": "2024-01-15"
},
# ... 9 more reviews across different products and sentiments
```

## Cost Considerations

**Embedding costs are very low:**
- text-embedding-3-small: $0.00002 per 1K tokens
- A typical review (50 words): ~$0.000001
- 10,000 reviews: ~$0.01

**One-time cost:**
- Generate embeddings once, reuse forever
- Store embeddings in database or file
- Only embed new content

**Comparison to LLM calls:**
- Embedding 1,000 reviews: $0.001
- Processing same reviews with GPT-3.5: $0.50+
- **500x cheaper!**

## Optimization Strategies

### 1. Batch Processing

```python
# Slow: One API call per review
for review in reviews:
    embedding = create_embedding(review['text'])

# Fast: Batch API call
texts = [review['text'] for review in reviews]
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts  # List of texts!
)
embeddings = [data.embedding for data in response.data]
```

**Benefits:**
- Faster processing
- Fewer API calls
- Same cost

### 2. Caching

```python
import json

# Save embeddings to file
def save_embeddings(reviews, filename="embeddings.json"):
    with open(filename, 'w') as f:
        json.dump(reviews, f)

# Load embeddings from file
def load_embeddings(filename="embeddings.json"):
    with open(filename, 'r') as f:
        return json.load(f)

# Use cached embeddings (free!)
if os.path.exists("embeddings.json"):
    reviews = load_embeddings()
else:
    reviews = create_and_embed_reviews()
    save_embeddings(reviews)
```

### 3. Dimensionality Reduction (Advanced)

For very large datasets, reduce embedding dimensions:

```python
# OpenAI embeddings support truncation
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
    dimensions=512  # Instead of 1536
)

# Smaller vectors:
# - 3x less storage
# - 3x faster similarity calculations
# - Minimal quality loss for most applications
```

## Demonstrations Included

### Demo 1: Creating Embeddings
- Generate embeddings for sample reviews
- View embedding vectors
- Understand embedding properties

### Demo 2: Semantic Search
- Search for similar reviews
- See similarity scores
- Compare to keyword search

### Demo 3: Similarity Calculations
- Calculate cosine similarity
- Find most/least similar pairs
- Understand similarity thresholds

### Demo 4: Clustering
- Group reviews by theme
- Identify common topics
- Generate cluster summaries

### Demo 5: Practical Applications
- FAQ matching
- Duplicate detection
- Ticket routing
- Review analysis

## Common Issues

### Memory Error with Large Datasets
```
MemoryError: Unable to allocate array
```
**Solution**: Process embeddings in batches, don't load all into memory at once

### Low Similarity Scores
All similarities are below 0.5
**Cause**: Text might be too short or too generic
**Solution**: Use more descriptive text, combine related fields

### Slow Search
Searching thousands of embeddings is slow
**Solution**: Use vector databases (covered later) like ChromaDB, Pinecone, or Weaviate

## Extension Ideas

1. **Multi-field Embeddings**: Combine product name, description, and reviews
2. **Weighted Search**: Boost recent reviews or verified purchases
3. **Hybrid Search**: Combine keyword search with semantic search
4. **Recommendation Engine**: "Customers who liked this also liked..."
5. **Sentiment-aware Search**: Filter by positive/negative before searching
6. **Multi-language Support**: Embeddings work across languages!

## Key Takeaways

✅ Embeddings capture semantic meaning, not just keywords
✅ Very cost-effective ($0.00002 per 1K tokens)
✅ Enable powerful applications: search, clustering, recommendations
✅ Cosine similarity measures how related two pieces of text are
✅ Cache embeddings - they don't change for the same text
✅ Batch processing speeds up embedding generation
✅ Foundation for RAG systems (coming in Lesson 8+)

## Next Steps

After mastering embeddings, you'll:
- Build RAG systems with vector databases (Lesson 8+)
- Implement advanced retrieval strategies
- Create production-grade semantic search applications

Embeddings are the foundation of modern AI applications - master them well! 🎯
