# Lesson 6: Tokenization Implementation

## Overview

In this lesson, you'll learn how to count tokens, estimate API costs, and optimize message processing for efficiency. You'll work with customer service messages to understand how tokenization impacts both performance and budget.

## Learning Objectives

By completing this exercise, you will be able to:

- Count tokens accurately using the tiktoken library
- Estimate API costs for different models and usage patterns
- Optimize conversation history to fit within token limits
- Chunk long messages while preserving meaning
- Make informed decisions about cost vs. quality tradeoffs

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (optional for this exercise - mainly offline token counting)
- Completion of Lesson 5 (Chatbot Implementation) recommended

## Setup

### Install Dependencies

```bash
pip install tiktoken openai
```

### Set Your API Key (Optional)

```bash
export OPENAI_API_KEY="your-key-here"
```

Note: Most of this exercise works offline using tiktoken for token counting. The API is only needed for the complete cost analysis demonstration.

## Exercise Structure

### Starter Code
`exercises/starter/message_tokenizer.py`

Implement the `MessageTokenizer` class with functions for:
- Token counting
- Cost estimation
- History optimization
- Message chunking
- Complete cost analysis

### Solution Code
`exercises/solution/message_tokenizer.py`

Complete implementation with 5 demonstration functions showing practical applications.

## Running the Exercise

```bash
# Run individual demos
python exercises/solution/message_tokenizer.py

# The solution includes 5 demos:
# 1. Token counting basics
# 2. Cost estimation
# 3. Conversation history optimization
# 4. Long message chunking
# 5. Complete cost analysis
```

## Key Concepts

### What Are Tokens?

Tokens are the basic units that LLMs process. They're not exactly words:

- **"Hello"** → 1 token
- **"Hello, world!"** → 4 tokens (Hello, ,, world, !)
- **"ChatGPT"** → 2 tokens (Chat, GPT)

**Rules of thumb:**
- 1 token ≈ 4 characters in English
- 1 token ≈ ¾ of a word
- 100 tokens ≈ 75 words

### Why Token Counting Matters

1. **Cost Control**: APIs charge per token
2. **Context Limits**: Models have maximum token limits (e.g., 4K, 8K, 128K)
3. **Performance**: More tokens = slower responses
4. **Quality**: Optimizing token usage can improve response quality

### Model Token Limits

| Model | Context Window | Cost per 1K Input | Cost per 1K Output |
|-------|----------------|-------------------|-------------------|
| gpt-3.5-turbo | 4,096 tokens | $0.0005 | $0.0015 |
| gpt-3.5-turbo-16k | 16,384 tokens | $0.001 | $0.002 |
| gpt-4 | 8,192 tokens | $0.03 | $0.06 |
| gpt-4-32k | 32,768 tokens | $0.06 | $0.12 |

## Implementation Guide

### 1. Token Counting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text for a specific model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

**Key points:**
- Different models use different tokenizers
- Always use the correct encoding for your model
- Count both input AND output tokens for cost estimation

### 2. Cost Estimation

```python
def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate the cost of an API call."""
    input_cost = (input_tokens / 1000) * INPUT_PRICE[model]
    output_cost = (output_tokens / 1000) * OUTPUT_PRICE[model]
    return input_cost + output_cost
```

**Remember:**
- Input tokens include: system prompt + conversation history + user message
- Output tokens are the generated response
- Output tokens are typically 2-3x more expensive than input

### 3. History Optimization

When conversation history grows too large:

**Strategy 1: Keep Recent**
- Keep system prompt (always)
- Keep last N message pairs
- Drop older messages

**Strategy 2: Summarize Old**
- Keep system prompt
- Keep recent messages
- Summarize older messages into single context message

**Strategy 3: Sliding Window**
- Keep system prompt
- Keep most recent messages within token budget
- Use priority: recent > older

### 4. Message Chunking

For long customer messages that exceed limits:

```python
def chunk_long_message(text: str, max_chunk_tokens: int = 500) -> List[str]:
    """Split long text into chunks with overlap."""
    # Split on sentences or paragraphs
    # Maintain overlap for context
    # Keep chunks under max_chunk_tokens
```

**Best practices:**
- Split on sentence boundaries (not mid-sentence)
- Use overlap (50-100 tokens) between chunks for context
- Process chunks sequentially or in parallel
- Combine results intelligently

## Practical Examples

### Example 1: Customer Complaint Analysis

A customer sends a 2,000-word complaint (≈ 2,667 tokens):

**Option A: Process whole message**
- Input: 2,667 tokens
- Output: ~300 tokens
- Cost: $0.0018 (gpt-3.5-turbo)

**Option B: Chunk into 3 parts**
- Input: 3 × 900 = 2,700 tokens (slightly more due to overlap)
- Output: 3 × 100 = 300 tokens (summaries)
- Cost: $0.0018
- Benefit: Better handling of long content

### Example 2: Conversation Optimization

10-turn conversation:
- System prompt: 100 tokens (every turn)
- 10 user messages: 10 × 30 = 300 tokens
- 10 assistant responses: 10 × 100 = 1,000 tokens

**Naive approach** (keep all history):
- Turn 10 input: 100 + 300 + 1,000 = 1,400 tokens
- Cumulative cost: $0.0042

**Optimized** (keep last 5 turns):
- Turn 10 input: 100 + 150 + 500 = 750 tokens
- Cumulative cost: $0.0028
- **Savings: 33%**

## Cost Optimization Strategies

### 1. Model Selection
- Use gpt-3.5-turbo for simple tasks (60x cheaper than gpt-4)
- Reserve gpt-4 for complex reasoning
- Consider model cascading (try cheap model first, escalate if needed)

### 2. Prompt Engineering
- Make system prompts concise
- Avoid verbose examples in prompts
- Use few-shot learning strategically

### 3. Response Length Control
- Use `max_tokens` parameter to limit response length
- Request concise responses in prompt
- Set appropriate temperature (lower = more focused)

### 4. Caching
- Cache responses to common questions
- Use embeddings for FAQ matching (covered in Lesson 7)
- Avoid redundant API calls

### 5. Batch Processing
- Combine multiple simple requests
- Use parallel processing for independent tasks
- Implement request queuing for rate limit management

## Demonstrations Included

### Demo 1: Token Counting Basics
- Count tokens in various customer messages
- See how different languages affect token counts
- Compare token counts across models

### Demo 2: Cost Estimation
- Calculate costs for different scenarios
- Compare gpt-3.5-turbo vs gpt-4 costs
- Understand the impact of conversation length

### Demo 3: History Optimization
- See conversation history grow over time
- Apply optimization to stay within limits
- Compare quality with/without optimization

### Demo 4: Message Chunking
- Process long customer complaints
- Handle chunking with overlap
- Combine chunk results effectively

### Demo 5: Complete Analysis
- End-to-end cost analysis
- Real API calls with token tracking
- See actual vs estimated costs

## Common Issues

### Encoding Errors
```
KeyError: 'model-name'
```
**Solution**: Use `tiktoken.get_encoding("cl100k_base")` for newer models

### Token Count Mismatch
Your count doesn't match OpenAI's reported usage.
**Cause**: Using wrong encoding or not counting special tokens
**Solution**: Use `encoding_for_model()` for accuracy

### Running Out of Budget
**Prevention**:
- Set `max_tokens` limits
- Implement conversation truncation
- Use cheaper models where possible
- Cache common responses

## Extension Ideas

1. **Real-time Cost Tracking**: Build a dashboard showing cumulative costs
2. **Budget Alerts**: Warn when approaching token budget limits
3. **Smart Summarization**: Use LLM to summarize old conversation history
4. **A/B Testing**: Compare cost vs. quality for different strategies
5. **Multi-language Support**: Handle tokenization for non-English languages
6. **Streaming Responses**: Implement token-by-token streaming for better UX

## Key Takeaways

✅ Tokens are the currency of LLM APIs - count them carefully
✅ Output tokens cost 2-3x more than input tokens
✅ Conversation history grows quadratically in cost (each turn includes all previous turns)
✅ Optimization can reduce costs by 30-50% with minimal quality impact
✅ Always estimate costs before deploying to production
✅ Different models have vastly different costs (60x difference!)

## Next Steps

After mastering tokenization, you'll:
- Learn about embeddings for semantic search (Lesson 7)
- Build cost-effective RAG systems (Lesson 8+)
- Implement production-grade applications with proper token management

Good luck! 💰
