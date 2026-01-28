# Lesson 5: Implementing a Chatbot with an LLM

## Overview

In this lesson, you'll build a customer service chatbot for an e-commerce platform using OpenAI's API. You'll learn how to maintain conversation context, classify user intents, and generate helpful responses.

## Learning Objectives

By completing this exercise, you will be able to:

- Initialize and configure the OpenAI API client
- Design effective system prompts that define bot behavior
- Maintain conversation history for contextual responses
- Classify customer intents using LLMs
- Generate natural, helpful customer service responses
- Implement conversation management features (reset, summary)

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (Vocareum key provided in course)
- Basic understanding of APIs and Python classes

## Setup

### Install Dependencies

```bash
pip install openai
```

### Set Your API Key

```bash
# For Vocareum keys (provided in course)
export OPENAI_API_KEY="voc-..."

# For standard OpenAI keys
export OPENAI_API_KEY="sk-..."
```

## Exercise Structure

### Starter Code
`exercises/starter/customer_service_bot.py`

This file contains the skeleton of the CustomerServiceBot class with TODO comments guiding you through the implementation. Work through each TODO systematically.

**Key Components to Implement:**
1. OpenAI client initialization
2. System prompt design
3. Intent classification
4. Response generation with conversation history
5. Conversation management (reset, summary)

### Solution Code
`exercises/solution/customer_service_bot.py`

A complete, working implementation. Use this to check your work or if you get stuck. Try to implement the starter code on your own first!

## Running the Exercise

### Run the Starter (as you build it)
```bash
python exercises/starter/customer_service_bot.py
```

### Run the Solution
```bash
python exercises/solution/customer_service_bot.py
```

## Sample Interactions

Try these questions with your bot:

1. **Order Status**: "Where is my order? I placed it 3 days ago."
2. **Product Info**: "Do you have wireless headphones in stock?"
3. **Returns**: "What's your return policy?"
4. **Technical Support**: "I can't log into my account"
5. **General**: "Can you recommend a laptop for students?"

## Commands

While chatting with the bot:
- `quit` or `exit` - End the session
- `reset` - Start a new conversation (clears history)
- `summary` - Get a summary of the current conversation

## Implementation Tips

### 1. System Prompt Design
Your system prompt should:
- Define the bot's role and personality
- List capabilities clearly
- Set boundaries (what it can't do)
- Provide response guidelines

### 2. Conversation History
- Store messages as `{"role": "user/assistant/system", "content": "..."}`
- Include the entire history in each API call for context
- The system prompt should always be the first message

### 3. Intent Classification
- Use a separate API call with temperature=0 for consistency
- Keep the classification prompt simple and specific
- Validate the returned intent against expected categories

### 4. Response Generation
- Use temperature=0.7 for natural but consistent responses
- Include the full conversation history
- Add both user and assistant messages to history after each turn

## Key Concepts

### Conversation Context
LLMs are stateless - they don't remember previous messages unless you include them in each request. That's why maintaining conversation history is crucial.

### Temperature Parameter
- **0.0**: Deterministic, always picks the most likely token
- **0.7**: Balanced creativity and consistency (good for customer service)
- **1.0+**: More creative and varied (riskier for factual responses)

### Token Management
Each API call consumes tokens for:
- System prompt (every call)
- All conversation history (grows over time)
- New user message
- Generated response

Long conversations can get expensive! Consider truncating old history for production systems.

## Cost Considerations

**GPT-3.5-turbo pricing** (approximate):
- Input: $0.0005 per 1K tokens
- Output: $0.0015 per 1K tokens

A typical conversation turn:
- System prompt: ~100 tokens
- Conversation history: 200-1000 tokens (grows)
- User message: 10-50 tokens
- Response: 50-200 tokens

**Total**: ~$0.0002 - $0.002 per turn

## Extension Ideas

Once you complete the basic implementation, try these enhancements:

1. **Intent-specific Responses**: Create different response templates for each intent
2. **Entity Extraction**: Pull out order numbers, product names, etc.
3. **Sentiment Analysis**: Detect frustrated customers and adjust tone
4. **Escalation Logic**: Automatically offer human agent for complex issues
5. **Response Templates**: Use templates for common questions (faster, cheaper)
6. **Multi-turn Tracking**: Track unresolved issues across conversation

## Common Issues

### API Key Not Found
```
Error: Please set OPENAI_API_KEY environment variable
```
**Solution**: Export your API key as shown in Setup section

### Import Error
```
ModuleNotFoundError: No module named 'openai'
```
**Solution**: `pip install openai`

### Conversation Too Long
If conversations get very long, you may hit token limits (4K for gpt-3.5-turbo).
**Solution**: Implement history truncation in `optimize_conversation_history()`

## Learning Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Best Practices for Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [Chat Completions Guide](https://platform.openai.com/docs/guides/chat)

## Next Steps

After completing this exercise, you'll be ready to:
- Optimize token usage (Lesson 6: Tokenization)
- Add semantic search capabilities (Lesson 7: Embeddings)
- Build complete RAG systems (Lesson 8+)

Good luck! 🚀
