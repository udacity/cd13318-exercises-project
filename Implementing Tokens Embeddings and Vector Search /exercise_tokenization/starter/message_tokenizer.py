"""
Message Tokenization and Cost Optimization
Lesson 6: Token Management for Customer Service

This exercise teaches you how to count tokens, estimate API costs, and optimize
message processing for a customer service chatbot. Understanding tokens helps you
control costs and handle long customer messages efficiently.

Learning Objectives:
- Count tokens in customer messages using tiktoken
- Estimate API costs based on token counts
- Optimize conversation history to fit within token limits
- Split long messages into manageable chunks
- Make cost-aware decisions in chatbot design
"""

import tiktoken
import os
from typing import List, Dict, Tuple, Optional
from openai import OpenAI


class MessageTokenizer:
    """
    Handles tokenization and cost optimization for customer service messages.

    This class helps you understand how tokens work and manage API costs
    by counting tokens, estimating costs, and optimizing message history.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the message tokenizer.

        Args:
            model: The model to use for tokenization (default: gpt-3.5-turbo)
        """
        # TODO: Store the model name
        self.model = None

        # TODO: Initialize the tokenizer using tiktoken
        # Hint: Use tiktoken.encoding_for_model(model)
        # This loads the correct encoding for the model
        self.encoding = None

        # TODO: Define pricing for different models (cost per 1K tokens)
        # Format: {model_name: {"input": cost, "output": cost}}
        # GPT-3.5-turbo: $0.0005 per 1K input tokens, $0.0015 per 1K output tokens
        # GPT-4: $0.03 per 1K input tokens, $0.06 per 1K output tokens
        self.pricing = {}

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        This is the foundation for understanding API costs. Each API call
        charges based on the number of tokens processed.

        Args:
            text: The text to tokenize

        Returns:
            Number of tokens in the text
        """
        # TODO: Use the encoding to count tokens
        # Hint: Use self.encoding.encode(text) to get token list
        # Then return the length of that list
        pass

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a message list (conversation format).

        The OpenAI API uses a message format with roles. Each message has
        some overhead tokens beyond just the content. This function accounts
        for that overhead.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Total number of tokens including message overhead
        """
        # TODO: Count tokens for a conversation
        # For each message:
        # 1. Count tokens in the content
        # 2. Add overhead tokens (approximately 4 tokens per message for role formatting)
        # 3. Add a small base overhead (approximately 3 tokens for the conversation)

        # Hint: Start with 3 tokens for base overhead
        # Then for each message, add 4 + count_tokens(content)
        pass

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """
        Estimate the cost of an API call based on token counts.

        Understanding costs helps you make smart decisions about how to
        structure your chatbot conversations and when to optimize.

        Args:
            input_tokens: Number of input tokens (prompt + history)
            output_tokens: Number of output tokens (response)
            model: Model to use for pricing (defaults to self.model)

        Returns:
            Estimated cost in dollars
        """
        # TODO: Calculate the cost based on pricing
        # 1. Get the model to use (parameter or self.model)
        # 2. Get pricing for that model from self.pricing
        # 3. Calculate: (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
        # 4. Return the total cost

        # Hint: Pricing is per 1000 tokens, so divide token counts by 1000
        pass

    def optimize_conversation_history(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        keep_system_prompt: bool = True
    ) -> List[Dict[str, str]]:
        """
        Optimize conversation history to fit within token limit.

        When conversations get long, they can exceed model context limits or
        become expensive. This function keeps the most recent messages while
        staying under the token limit.

        Args:
            messages: Full conversation history
            max_tokens: Maximum tokens allowed
            keep_system_prompt: Whether to always keep the first (system) message

        Returns:
            Optimized message list within token limit
        """
        # TODO: Implement conversation history optimization
        # Strategy: Keep recent messages, drop older ones

        # 1. If keep_system_prompt and first message is system, save it separately
        # 2. Start with empty optimized list
        # 3. Work backwards through messages (most recent first)
        # 4. Keep adding messages while total tokens < max_tokens
        # 5. Reverse the list to restore chronological order
        # 6. Add system prompt back at the beginning if needed

        # Hint: Use count_message_tokens() to check total size
        # Hint: Work backwards with messages[::-1] or reversed()
        pass

    def chunk_long_message(
        self,
        text: str,
        max_chunk_tokens: int,
        overlap_tokens: int = 50
    ) -> List[str]:
        """
        Split a long message into smaller chunks that fit within token limits.

        Sometimes customers send very long messages (like detailed complaints
        or product reviews). This function breaks them into manageable pieces
        while maintaining some overlap for context.

        Args:
            text: The long text to split
            max_chunk_tokens: Maximum tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks

        Returns:
            List of text chunks, each within token limit
        """
        # TODO: Implement message chunking with overlap

        # 1. Encode the entire text into tokens
        # 2. Split tokens into chunks of max_chunk_tokens
        # 3. Add overlap_tokens from previous chunk to each new chunk
        # 4. Decode each chunk back to text
        # 5. Return list of text chunks

        # Hint: Use self.encoding.encode(text) to get tokens
        # Hint: Use self.encoding.decode(token_list) to get text back
        # Hint: tokens[i:i+max_chunk_tokens] to slice a chunk
        pass

    def analyze_message_cost(self, message: str, expected_response_tokens: int = 150) -> Dict:
        """
        Analyze a single message for token count and cost.

        This helps you understand the cost impact of different message lengths
        and provides useful information for optimization decisions.

        Args:
            message: The customer message to analyze
            expected_response_tokens: Estimated tokens in the response

        Returns:
            Dictionary with analysis: tokens, cost, recommendations
        """
        # TODO: Perform comprehensive message analysis

        # 1. Count tokens in the message
        # 2. Estimate cost (message tokens + expected response tokens)
        # 3. Determine if message is "short" (<50), "medium" (50-200), or "long" (>200)
        # 4. Provide recommendations based on length

        # Return a dict with keys:
        # - "input_tokens": int
        # - "expected_output_tokens": int
        # - "estimated_cost": float
        # - "message_length": str (short/medium/long)
        # - "recommendation": str (advice on handling this message)
        pass


def demonstrate_token_counting():
    """
    Demonstrate basic token counting with customer messages.
    """
    print("\n" + "="*70)
    print("DEMO 1: Token Counting Basics")
    print("="*70)

    tokenizer = MessageTokenizer()

    # Sample customer messages of different lengths
    messages = [
        "Hi, I need help.",
        "Where is my order? I placed it 3 days ago and haven't received any updates.",
        """I'm extremely frustrated with my recent purchase. I ordered a laptop two weeks ago,
        and it arrived damaged. I contacted support immediately, but I've been getting the runaround
        with no clear resolution. The screen has a crack, the keyboard is missing keys, and the
        packaging looked like it had been dropped multiple times. I need a replacement or full
        refund immediately. This is unacceptable service."""
    ]

    print("\nAnalyzing different message lengths:\n")

    # TODO: Uncomment and complete this section
    # for i, msg in enumerate(messages, 1):
    #     token_count = tokenizer.count_tokens(msg)
    #     char_count = len(msg)
    #     print(f"Message {i}:")
    #     print(f"  Characters: {char_count}")
    #     print(f"  Tokens: {token_count}")
    #     print(f"  Ratio: {char_count/token_count:.2f} chars per token")
    #     print(f"  Preview: {msg[:60]}...")
    #     print()

    print("Notice: Shorter messages have fewer tokens, but the char/token ratio varies!")


def demonstrate_cost_estimation():
    """
    Demonstrate cost estimation for API calls.
    """
    print("\n" + "="*70)
    print("DEMO 2: Cost Estimation")
    print("="*70)

    tokenizer = MessageTokenizer()

    # Simulate different conversation scenarios
    scenarios = [
        {
            "name": "Simple inquiry",
            "input_tokens": 50,
            "output_tokens": 100
        },
        {
            "name": "Detailed support conversation",
            "input_tokens": 500,
            "output_tokens": 300
        },
        {
            "name": "Long conversation with history",
            "input_tokens": 2000,
            "output_tokens": 400
        }
    ]

    print("\nComparing costs for different scenarios:\n")

    # TODO: Uncomment and complete this section
    # for scenario in scenarios:
    #     cost_35 = tokenizer.estimate_cost(
    #         scenario["input_tokens"],
    #         scenario["output_tokens"],
    #         "gpt-3.5-turbo"
    #     )
    #     cost_4 = tokenizer.estimate_cost(
    #         scenario["input_tokens"],
    #         scenario["output_tokens"],
    #         "gpt-4"
    #     )
    #
    #     print(f"{scenario['name']}:")
    #     print(f"  Input tokens: {scenario['input_tokens']}")
    #     print(f"  Output tokens: {scenario['output_tokens']}")
    #     print(f"  GPT-3.5-turbo cost: ${cost_35:.6f}")
    #     print(f"  GPT-4 cost: ${cost_4:.6f}")
    #     print(f"  GPT-4 is {cost_4/cost_35:.1f}x more expensive")
    #     print()


def demonstrate_history_optimization():
    """
    Demonstrate optimizing conversation history to fit token limits.
    """
    print("\n" + "="*70)
    print("DEMO 3: Conversation History Optimization")
    print("="*70)

    tokenizer = MessageTokenizer()

    # Simulate a long customer conversation
    conversation = [
        {"role": "system", "content": "You are a helpful customer service assistant."},
        {"role": "user", "content": "Hi, I need help with my order."},
        {"role": "assistant", "content": "I'd be happy to help! Could you provide your order number?"},
        {"role": "user", "content": "It's ORDER-12345."},
        {"role": "assistant", "content": "Thank you! I'm looking that up now. Your order was shipped yesterday."},
        {"role": "user", "content": "Great! When will it arrive?"},
        {"role": "assistant", "content": "It should arrive within 3-5 business days. You'll get tracking updates via email."},
        {"role": "user", "content": "Actually, I need to change the delivery address. Can I do that?"},
        {"role": "assistant", "content": "Since it's already shipped, I'll need to contact the carrier. What's the new address?"},
        {"role": "user", "content": "123 New Street, New City, NC 12345"},
    ]

    # TODO: Uncomment and complete this section
    # original_tokens = tokenizer.count_message_tokens(conversation)
    # print(f"\nOriginal conversation: {len(conversation)} messages, {original_tokens} tokens")

    # Optimize for different limits
    # limits = [200, 150, 100]
    #
    # for limit in limits:
    #     optimized = tokenizer.optimize_conversation_history(conversation, limit)
    #     optimized_tokens = tokenizer.count_message_tokens(optimized)
    #     print(f"\nOptimized for {limit} tokens:")
    #     print(f"  Messages kept: {len(optimized)}")
    #     print(f"  Actual tokens: {optimized_tokens}")
    #     print(f"  Messages removed: {len(conversation) - len(optimized)}")


def demonstrate_message_chunking():
    """
    Demonstrate splitting long messages into chunks.
    """
    print("\n" + "="*70)
    print("DEMO 4: Long Message Chunking")
    print("="*70)

    tokenizer = MessageTokenizer()

    # A very long customer complaint
    long_complaint = """I am writing to express my extreme dissatisfaction with the recent purchase
    I made from your company. On January 15th, I ordered a premium laptop computer (Model X-2000)
    for $1,500. The product was advertised as "brand new" with "factory sealed packaging" and
    promised delivery within 3-5 business days. However, my experience has been nothing short of
    a nightmare. First, the delivery took 12 days, nearly triple the promised time. When the
    package finally arrived, it was clear that it had been damaged during shipping - the box was
    crushed on one corner and had visible water damage. Against my better judgment, I opened it
    hoping the contents were intact. Unfortunately, the laptop itself showed signs of previous use:
    fingerprints on the screen, scratches on the case, and the "sealed" plastic wrap had clearly
    been removed and reapplied. When I powered it on, I discovered it had files from a previous
    user still on the hard drive. This is completely unacceptable from a security and privacy
    standpoint. I immediately contacted your customer service department and spent 45 minutes on
    hold before speaking with a representative who seemed unfamiliar with the return process. After
    explaining the situation three times to three different people, I was finally told I could
    return the item but would have to pay for shipping and a 15% restocking fee. This is outrageous
    given that I received a used, damaged product instead of the new item I paid for. I am demanding
    a full refund including shipping costs, or I will be forced to dispute the charge with my credit
    card company and report this to the Better Business Bureau."""

    # TODO: Uncomment and complete this section
    # print(f"\nOriginal message length: {len(long_complaint)} characters")
    # print(f"Original message tokens: {tokenizer.count_tokens(long_complaint)}")
    #
    # # Chunk into smaller pieces
    # chunks = tokenizer.chunk_long_message(long_complaint, max_chunk_tokens=100, overlap_tokens=20)
    #
    # print(f"\nSplit into {len(chunks)} chunks:")
    # for i, chunk in enumerate(chunks, 1):
    #     chunk_tokens = tokenizer.count_tokens(chunk)
    #     print(f"\nChunk {i}: {chunk_tokens} tokens")
    #     print(f"Preview: {chunk[:100]}...")


def main():
    """
    Run all demonstrations of the tokenization system.
    """
    print("\n" + "="*70)
    print("MESSAGE TOKENIZATION AND COST OPTIMIZATION")
    print("Customer Service Use Case")
    print("="*70)

    print("\nThis demo shows you how to:")
    print("1. Count tokens in customer messages")
    print("2. Estimate API costs for different scenarios")
    print("3. Optimize conversation history to reduce costs")
    print("4. Handle long customer messages by chunking")

    # Check if OpenAI API key is available (optional for this demo)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nNote: OPENAI_API_KEY not set. This demo focuses on tokenization")
        print("and doesn't require API calls, but you'll need it for real usage.")

    # Run demonstrations
    # TODO: Uncomment these as you implement the functions
    # demonstrate_token_counting()
    # demonstrate_cost_estimation()
    # demonstrate_history_optimization()
    # demonstrate_message_chunking()

    print("\n" + "="*70)
    print("Key Takeaways:")
    print("- Tokens are the basic unit of LLM processing (roughly 4 chars = 1 token)")
    print("- API costs scale with token count, so optimization matters")
    print("- Keep conversation history concise to reduce costs")
    print("- Long messages can be chunked for better processing")
    print("- GPT-4 is much more expensive than GPT-3.5-turbo per token")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
