"""
Message Tokenization and Cost Optimization - SOLUTION
Lesson 6: Token Management for Customer Service

This is the complete solution for the tokenization exercise.
"""

import tiktoken
import os
from typing import List, Dict, Tuple, Optional
from openai import OpenAI


class MessageTokenizer:
    """
    Handles tokenization and cost optimization for customer service messages.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the message tokenizer.

        Args:
            model: The model to use for tokenization (default: gpt-3.5-turbo)
        """
        self.model = model

        # Initialize the tokenizer using tiktoken
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Define pricing for different models (cost per 1K tokens)
        self.pricing = {
            "gpt-3.5-turbo": {
                "input": 0.0005,
                "output": 0.0015
            },
            "gpt-4": {
                "input": 0.03,
                "output": 0.06
            },
            "gpt-4-turbo": {
                "input": 0.01,
                "output": 0.03
            }
        }

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to tokenize

        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0

        # Encode the text and count tokens
        tokens = self.encoding.encode(text)
        return len(tokens)

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a message list (conversation format).

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Total number of tokens including message overhead
        """
        if not messages:
            return 0

        # Start with base overhead for conversation formatting
        total_tokens = 3

        for message in messages:
            # Each message has overhead: role formatting + content
            total_tokens += 4  # Message formatting overhead

            # Count tokens in role and content
            total_tokens += self.count_tokens(message.get("role", ""))
            total_tokens += self.count_tokens(message.get("content", ""))

        return total_tokens

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """
        Estimate the cost of an API call based on token counts.

        Args:
            input_tokens: Number of input tokens (prompt + history)
            output_tokens: Number of output tokens (response)
            model: Model to use for pricing (defaults to self.model)

        Returns:
            Estimated cost in dollars
        """
        # Use provided model or default
        model_name = model or self.model

        # Get pricing for the model
        if model_name not in self.pricing:
            # Use gpt-3.5-turbo pricing as fallback
            pricing = self.pricing["gpt-3.5-turbo"]
        else:
            pricing = self.pricing[model_name]

        # Calculate cost: (tokens / 1000) * price per 1K tokens
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost

    def optimize_conversation_history(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        keep_system_prompt: bool = True
    ) -> List[Dict[str, str]]:
        """
        Optimize conversation history to fit within token limit.

        Args:
            messages: Full conversation history
            max_tokens: Maximum tokens allowed
            keep_system_prompt: Whether to always keep the first (system) message

        Returns:
            Optimized message list within token limit
        """
        if not messages:
            return []

        # Handle system prompt
        system_message = None
        conversation_messages = messages

        if keep_system_prompt and messages[0].get("role") == "system":
            system_message = messages[0]
            conversation_messages = messages[1:]

        # Start with empty optimized list
        optimized = []
        current_tokens = 0

        # If keeping system prompt, count its tokens
        if system_message:
            current_tokens = self.count_message_tokens([system_message])

        # Work backwards through messages (keep most recent)
        for message in reversed(conversation_messages):
            # Count tokens for this message (with overhead)
            message_tokens = self.count_message_tokens([message])

            # Check if adding this message would exceed limit
            if current_tokens + message_tokens <= max_tokens:
                optimized.insert(0, message)  # Add to beginning
                current_tokens += message_tokens
            else:
                # Can't fit any more messages
                break

        # Add system prompt back at the beginning
        if system_message:
            optimized.insert(0, system_message)

        return optimized

    def chunk_long_message(
        self,
        text: str,
        max_chunk_tokens: int,
        overlap_tokens: int = 50
    ) -> List[str]:
        """
        Split a long message into smaller chunks that fit within token limits.

        Args:
            text: The long text to split
            max_chunk_tokens: Maximum tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks

        Returns:
            List of text chunks, each within token limit
        """
        if not text:
            return []

        # Encode the entire text into tokens
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        # If text fits in one chunk, return it as-is
        if total_tokens <= max_chunk_tokens:
            return [text]

        chunks = []
        start = 0

        while start < total_tokens:
            # Calculate end position for this chunk
            end = min(start + max_chunk_tokens, total_tokens)

            # Extract chunk tokens
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start position (with overlap)
            # If this is not the last chunk, add overlap
            if end < total_tokens:
                start = end - overlap_tokens
            else:
                break

        return chunks

    def analyze_message_cost(self, message: str, expected_response_tokens: int = 150) -> Dict:
        """
        Analyze a single message for token count and cost.

        Args:
            message: The customer message to analyze
            expected_response_tokens: Estimated tokens in the response

        Returns:
            Dictionary with analysis: tokens, cost, recommendations
        """
        # Count tokens in the message
        input_tokens = self.count_tokens(message)

        # Estimate cost
        estimated_cost = self.estimate_cost(input_tokens, expected_response_tokens)

        # Categorize message length
        if input_tokens < 50:
            length_category = "short"
            recommendation = "Optimal message length. Low cost per interaction."
        elif input_tokens < 200:
            length_category = "medium"
            recommendation = "Good message length. Consider if all details are necessary."
        else:
            length_category = "long"
            recommendation = "Long message. Consider chunking or summarizing for cost optimization."

        return {
            "input_tokens": input_tokens,
            "expected_output_tokens": expected_response_tokens,
            "estimated_cost": estimated_cost,
            "message_length": length_category,
            "recommendation": recommendation,
            "cost_per_100_interactions": estimated_cost * 100
        }


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

    for i, msg in enumerate(messages, 1):
        token_count = tokenizer.count_tokens(msg)
        char_count = len(msg)
        print(f"Message {i}:")
        print(f"  Characters: {char_count}")
        print(f"  Tokens: {token_count}")
        print(f"  Ratio: {char_count/token_count:.2f} chars per token")
        print(f"  Preview: {msg[:60]}...")
        print()

    print("Key insight: On average, 1 token ≈ 4 characters in English.")
    print("Shorter messages have fewer tokens, but the char/token ratio varies!")


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

    for scenario in scenarios:
        cost_35 = tokenizer.estimate_cost(
            scenario["input_tokens"],
            scenario["output_tokens"],
            "gpt-3.5-turbo"
        )
        cost_4 = tokenizer.estimate_cost(
            scenario["input_tokens"],
            scenario["output_tokens"],
            "gpt-4"
        )

        print(f"{scenario['name']}:")
        print(f"  Input tokens: {scenario['input_tokens']}")
        print(f"  Output tokens: {scenario['output_tokens']}")
        print(f"  GPT-3.5-turbo cost: ${cost_35:.6f}")
        print(f"  GPT-4 cost: ${cost_4:.6f}")
        print(f"  GPT-4 is {cost_4/cost_35:.1f}x more expensive")
        print(f"  Cost per 1000 interactions (GPT-3.5): ${cost_35 * 1000:.2f}")
        print()

    print("Key insight: Model choice significantly impacts costs at scale!")


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

    original_tokens = tokenizer.count_message_tokens(conversation)
    print(f"\nOriginal conversation: {len(conversation)} messages, {original_tokens} tokens")

    # Optimize for different limits
    limits = [200, 150, 100]

    for limit in limits:
        optimized = tokenizer.optimize_conversation_history(conversation, limit)
        optimized_tokens = tokenizer.count_message_tokens(optimized)
        print(f"\nOptimized for {limit} tokens:")
        print(f"  Messages kept: {len(optimized)}")
        print(f"  Actual tokens: {optimized_tokens}")
        print(f"  Messages removed: {len(conversation) - len(optimized)}")
        print(f"  System prompt kept: {optimized[0]['role'] == 'system' if optimized else False}")

    print("\nKey insight: Keep recent context while respecting token limits.")


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

    print(f"\nOriginal message:")
    print(f"  Characters: {len(long_complaint)}")
    print(f"  Tokens: {tokenizer.count_tokens(long_complaint)}")

    # Chunk into smaller pieces
    chunks = tokenizer.chunk_long_message(long_complaint, max_chunk_tokens=100, overlap_tokens=20)

    print(f"\nSplit into {len(chunks)} chunks (max 100 tokens, 20 token overlap):")

    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = tokenizer.count_tokens(chunk)
        print(f"\nChunk {i}: {chunk_tokens} tokens")
        print(f"  Preview: {chunk[:80]}...")

        # Show overlap detection
        if i > 1:
            # Check if this chunk starts with similar content to previous chunk's end
            print(f"  (Contains overlap from previous chunk for context)")

    print("\nKey insight: Chunking with overlap preserves context across segments.")


def demonstrate_complete_analysis():
    """
    Demonstrate complete message analysis.
    """
    print("\n" + "="*70)
    print("DEMO 5: Complete Message Cost Analysis")
    print("="*70)

    tokenizer = MessageTokenizer()

    test_messages = [
        "Order status?",
        "I received the wrong item in my order. Can you help me return it and get the correct one?",
        """I've been a loyal customer for 5 years, but my recent experience has been terrible.
        I ordered 3 items, only 1 arrived, it was the wrong color, and customer service has been
        unhelpful. I've called 4 times, been transferred to different departments, and still no
        resolution. I'm considering taking my business elsewhere if this isn't resolved immediately."""
    ]

    print("\nAnalyzing message costs:\n")

    for i, msg in enumerate(test_messages, 1):
        analysis = tokenizer.analyze_message_cost(msg)

        print(f"Message {i}: {msg[:50]}...")
        print(f"  Input tokens: {analysis['input_tokens']}")
        print(f"  Expected output tokens: {analysis['expected_output_tokens']}")
        print(f"  Length category: {analysis['message_length']}")
        print(f"  Cost per interaction: ${analysis['estimated_cost']:.6f}")
        print(f"  Cost per 100 interactions: ${analysis['cost_per_100_interactions']:.4f}")
        print(f"  Recommendation: {analysis['recommendation']}")
        print()


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
    print("5. Perform complete cost analysis")

    # Check if OpenAI API key is available (optional for this demo)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nNote: OPENAI_API_KEY not set. This demo focuses on tokenization")
        print("and doesn't require API calls, but you'll need it for real usage.")

    # Run all demonstrations
    demonstrate_token_counting()
    demonstrate_cost_estimation()
    demonstrate_history_optimization()
    demonstrate_message_chunking()
    demonstrate_complete_analysis()

    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. Token Basics:")
    print("   - 1 token ≈ 4 characters in English")
    print("   - Tokens are the billing unit for LLM APIs")
    print("   - Different languages may have different token ratios")

    print("\n2. Cost Optimization:")
    print("   - GPT-3.5-turbo is much cheaper than GPT-4")
    print("   - Conversation history adds up quickly")
    print("   - Keep only necessary context to reduce costs")

    print("\n3. Practical Strategies:")
    print("   - Limit conversation history to recent messages")
    print("   - Chunk long messages for better processing")
    print("   - Choose the right model for the task")
    print("   - Monitor token usage in production")

    print("\n4. Customer Service Applications:")
    print("   - Short inquiries are cheap to process")
    print("   - Long conversations need history management")
    print("   - Detailed complaints may need chunking")
    print("   - Balance cost and quality for best experience")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
