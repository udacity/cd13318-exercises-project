"""
Customer Service Chatbot Implementation - SOLUTION
Lesson 3: Implementing a Chatbot with an LLM

This is the complete solution for the customer service chatbot exercise.
"""

from openai import OpenAI
import os
from typing import List, Dict, Optional
from datetime import datetime


class CustomerServiceBot:
    """
    A chatbot that handles common customer inquiries for an e-commerce platform.
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the customer service bot.

        Args:
            api_key: OpenAI API key (or Vocareum key)
            model: The model to use for responses (default: gpt-3.5-turbo)
        """
        # Initialize the OpenAI client with Vocareum base URL
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=api_key
        )
        self.model = model

        # Initialize conversation history with system prompt
        self.conversation_history: List[Dict[str, str]] = []
        self.conversation_history.append({
            "role": "system",
            "content": self._get_system_prompt()
        })

    def _get_system_prompt(self) -> str:
        """Define the system prompt that sets the bot's behavior."""
        return """You are a helpful customer service assistant for ShopEasy, an e-commerce platform.

Your role is to assist customers with:
- Order status and tracking
- Product information and recommendations
- Return and refund policies
- Technical support issues
- Account questions

Guidelines:
- Be professional, friendly, and empathetic
- Provide clear, concise answers
- Ask clarifying questions when the customer's intent is unclear
- If you don't have specific information (like order numbers), ask for it
- Always prioritize customer satisfaction

If a request is outside your capabilities, politely explain and offer to escalate to a human agent."""

    def classify_intent(self, message: str) -> str:
        """
        Classify the customer's intent to route to the appropriate handler.

        Args:
            message: The customer's message

        Returns:
            Intent category
        """
        classification_prompt = f"""Classify the following customer message into ONE of these categories:
- order_status: Questions about order tracking, delivery, or status
- product_info: Questions about products, features, availability, or recommendations
- returns: Questions about returns, refunds, or exchanges
- technical_support: Technical issues with the website, app, or account
- general: General inquiries or greetings

Customer message: "{message}"

Respond with ONLY the category name, nothing else."""

        try:
            # Make a simple API call for classification
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0,  # Use 0 for consistent classification
                max_tokens=20
            )

            intent = response.choices[0].message.content.strip().lower()

            # Validate the intent is one of our expected categories
            valid_intents = ['order_status', 'product_info', 'returns', 'technical_support', 'general']
            if intent not in valid_intents:
                intent = 'general'

            return intent

        except Exception as e:
            print(f"Error classifying intent: {e}")
            return "general"

    def generate_response(self, user_message: str, intent: Optional[str] = None) -> str:
        """
        Generate a contextual response to the user's message.

        Args:
            user_message: The customer's message
            intent: Optional intent classification

        Returns:
            The bot's response
        """
        # Classify intent if not provided
        if intent is None:
            intent = self.classify_intent(user_message)
            print(f"[Intent detected: {intent}]")  # For debugging/demonstration

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            # Generate response with full conversation context
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,  # Balanced creativity and consistency
                max_tokens=300  # Reasonable length for customer service
            )

            assistant_message = response.choices[0].message.content

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except Exception as e:
            error_msg = "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
            print(f"Error generating response: {e}")
            return error_msg

    def reset_conversation(self):
        """Reset the conversation history, keeping only the system prompt."""
        self.conversation_history = [{
            "role": "system",
            "content": self._get_system_prompt()
        }]
        print("[Conversation reset]")

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for handoff to human agent."""
        if len(self.conversation_history) <= 1:  # Only system prompt
            return "No conversation to summarize yet."

        summary_prompt = """Please provide a brief summary of this customer service conversation.
Include:
1. Main customer concerns or questions
2. Information provided by the bot
3. Current status or next steps

Keep it concise (2-3 sentences)."""

        # Create temporary message list for summary
        summary_messages = self.conversation_history + [{
            "role": "user",
            "content": summary_prompt
        }]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_messages,
                temperature=0.3,  # Lower temperature for factual summary
                max_tokens=200
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Unable to generate summary at this time."


def main():
    """Demo the customer service bot with sample interactions."""

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("\nFor Vocareum keys:")
        print('  export OPENAI_API_KEY="voc-..."')
        print("\nFor standard OpenAI keys:")
        print('  export OPENAI_API_KEY="sk-..."')
        return

    # Initialize the bot
    print("Initializing Customer Service Bot...")
    bot = CustomerServiceBot(api_key)

    print("\n" + "="*60)
    print("Customer Service Bot Ready!")
    print("="*60)
    print("\nCommands:")
    print("  'quit' or 'exit' - End the session")
    print("  'reset' - Start a new conversation")
    print("  'summary' - Get a conversation summary")
    print("\nSample questions to try:")

    sample_questions = [
        "Where is my order? I placed it 3 days ago.",
        "Do you have wireless headphones in stock?",
        "What's your return policy?",
        "I can't log into my account",
        "Can you recommend a laptop for students?"
    ]

    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")

    print("\n" + "="*60 + "\n")

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for using Customer Service Bot!")
                print("Have a great day! 👋")
                break

            if user_input.lower() == 'reset':
                bot.reset_conversation()
                continue

            if user_input.lower() == 'summary':
                print("\n--- Conversation Summary ---")
                print(bot.get_conversation_summary())
                print("----------------------------\n")
                continue

            # Generate response
            response = bot.generate_response(user_input)
            print(f"\nBot: {response}\n")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Type 'quit' to exit or continue chatting.\n")


if __name__ == "__main__":
    main()
