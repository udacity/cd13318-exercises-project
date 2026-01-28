"""
Customer Service Chatbot Implementation
Lesson 3: Implementing a Chatbot with an LLM

This exercise guides you through building a customer service chatbot for an e-commerce platform.
The bot handles common customer inquiries about orders, products, returns, and technical support.

Learning Objectives:
- Initialize and configure the OpenAI API client
- Design prompt templates for different intent types
- Maintain conversation history for context
- Classify customer intents and route to appropriate handlers
- Generate contextual, helpful responses
"""

from openai import OpenAI
import os
from typing import List, Dict, Optional
from datetime import datetime


class CustomerServiceBot:
    """
    A chatbot that handles common customer inquiries for an e-commerce platform.

    Capabilities:
    - Order status inquiries
    - Product information requests
    - Return and refund policies
    - Technical support
    - General customer service
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the customer service bot.

        Args:
            api_key: OpenAI API key (or Vocareum key)
            model: The model to use for responses (default: gpt-3.5-turbo)
        """
        # TODO: Initialize the OpenAI client
        # Hint: Use OpenAI(api_key=api_key) for standard keys
        # For Vocareum keys, add: base_url="https://openai.vocareum.com/v1"
        self.client = None

        # TODO: Store the model name
        self.model = None

        # TODO: Initialize conversation history as an empty list
        # Each message should be a dict with 'role' and 'content'
        self.conversation_history = None

        # TODO: Load the system prompt that defines the bot's behavior
        # Call self._get_system_prompt() and add it to conversation_history
        # The system message should have role='system'
        pass

    def _get_system_prompt(self) -> str:
        """
        Define the system prompt that sets the bot's behavior and personality.

        Returns:
            The system prompt as a string
        """
        # TODO: Create a comprehensive system prompt that:
        # 1. Defines the bot's role (helpful customer service assistant)
        # 2. Specifies the tone (professional, friendly, empathetic)
        # 3. Lists the types of inquiries it can handle
        # 4. Provides guidelines (be concise, ask clarifying questions if needed)

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
            Intent category: 'order_status', 'product_info', 'returns',
                           'technical_support', or 'general'
        """
        # TODO: Use the LLM to classify the intent
        # Create a prompt that asks the model to categorize the message
        # Use a simple API call with specific instructions
        # Return just the category name

        classification_prompt = f"""Classify the following customer message into ONE of these categories:
- order_status: Questions about order tracking, delivery, or status
- product_info: Questions about products, features, availability, or recommendations
- returns: Questions about returns, refunds, or exchanges
- technical_support: Technical issues with the website, app, or account
- general: General inquiries or greetings

Customer message: "{message}"

Respond with ONLY the category name, nothing else."""

        try:
            # TODO: Make an API call to get the classification
            # Use client.chat.completions.create()
            # Pass a simple message list with just the classification prompt
            # Extract the category from the response

            response = None  # Replace with actual API call

            # TODO: Extract and return the intent category
            # Get response.choices[0].message.content and strip whitespace
            intent = "general"  # Default fallback

            return intent

        except Exception as e:
            print(f"Error classifying intent: {e}")
            return "general"  # Default to general on error

    def generate_response(self, user_message: str, intent: Optional[str] = None) -> str:
        """
        Generate a contextual response to the user's message.

        Args:
            user_message: The customer's message
            intent: Optional intent classification (will auto-classify if not provided)

        Returns:
            The bot's response as a string
        """
        # TODO: If intent not provided, classify it
        if intent is None:
            pass  # Call classify_intent()

        # TODO: Add the user's message to conversation history
        # Append a dict with role='user' and content=user_message

        try:
            # TODO: Make the API call to generate a response
            # Use client.chat.completions.create()
            # Pass the entire conversation_history
            # Use temperature=0.7 for natural but consistent responses

            response = None  # Replace with actual API call

            # TODO: Extract the assistant's response
            assistant_message = ""  # Get from response.choices[0].message.content

            # TODO: Add the assistant's message to conversation history
            # Append a dict with role='assistant' and content=assistant_message

            return assistant_message

        except Exception as e:
            error_msg = f"I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
            print(f"Error generating response: {e}")
            return error_msg

    def reset_conversation(self):
        """
        Reset the conversation history, keeping only the system prompt.
        Useful when starting a new customer conversation.
        """
        # TODO: Clear conversation_history and re-add the system prompt
        pass

    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation for handoff to human agent.

        Returns:
            A brief summary of the customer's inquiries and bot responses
        """
        # TODO: Use the LLM to create a summary of the conversation
        # This is useful when escalating to a human agent

        summary_prompt = """Please provide a brief summary of this customer service conversation.
Include:
1. Main customer concerns or questions
2. Information provided by the bot
3. Current status or next steps

Keep it concise (2-3 sentences)."""

        # TODO: Create a temporary message list with conversation_history + summary request
        # Make an API call to get the summary
        # Return the summary text

        return "Conversation summary not yet implemented"


def main():
    """
    Demo the customer service bot with sample interactions.
    """
    # TODO: Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    # TODO: Initialize the bot
    bot = None  # Replace with CustomerServiceBot(api_key)

    print("Customer Service Bot initialized!")
    print("Try asking about orders, products, returns, or technical issues.")
    print("Type 'quit' to exit, 'reset' to start a new conversation, or 'summary' for conversation summary.\n")

    # Sample questions to try:
    sample_questions = [
        "Where is my order? I placed it 3 days ago.",
        "Do you have wireless headphones in stock?",
        "What's your return policy?",
        "I can't log into my account"
    ]

    print("Sample questions you can try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"{i}. {q}")
    print()

    # TODO: Implement the chat loop
    # - Get user input
    # - Handle special commands (quit, reset, summary)
    # - Generate and print responses
    # - Continue until user quits

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Thank you for using Customer Service Bot!")
            break

        # TODO: Implement reset command

        # TODO: Implement summary command

        # TODO: Generate and print response
        # response = bot.generate_response(user_input)
        # print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
