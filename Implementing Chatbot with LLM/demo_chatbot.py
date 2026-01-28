from openai import OpenAI

# Configure Vocareum OpenAI client
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key="voc-00000000000000000000000000000000abcd.12345678"  # Replace with your actual key
)

def demonstrate_stateless_conversation():
    """
    Demonstrates how LLMs maintain context through conversation history,
    despite being completely stateless.
    """

    # Initialize conversation with system message
    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful tech support assistant. Be patient and clear."
        }
    ]

    print("=" * 60)
    print("DEMONSTRATION: The Stateless Nature of LLMs")
    print("=" * 60)

    # Turn 1: First user message (no previous context)
    print("\n--- TURN 1 ---")
    user_message_1 = "What's the weather like today?"
    conversation_history.append({"role": "user", "content": user_message_1})

    print(f"User: {user_message_1}")
    print(f"Messages sent to API: {len(conversation_history)} messages")

    response_1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    assistant_message_1 = response_1.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message_1})
    print(f"Assistant: {assistant_message_1}")

    # Turn 2: Second user message (includes full history)
    print("\n--- TURN 2 ---")
    user_message_2 = "Should I bring an umbrella?"
    conversation_history.append({"role": "user", "content": user_message_2})

    print(f"User: {user_message_2}")
    print(f"Messages sent to API: {len(conversation_history)} messages")
    print(f"  (includes: system message, previous user message, previous assistant response)")

    response_2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    assistant_message_2 = response_2.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message_2})
    print(f"Assistant: {assistant_message_2}")

    # Turn 3: Third user message with pronoun reference
    print("\n--- TURN 3 ---")
    user_message_3 = "Can you explain it differently?"
    conversation_history.append({"role": "user", "content": user_message_3})

    print(f"User: {user_message_3}")
    print(f"Messages sent to API: {len(conversation_history)} messages")
    print(f"  (The model needs ALL previous messages to understand what 'it' refers to)")

    response_3 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    assistant_message_3 = response_3.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message_3})
    print(f"Assistant: {assistant_message_3}")

    # Display final conversation structure
    print("\n" + "=" * 60)
    print("FINAL CONVERSATION HISTORY")
    print("=" * 60)
    for i, msg in enumerate(conversation_history, 1):
        print(f"{i}. [{msg['role'].upper()}]: {msg['content'][:60]}...")

if __name__ == "__main__":
    print("\nREMEMBER: Replace the API key with your actual Vocareum key!")
    print("Find it in: Cloud Resources → OpenAI Key\n")

    demonstrate_stateless_conversation()
