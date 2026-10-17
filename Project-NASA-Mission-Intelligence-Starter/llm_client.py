"""OpenAI chat completion client for grounded NASA mission answers."""

from typing import Dict, List

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a careful NASA mission history and operations expert. Answer using the "
    "retrieved context supplied with the current question. Cite supporting sources by "
    "their labels, such as [Source 1]. If the context does not contain enough evidence, "
    "say what is unknown instead of inventing details. Distinguish documented facts "
    "from inference, and keep answers clear and direct."
)


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate a context-grounded response while preserving prior turns."""
    if not openai_key:
        raise ValueError("An OpenAI API key is required")
    if not user_message or not user_message.strip():
        raise ValueError("user_message must not be empty")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (conversation_history or [])[-20:]:
        role, content = turn.get("role"), turn.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    user_content = (
        f"Retrieved context:\n{context.strip() if context and context.strip() else '[No relevant source context was retrieved.]'}"
        f"\n\nQuestion: {user_message.strip()}"
    )
    messages.append({"role": "user", "content": user_content})
    response = OpenAI(api_key=openai_key).chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return answer.strip() if answer else "I could not generate a response."
