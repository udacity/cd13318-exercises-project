"""
LLM Inference Parameters Exercise - SOLUTION
Lesson 1: Understanding Temperature, Top-P, and Other Parameters

Complete implementation showing how different inference parameters affect LLM output.
"""

from openai import OpenAI
import os
from typing import Dict, List
from pprint import pprint


class InferenceExplorer:
    """
    A tool for exploring different LLM inference parameters and their effects.
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """Initialize the inference explorer."""
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=api_key
        )
        self.model = model

    def generate_with_temperature(self, prompt: str, temperature: float) -> str:
        """
        Generate text with a specific temperature setting.

        Temperature controls randomness:
        - 0.0: Deterministic (always picks most likely token)
        - 0.3-0.5: Focused and consistent
        - 0.7-0.9: Balanced creativity
        - 1.0+: Very creative and unpredictable
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=100
        )
        return response.choices[0].message.content

    def compare_temperatures(self, prompt: str, temperatures: List[float]) -> Dict[float, str]:
        """Compare outputs at different temperature values."""
        results = {}
        for temp in temperatures:
            results[temp] = self.generate_with_temperature(prompt, temp)
        return results

    def generate_with_top_p(self, prompt: str, top_p: float, temperature: float = 1.0) -> str:
        """
        Generate text using nucleus sampling (top_p).

        Top-p sampling:
        - Considers only tokens whose cumulative probability mass is top_p
        - 0.1: Very focused (top 10% probability mass)
        - 0.5: Moderately focused
        - 0.9: Diverse but coherent (most common)
        - 1.0: Considers all tokens
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=100
        )
        return response.choices[0].message.content

    def generate_with_max_tokens(self, prompt: str, max_tokens: int) -> str:
        """Generate text with a maximum token limit."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content

    def generate_with_frequency_penalty(
        self,
        prompt: str,
        frequency_penalty: float
    ) -> str:
        """
        Generate text with frequency penalty to reduce repetition.

        Frequency penalty:
        - 0.0: No penalty (may repeat)
        - 0.5: Moderate penalty
        - 1.0: Strong penalty against repetition
        - 2.0: Maximum penalty
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            frequency_penalty=frequency_penalty,
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content

    def analyze_logprobs(self, prompt: str, top_logprobs: int = 5) -> Dict:
        """
        Generate text and analyze token probabilities (logprobs).

        Logprobs show:
        - How confident the model is about each token
        - Alternative tokens and their probabilities
        - Useful for understanding model behavior
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50,
            logprobs=True,
            top_logprobs=top_logprobs
        )

        return {
            'text': response.choices[0].message.content,
            'logprobs': response.choices[0].logprobs
        }

    def find_optimal_temperature(
        self,
        prompt: str,
        task_type: str
    ) -> float:
        """
        Recommend optimal temperature based on task type.

        Task types:
        - 'factual': Answering factual questions (low temp)
        - 'creative': Creative writing (high temp)
        - 'code': Code generation (low temp)
        - 'conversation': Natural dialogue (medium temp)
        - 'classification': Text classification (very low temp)
        """
        recommendations = {
            'factual': 0.0,
            'creative': 0.9,
            'code': 0.2,
            'conversation': 0.7,
            'classification': 0.0
        }
        return recommendations.get(task_type, 0.7)


def experiment_1_temperature_effects():
    """
    Experiment 1: How temperature affects creativity and consistency.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Temperature Effects")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    explorer = InferenceExplorer(api_key)

    prompt = "Write a creative opening sentence for a science fiction story about time travel."

    temperatures = [0.0, 0.5, 1.0, 1.5]

    results = explorer.compare_temperatures(prompt, temperatures)

    for temp, response in results.items():
        print(f"\n🌡️  Temperature: {temp}")
        print(f"Response: {response}")

    print("\n✅ Observations:")
    print("- Low temperature (0.0): Most predictable and consistent")
    print("- Medium temperature (0.5-0.7): Balanced creativity")
    print("- High temperature (1.0+): More creative but less predictable")


def experiment_2_top_p_sampling():
    """
    Experiment 2: Understanding nucleus sampling (top_p).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Top-P (Nucleus) Sampling")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    explorer = InferenceExplorer(api_key)

    prompt = "Complete this sentence: The most important factor in building reliable software is"

    top_p_values = [0.1, 0.5, 0.9, 1.0]

    for top_p in top_p_values:
        response = explorer.generate_with_top_p(prompt, top_p, temperature=1.0)
        print(f"\n🎯 Top-P: {top_p}")
        print(f"Response: {response}")

    print("\n✅ Observations:")
    print("- Low top_p (0.1): Very focused, picks from top tokens only")
    print("- Medium top_p (0.5): Balanced diversity")
    print("- High top_p (0.9): More diverse but still coherent")


def experiment_3_length_control():
    """
    Experiment 3: Controlling response length with max_tokens.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Response Length Control")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    explorer = InferenceExplorer(api_key)

    prompt = "Explain the concept of machine learning in detail."

    max_tokens_values = [50, 100, 200]

    for max_tokens in max_tokens_values:
        response = explorer.generate_with_max_tokens(prompt, max_tokens)
        print(f"\n📏 Max Tokens: {max_tokens}")
        print(f"Response: {response}")
        print(f"(Approximate word count: {len(response.split())})")

    print("\n✅ Observations:")
    print("- max_tokens controls the maximum response length")
    print("- Responses may be truncated if they exceed the limit")
    print("- Useful for controlling costs and keeping responses concise")


def experiment_4_repetition_penalty():
    """
    Experiment 4: Reducing repetition with frequency_penalty.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Frequency Penalty for Repetition")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    explorer = InferenceExplorer(api_key)

    prompt = "List 10 benefits of regular exercise."

    penalties = [0.0, 0.5, 1.0, 2.0]

    for penalty in penalties:
        response = explorer.generate_with_frequency_penalty(prompt, penalty)
        print(f"\n🔁 Frequency Penalty: {penalty}")
        print(f"Response: {response}")

    print("\n✅ Observations:")
    print("- frequency_penalty=0.0: May repeat words and phrases")
    print("- frequency_penalty=0.5-1.0: Balanced variation")
    print("- frequency_penalty=2.0: Maximum variety, avoids repetition")


def experiment_5_logprobs_analysis():
    """
    Experiment 5: Analyzing token probabilities with logprobs.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Token Probability Analysis")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    explorer = InferenceExplorer(api_key)

    prompt = "The capital of France is"

    result = explorer.analyze_logprobs(prompt, top_logprobs=5)

    print(f"\n📝 Generated Text: {result['text']}")
    print(f"\n🔍 Token Probabilities:")

    if result['logprobs'] and result['logprobs'].content:
        for i, token_data in enumerate(result['logprobs'].content[:5]):  # Show first 5 tokens
            print(f"\n  Token {i+1}: '{token_data.token}'")
            print(f"  Logprob: {token_data.logprob:.4f}")
            print(f"  Probability: {(2.71828 ** token_data.logprob):.4f}")

            if token_data.top_logprobs:
                print(f"  Top alternatives:")
                for alt in token_data.top_logprobs[:3]:
                    prob = 2.71828 ** alt.logprob
                    print(f"    - '{alt.token}': {prob:.4f}")

    print("\n✅ Observations:")
    print("- Logprobs show model confidence for each token")
    print("- High probability = model is confident")
    print("- Multiple alternatives = model is uncertain")
    print("- Useful for debugging and understanding model behavior")


def main():
    """
    Run all inference parameter experiments.
    """
    print("\n🔬 LLM INFERENCE PARAMETERS EXPLORATION\n")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("\nFor Vocareum keys:")
        print('  export OPENAI_API_KEY="voc-..."')
        print("\nFor standard OpenAI keys:")
        print('  export OPENAI_API_KEY="sk-..."')
        return

    # Run all experiments
    experiment_1_temperature_effects()
    experiment_2_top_p_sampling()
    experiment_3_length_control()
    experiment_4_repetition_penalty()
    experiment_5_logprobs_analysis()

    print("\n" + "=" * 70)
    print("✅ All experiments complete!")
    print("=" * 70)

    print("\n📋 Summary of Inference Parameters:")
    print("  • temperature: Controls randomness (0=deterministic, 2=very creative)")
    print("  • top_p: Nucleus sampling for controlled diversity (0.1-1.0)")
    print("  • max_tokens: Limits response length")
    print("  • frequency_penalty: Reduces repetition (0-2)")
    print("  • logprobs: Shows token probabilities for analysis")

    print("\n💡 Best Practices:")
    print("  • Use temperature=0 for factual/classification tasks")
    print("  • Use temperature=0.7 for conversational tasks")
    print("  • Use temperature=0.9+ for creative writing")
    print("  • Combine temperature and top_p carefully (usually set one)")
    print("  • Use frequency_penalty to avoid repetitive responses")
    print("  • Monitor logprobs to understand model confidence")


if __name__ == "__main__":
    main()
