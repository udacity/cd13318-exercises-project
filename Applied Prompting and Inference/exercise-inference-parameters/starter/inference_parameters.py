"""
LLM Inference Parameters Exercise
Lesson 1: Understanding Temperature, Top-P, and Other Parameters

This exercise explores how different inference parameters affect LLM output.
You'll experiment with temperature, top_p, max_tokens, and frequency_penalty
to understand their impact on response quality and creativity.

Learning Objectives:
- Understand how temperature affects randomness and creativity
- Learn when to use different temperature values
- Explore top_p (nucleus sampling) for controlled diversity
- Use max_tokens to control response length
- Apply frequency_penalty to reduce repetition
- Analyze logprobs to understand token probabilities
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
        """
        Initialize the inference explorer.

        Args:
            api_key: OpenAI API key
            model: The model to use for experiments
        """
        # TODO: Initialize the OpenAI client
        self.client = None

        # TODO: Store the model name
        self.model = None

    def generate_with_temperature(self, prompt: str, temperature: float) -> str:
        """
        Generate text with a specific temperature setting.

        Temperature controls randomness:
        - 0.0: Deterministic (always picks most likely token)
        - 0.3-0.5: Focused and consistent
        - 0.7-0.9: Balanced creativity
        - 1.0+: Very creative and unpredictable

        Args:
            prompt: The input prompt
            temperature: Temperature value (0.0 to 2.0)

        Returns:
            Generated text
        """
        # TODO: Make an API call with the specified temperature
        # Use client.chat.completions.create()
        # Set temperature parameter
        # Return the generated text

        pass

    def compare_temperatures(self, prompt: str, temperatures: List[float]) -> Dict[float, str]:
        """
        Compare outputs at different temperature values.

        Args:
            prompt: The input prompt
            temperatures: List of temperature values to test

        Returns:
            Dictionary mapping temperature to generated text
        """
        # TODO: Generate responses for each temperature
        # Store results in a dictionary
        # Return the dictionary

        results = {}
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

        Args:
            prompt: The input prompt
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            temperature: Temperature value

        Returns:
            Generated text
        """
        # TODO: Make an API call with top_p parameter
        # Set both temperature and top_p
        # Return the generated text

        pass

    def generate_with_max_tokens(self, prompt: str, max_tokens: int) -> str:
        """
        Generate text with a maximum token limit.

        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text (may be truncated if limit reached)
        """
        # TODO: Make an API call with max_tokens parameter
        # Set max_tokens to limit response length
        # Return the generated text

        pass

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

        Args:
            prompt: The input prompt
            frequency_penalty: Penalty value (0.0 to 2.0)

        Returns:
            Generated text
        """
        # TODO: Make an API call with frequency_penalty parameter
        # Set frequency_penalty to reduce repetition
        # Return the generated text

        pass

    def analyze_logprobs(self, prompt: str, top_logprobs: int = 5) -> Dict:
        """
        Generate text and analyze token probabilities (logprobs).

        Logprobs show:
        - How confident the model is about each token
        - Alternative tokens and their probabilities
        - Useful for understanding model behavior

        Args:
            prompt: The input prompt
            top_logprobs: Number of alternative tokens to show (1-20)

        Returns:
            Dictionary with response and logprob information
        """
        # TODO: Make an API call with logprobs=True and top_logprobs parameter
        # Extract both the text and logprobs from the response
        # Return a dictionary with:
        #   - 'text': the generated text
        #   - 'logprobs': the logprobs data structure

        pass

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

        Args:
            prompt: The input prompt
            task_type: Type of task

        Returns:
            Recommended temperature value
        """
        # TODO: Implement temperature recommendations based on task type
        # Use this mapping:
        # 'factual' -> 0.0
        # 'creative' -> 0.9
        # 'code' -> 0.2
        # 'conversation' -> 0.7
        # 'classification' -> 0.0

        recommendations = {}
        return recommendations.get(task_type, 0.7)


def experiment_1_temperature_effects():
    """
    Experiment 1: How temperature affects creativity and consistency.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Temperature Effects")
    print("=" * 70)

    # TODO: Initialize the InferenceExplorer
    api_key = os.getenv("OPENAI_API_KEY")
    explorer = None  # Replace with InferenceExplorer(api_key)

    # TODO: Create a prompt that will show temperature effects
    prompt = "Write a creative opening sentence for a science fiction story about"

    # TODO: Test temperatures: 0.0, 0.5, 1.0, 1.5
    temperatures = [0.0, 0.5, 1.0, 1.5]

    # TODO: Use compare_temperatures() to generate responses
    # Print each temperature and its corresponding output

    print("\nObservations:")
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

    # TODO: Initialize the InferenceExplorer
    api_key = os.getenv("OPENAI_API_KEY")
    explorer = None  # Replace with InferenceExplorer(api_key)

    # TODO: Create a prompt
    prompt = "Complete this sentence: The most important factor in building reliable software is"

    # TODO: Test different top_p values: 0.1, 0.5, 0.9, 1.0
    # Use temperature=1.0 to see top_p effects clearly
    top_p_values = [0.1, 0.5, 0.9, 1.0]

    # TODO: Generate and print responses for each top_p value

    print("\nObservations:")
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

    # TODO: Initialize the InferenceExplorer
    api_key = os.getenv("OPENAI_API_KEY")
    explorer = None  # Replace with InferenceExplorer(api_key)

    # TODO: Create a prompt that would generate a long response
    prompt = "Explain the concept of machine learning in detail."

    # TODO: Test different max_tokens: 50, 100, 200
    max_tokens_values = [50, 100, 200]

    # TODO: Generate and print responses for each max_tokens value
    # Note: Responses may be cut off mid-sentence

    print("\nObservations:")
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

    # TODO: Initialize the InferenceExplorer
    api_key = os.getenv("OPENAI_API_KEY")
    explorer = None  # Replace with InferenceExplorer(api_key)

    # TODO: Create a prompt that might lead to repetition
    prompt = "List 10 benefits of regular exercise."

    # TODO: Test frequency penalties: 0.0, 0.5, 1.0, 2.0
    penalties = [0.0, 0.5, 1.0, 2.0]

    # TODO: Generate and print responses for each penalty value

    print("\nObservations:")
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

    # TODO: Initialize the InferenceExplorer
    api_key = os.getenv("OPENAI_API_KEY")
    explorer = None  # Replace with InferenceExplorer(api_key)

    # TODO: Create a simple prompt
    prompt = "The capital of France is"

    # TODO: Use analyze_logprobs() to get probability information
    # Print the generated text and the top alternative tokens

    print("\nObservations:")
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
        return

    # TODO: Run each experiment
    # Uncomment these as you implement them:

    # experiment_1_temperature_effects()
    # experiment_2_top_p_sampling()
    # experiment_3_length_control()
    # experiment_4_repetition_penalty()
    # experiment_5_logprobs_analysis()

    print("\n" + "=" * 70)
    print("✅ All experiments complete!")
    print("=" * 70)

    print("\n📋 Summary of Inference Parameters:")
    print("  • temperature: Controls randomness (0=deterministic, 2=very creative)")
    print("  • top_p: Nucleus sampling for controlled diversity (0.1-1.0)")
    print("  • max_tokens: Limits response length")
    print("  • frequency_penalty: Reduces repetition (0-2)")
    print("  • logprobs: Shows token probabilities for analysis")


if __name__ == "__main__":
    main()
