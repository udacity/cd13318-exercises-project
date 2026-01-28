"""
Prompt Engineering Exercise
Chain-of-Thought and Advanced Prompting Techniques

This exercise explores how prompt engineering techniques affect LLM reasoning
and output quality. You'll experiment with chain-of-thought prompting, few-shot
learning, and structured output formats.

Learning Objectives:
- Understand chain-of-thought (CoT) prompting for complex reasoning
- Apply few-shot learning with examples
- Structure prompts for consistent output formats
- Compare zero-shot vs few-shot vs chain-of-thought approaches
- Learn when to use each prompting technique
"""

from openai import OpenAI
import os
from typing import List, Dict
import json


class PromptEngineer:
    """
    A tool for exploring and comparing different prompt engineering techniques.
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the prompt engineer.

        Args:
            api_key: OpenAI API key
            model: The model to use for experiments
        """
        # TODO: Initialize the OpenAI client
        self.client = None

        # TODO: Store the model name
        self.model = None

    def zero_shot_prompt(self, task: str) -> str:
        """
        Generate response using zero-shot prompting (no examples).

        Zero-shot:
        - No examples provided
        - Relies on model's pre-training
        - Works well for simple, common tasks
        - May struggle with complex or specific tasks

        Args:
            task: The task description

        Returns:
            Generated response
        """
        # TODO: Make an API call with just the task description
        # Use temperature=0 for consistency
        # Return the response

        pass

    def few_shot_prompt(self, task: str, examples: List[Dict[str, str]]) -> str:
        """
        Generate response using few-shot prompting (with examples).

        Few-shot:
        - Provides 2-5 examples
        - Demonstrates desired format and style
        - Improves consistency and accuracy
        - Useful for specific formats or patterns

        Args:
            task: The task description
            examples: List of example input-output pairs

        Returns:
            Generated response
        """
        # TODO: Build a prompt that includes examples
        # Format: "Input: X\nOutput: Y\n\n" for each example
        # Then add the actual task
        # Make API call and return response

        pass

    def chain_of_thought_prompt(self, problem: str) -> str:
        """
        Generate response using chain-of-thought (CoT) prompting.

        Chain-of-thought:
        - Instructs model to show reasoning steps
        - Dramatically improves accuracy on complex problems
        - Useful for math, logic, multi-step reasoning
        - Add "Let's think step by step" or "Show your work"

        Args:
            problem: The problem to solve

        Returns:
            Response with reasoning steps
        """
        # TODO: Add chain-of-thought instruction to the problem
        # Common phrases:
        # - "Let's think step by step."
        # - "Show your reasoning."
        # - "Explain your thought process."

        # Make API call with the enhanced prompt
        # Return the response with reasoning

        pass

    def structured_output_prompt(
        self,
        task: str,
        output_format: str
    ) -> str:
        """
        Generate response in a specific structured format.

        Structured output:
        - JSON, YAML, markdown tables, etc.
        - Ensures consistent parsing
        - Useful for integrating with code
        - Provide clear format examples

        Args:
            task: The task description
            output_format: Desired output format (e.g., "JSON", "markdown table")

        Returns:
            Response in specified format
        """
        # TODO: Create a prompt that specifies the output format
        # Include format instructions and example structure
        # Make API call and return formatted response

        pass

    def compare_approaches(
        self,
        problem: str,
        examples: List[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Compare zero-shot, few-shot, and chain-of-thought approaches.

        Args:
            problem: The problem to solve
            examples: Optional examples for few-shot

        Returns:
            Dictionary with results from each approach
        """
        # TODO: Generate responses using all three approaches:
        # 1. Zero-shot (no examples, no CoT)
        # 2. Few-shot (with examples if provided)
        # 3. Chain-of-thought (with reasoning steps)

        # Return dictionary with results

        results = {
            'zero_shot': "",
            'few_shot': "",
            'chain_of_thought': ""
        }
        return results


def experiment_1_zero_shot_vs_few_shot():
    """
    Experiment 1: Compare zero-shot and few-shot learning.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Zero-Shot vs Few-Shot Learning")
    print("=" * 70)

    # TODO: Initialize PromptEngineer
    api_key = os.getenv("OPENAI_API_KEY")
    engineer = None  # Replace with PromptEngineer(api_key)

    # TODO: Define a classification task
    task = "Classify the sentiment of this review: 'The product is okay, but shipping was slow.'"

    # TODO: Generate zero-shot response
    # zero_shot_result = engineer.zero_shot_prompt(task)

    # TODO: Define examples for few-shot
    examples = [
        {
            "input": "This product exceeded my expectations!",
            "output": "Positive"
        },
        {
            "input": "Terrible quality, wouldn't recommend.",
            "output": "Negative"
        },
        {
            "input": "It's fine, nothing special.",
            "output": "Neutral"
        }
    ]

    # TODO: Generate few-shot response
    # few_shot_result = engineer.few_shot_prompt(task, examples)

    # TODO: Print and compare results

    print("\n✅ Observations:")
    print("- Zero-shot: Works for common tasks but may be inconsistent")
    print("- Few-shot: More consistent format and better accuracy")
    print("- Few-shot is especially useful for specific output formats")


def experiment_2_chain_of_thought():
    """
    Experiment 2: Chain-of-thought for complex reasoning.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Chain-of-Thought Reasoning")
    print("=" * 70)

    # TODO: Initialize PromptEngineer
    api_key = os.getenv("OPENAI_API_KEY")
    engineer = None  # Replace with PromptEngineer(api_key)

    # TODO: Define a multi-step math problem
    problem = """A bakery sells cupcakes for $3 each. If they offer a 20% discount on
orders of 10 or more, and a customer orders 15 cupcakes, what is the total cost?"""

    # TODO: Generate response without CoT
    # regular_response = engineer.zero_shot_prompt(problem)

    # TODO: Generate response with CoT
    # cot_response = engineer.chain_of_thought_prompt(problem)

    # TODO: Print and compare results

    print("\n✅ Observations:")
    print("- Without CoT: May give answer without showing work")
    print("- With CoT: Shows step-by-step reasoning")
    print("- CoT improves accuracy on complex problems")
    print("- CoT responses are more trustworthy (can verify logic)")


def experiment_3_structured_output():
    """
    Experiment 3: Getting structured output formats.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Structured Output Formats")
    print("=" * 70)

    # TODO: Initialize PromptEngineer
    api_key = os.getenv("OPENAI_API_KEY")
    engineer = None  # Replace with PromptEngineer(api_key)

    # TODO: Define a task that requires structured output
    task = """Extract the following information from this text:
'John Smith, age 35, lives in Seattle and works as a software engineer at TechCorp.'

Extract: name, age, city, occupation, company"""

    # TODO: Request JSON format
    # json_response = engineer.structured_output_prompt(task, "JSON")

    # TODO: Print the response
    # Try to parse it as JSON to verify format

    print("\n✅ Observations:")
    print("- Structured output is easier to parse programmatically")
    print("- Specify exact format (JSON keys, markdown headers, etc.)")
    print("- Provide an example structure for best results")


def experiment_4_comparison():
    """
    Experiment 4: Side-by-side comparison of all approaches.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Comprehensive Comparison")
    print("=" * 70)

    # TODO: Initialize PromptEngineer
    api_key = os.getenv("OPENAI_API_KEY")
    engineer = None  # Replace with PromptEngineer(api_key)

    # TODO: Define a moderately complex problem
    problem = """A train travels 120 miles in 2 hours, then stops for 30 minutes,
then travels another 90 miles in 1.5 hours. What is the train's average speed
for the entire journey (including the stop)?"""

    # TODO: Define examples for few-shot
    examples = [
        {
            "input": "A car travels 60 miles in 1 hour. What is its speed?",
            "output": "Speed = Distance / Time = 60 miles / 1 hour = 60 mph"
        }
    ]

    # TODO: Use compare_approaches() to get all three results
    # results = engineer.compare_approaches(problem, examples)

    # TODO: Print all three results for comparison

    print("\n✅ Observations:")
    print("- Different approaches work better for different tasks")
    print("- Chain-of-thought is best for multi-step reasoning")
    print("- Few-shot is best for specific formats or styles")
    print("- Zero-shot is fastest but least reliable for complex tasks")


def experiment_5_real_world_application():
    """
    Experiment 5: Real-world prompt engineering for data extraction.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Real-World Data Extraction")
    print("=" * 70)

    # TODO: Initialize PromptEngineer
    api_key = os.getenv("OPENAI_API_KEY")
    engineer = None  # Replace with PromptEngineer(api_key)

    # TODO: Define a real-world text to extract from
    text = """
    Product Review:
    I purchased the UltraBook Pro 15" laptop last month for $1,299.
    The performance is excellent with its Intel i7 processor and 16GB RAM.
    Battery life is impressive - easily 10 hours of normal use.
    However, the keyboard feels a bit mushy and the trackpad could be more responsive.
    Overall rating: 4/5 stars.
    Would I recommend? Yes, especially for productivity work.
    """

    # TODO: Create a task to extract structured information
    task = f"""Extract product information from this review in JSON format.
Include: product_name, price, pros (list), cons (list), rating, recommendation.

Review: {text}"""

    # TODO: Use few-shot with examples of desired format
    examples = [
        {
            "input": "Review: The Widget 3000 costs $50. It's durable but heavy. Rating: 3/5.",
            "output": json.dumps({
                "product_name": "Widget 3000",
                "price": 50,
                "pros": ["durable"],
                "cons": ["heavy"],
                "rating": 3,
                "recommendation": None
            }, indent=2)
        }
    ]

    # TODO: Generate structured extraction
    # result = engineer.few_shot_prompt(task, examples)

    # TODO: Print and attempt to parse as JSON

    print("\n✅ Observations:")
    print("- Combining techniques (few-shot + structured output) is powerful")
    print("- Clear format specifications reduce parsing errors")
    print("- Real-world applications often need multiple techniques")


def main():
    """
    Run all prompt engineering experiments.
    """
    print("\n🎯 PROMPT ENGINEERING EXPLORATION\n")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    # TODO: Run each experiment
    # Uncomment these as you implement them:

    # experiment_1_zero_shot_vs_few_shot()
    # experiment_2_chain_of_thought()
    # experiment_3_structured_output()
    # experiment_4_comparison()
    # experiment_5_real_world_application()

    print("\n" + "=" * 70)
    print("✅ All experiments complete!")
    print("=" * 70)

    print("\n📋 Summary of Prompt Engineering Techniques:")
    print("  • Zero-shot: No examples, relies on pre-training")
    print("  • Few-shot: Provide 2-5 examples for consistency")
    print("  • Chain-of-thought: Request step-by-step reasoning")
    print("  • Structured output: Specify exact format (JSON, etc.)")

    print("\n💡 When to Use Each Technique:")
    print("  • Zero-shot: Simple, common tasks")
    print("  • Few-shot: Specific formats, domain-specific tasks")
    print("  • Chain-of-thought: Math, logic, multi-step reasoning")
    print("  • Structured output: Data extraction, API integration")


if __name__ == "__main__":
    main()
