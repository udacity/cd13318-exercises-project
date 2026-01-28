"""
Prompt Engineering Exercise - SOLUTION
Chain-of-Thought and Advanced Prompting Techniques

Complete implementation of prompt engineering techniques.
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
        """Initialize the prompt engineer."""
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=api_key
        )
        self.model = model

    def zero_shot_prompt(self, task: str) -> str:
        """
        Generate response using zero-shot prompting (no examples).
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": task}],
            temperature=0  # Use 0 for consistency
        )
        return response.choices[0].message.content

    def few_shot_prompt(self, task: str, examples: List[Dict[str, str]]) -> str:
        """
        Generate response using few-shot prompting (with examples).
        """
        # Build prompt with examples
        prompt = "Here are some examples:\n\n"

        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"

        prompt += f"Now, complete this task:\n{task}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    def chain_of_thought_prompt(self, problem: str) -> str:
        """
        Generate response using chain-of-thought (CoT) prompting.
        """
        # Add CoT instruction
        cot_prompt = f"{problem}\n\nLet's think step by step and show the reasoning:"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": cot_prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    def structured_output_prompt(
        self,
        task: str,
        output_format: str
    ) -> str:
        """
        Generate response in a specific structured format.
        """
        format_instructions = {
            "JSON": "Respond in valid JSON format. Use proper JSON syntax with quoted keys.",
            "markdown table": "Respond as a markdown table with headers and properly aligned columns.",
            "YAML": "Respond in valid YAML format.",
            "CSV": "Respond in CSV format with headers."
        }

        instruction = format_instructions.get(output_format, f"Respond in {output_format} format.")

        prompt = f"{task}\n\n{instruction}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    def compare_approaches(
        self,
        problem: str,
        examples: List[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Compare zero-shot, few-shot, and chain-of-thought approaches.
        """
        results = {}

        # Zero-shot
        results['zero_shot'] = self.zero_shot_prompt(problem)

        # Few-shot (if examples provided)
        if examples:
            results['few_shot'] = self.few_shot_prompt(problem, examples)
        else:
            results['few_shot'] = "No examples provided"

        # Chain-of-thought
        results['chain_of_thought'] = self.chain_of_thought_prompt(problem)

        return results


def experiment_1_zero_shot_vs_few_shot():
    """
    Experiment 1: Compare zero-shot and few-shot learning.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Zero-Shot vs Few-Shot Learning")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    engineer = PromptEngineer(api_key)

    task = "Classify the sentiment of this review: 'The product is okay, but shipping was slow.'"

    # Zero-shot
    print("\n📝 Zero-Shot Approach:")
    zero_shot_result = engineer.zero_shot_prompt(task)
    print(f"Result: {zero_shot_result}")

    # Few-shot with examples
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

    print("\n📚 Few-Shot Approach (with 3 examples):")
    few_shot_result = engineer.few_shot_prompt(task, examples)
    print(f"Result: {few_shot_result}")

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

    api_key = os.getenv("OPENAI_API_KEY")
    engineer = PromptEngineer(api_key)

    problem = """A bakery sells cupcakes for $3 each. If they offer a 20% discount on
orders of 10 or more, and a customer orders 15 cupcakes, what is the total cost?"""

    # Without CoT
    print("\n📝 Without Chain-of-Thought:")
    regular_response = engineer.zero_shot_prompt(problem)
    print(f"Result: {regular_response}")

    # With CoT
    print("\n🧠 With Chain-of-Thought:")
    cot_response = engineer.chain_of_thought_prompt(problem)
    print(f"Result: {cot_response}")

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

    api_key = os.getenv("OPENAI_API_KEY")
    engineer = PromptEngineer(api_key)

    task = """Extract the following information from this text:
'John Smith, age 35, lives in Seattle and works as a software engineer at TechCorp.'

Extract: name, age, city, occupation, company"""

    # Request JSON format
    print("\n📊 Requesting JSON Format:")
    json_response = engineer.structured_output_prompt(task, "JSON")
    print(f"Result:\n{json_response}")

    # Try to parse as JSON
    try:
        parsed = json.loads(json_response)
        print("\n✅ Successfully parsed as JSON!")
        print(f"Parsed data: {json.dumps(parsed, indent=2)}")
    except json.JSONDecodeError:
        print("\n⚠️ Response is not valid JSON, but contains the information")

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

    api_key = os.getenv("OPENAI_API_KEY")
    engineer = PromptEngineer(api_key)

    problem = """A train travels 120 miles in 2 hours, then stops for 30 minutes,
then travels another 90 miles in 1.5 hours. What is the train's average speed
for the entire journey (including the stop)?"""

    examples = [
        {
            "input": "A car travels 60 miles in 1 hour. What is its speed?",
            "output": "Speed = Distance / Time = 60 miles / 1 hour = 60 mph"
        }
    ]

    # Compare all approaches
    results = engineer.compare_approaches(problem, examples)

    print("\n1️⃣ Zero-Shot Approach:")
    print(f"{results['zero_shot']}")

    print("\n2️⃣ Few-Shot Approach (with example):")
    print(f"{results['few_shot']}")

    print("\n3️⃣ Chain-of-Thought Approach:")
    print(f"{results['chain_of_thought']}")

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

    api_key = os.getenv("OPENAI_API_KEY")
    engineer = PromptEngineer(api_key)

    text = """
    Product Review:
    I purchased the UltraBook Pro 15" laptop last month for $1,299.
    The performance is excellent with its Intel i7 processor and 16GB RAM.
    Battery life is impressive - easily 10 hours of normal use.
    However, the keyboard feels a bit mushy and the trackpad could be more responsive.
    Overall rating: 4/5 stars.
    Would I recommend? Yes, especially for productivity work.
    """

    task = f"""Extract product information from this review in JSON format.
Include: product_name, price, pros (list), cons (list), rating, recommendation.

Review: {text}"""

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

    print("\n🔍 Extracting structured data from review...")
    result = engineer.few_shot_prompt(task, examples)
    print(f"\nResult:\n{result}")

    # Try to parse as JSON
    try:
        # Extract JSON from response (may be wrapped in markdown code blocks)
        json_str = result
        if "```json" in result:
            json_str = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            json_str = result.split("```")[1].split("```")[0].strip()

        parsed = json.loads(json_str)
        print("\n✅ Successfully parsed as JSON!")
        print(f"Extracted data: {json.dumps(parsed, indent=2)}")
    except (json.JSONDecodeError, IndexError) as e:
        print(f"\n⚠️ Could not parse as JSON: {e}")
        print("But the information is extracted in the response above")

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
        print("\nFor Vocareum keys:")
        print('  export OPENAI_API_KEY="voc-..."')
        print("\nFor standard OpenAI keys:")
        print('  export OPENAI_API_KEY="sk-..."')
        return

    # Run all experiments
    experiment_1_zero_shot_vs_few_shot()
    experiment_2_chain_of_thought()
    experiment_3_structured_output()
    experiment_4_comparison()
    experiment_5_real_world_application()

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

    print("\n🎓 Best Practices:")
    print("  • Start with zero-shot, add examples only if needed")
    print("  • Use CoT for any multi-step reasoning task")
    print("  • Provide 2-5 examples (not just 1, not more than 10)")
    print("  • Be specific about output format requirements")
    print("  • Combine techniques for complex real-world tasks")


if __name__ == "__main__":
    main()
