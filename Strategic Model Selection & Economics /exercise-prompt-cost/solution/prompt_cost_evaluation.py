# Prompt Engineering and Cost Evaluation Script
# This script demonstrates how to optimize prompts for effectiveness and cost efficiency
# It compares different prompt strategies and analyzes their cost-performance trade-offs

import openai
from openai import OpenAI
import pandas as pd
import time
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Configuration dictionary defining different prompt strategies for cost optimization
# Each configuration balances effectiveness with cost considerations
PROMPT_CONFIGS = {
    # Minimal prompt strategy - lowest cost but potentially less effective
    "minimal": {
        "model": "gpt-4o-mini",  # Most cost-effective model
        "temperature": 0.7,      # Balanced temperature for consistent results
        "max_tokens": 150,       # Limited tokens to control costs
        "description": "Minimal prompts with cost-effective model for budget optimization"
    },
    # Standard prompt strategy - balanced approach
    "standard": {
        "model": "gpt-4o",       # Standard model with good performance
        "temperature": 0.7,      # Balanced temperature
        "max_tokens": 300,       # Moderate token limit
        "description": "Standard prompts with balanced cost-performance ratio"
    },
    # Premium prompt strategy - highest quality but most expensive
    "premium": {
        "model": "gpt-4.1",       # High-performance model
        "temperature": 0.5,      # Lower temperature for consistency
        "max_tokens": 500,       # Higher token limit for detailed responses
        "description": "Detailed prompts with premium model for maximum effectiveness"
    }
}

# Pricing information for cost calculations (as of 2024 - update as needed)
# Prices are per 1K tokens in USD
MODEL_PRICING = {
    "gpt-4o-mini": {
        "input": 0.00015,   # $0.15 per 1M input tokens
        "output": 0.0006    # $0.60 per 1M output tokens
    },
    "gpt-4o": {
        "input": 0.0025,    # $2.50 per 1M input tokens
        "output": 0.01      # $10.00 per 1M output tokens
    },
    "gpt-4.1": {
        "input": 0.003,      # $3.00 per 1M input tokens
        "output": 0.012      # $12.00 per 1M output tokens
    }
}

# Test prompts designed to evaluate different prompt engineering strategies
# Each category tests how prompt complexity affects output quality and cost
PROMPT_STRATEGIES = [
    {
        "category": "task_completion",
        "minimal": {
            "prompt": "Summarize this text: [TEXT_PLACEHOLDER]",
            "description": "Basic instruction without context or examples"
        },
        "standard": {
            "prompt": "Please provide a concise summary of the following text, focusing on the main points and key takeaways: [TEXT_PLACEHOLDER]",
            "description": "Clear instruction with specific guidance"
        },
        "premium": {
            "prompt": "You are an expert content analyst. Please provide a comprehensive summary of the following text. Focus on: 1) Main arguments or points, 2) Supporting evidence, 3) Key conclusions. Format your response with clear headings and bullet points for easy reading: [TEXT_PLACEHOLDER]",
            "description": "Detailed instruction with role, structure, and formatting requirements"
        }
    },
    {
        "category": "creative_writing",
        "minimal": {
            "prompt": "Write a story about a robot.",
            "description": "Simple creative prompt without constraints"
        },
        "standard": {
            "prompt": "Write a short story (200-300 words) about a robot who discovers emotions for the first time. Include dialogue and describe the robot's internal experience.",
            "description": "Structured creative prompt with length and content specifications"
        },
        "premium": {
            "prompt": "You are a skilled science fiction author. Write a compelling short story (200-300 words) about a robot who discovers emotions for the first time. Requirements: 1) Include meaningful dialogue between characters, 2) Show the robot's emotional journey through actions and internal thoughts, 3) Create a satisfying narrative arc with beginning, middle, and end, 4) Use vivid, descriptive language to engage the reader. Focus on the contrast between the robot's logical programming and newfound emotional experiences.",
            "description": "Comprehensive creative prompt with role-playing, detailed requirements, and quality guidelines"
        }
    },
    {
        "category": "problem_solving",
        "minimal": {
            "prompt": "How do I reduce customer churn?",
            "description": "Direct question without context"
        },
        "standard": {
            "prompt": "I'm running a SaaS business and experiencing 15% monthly customer churn. What are the most effective strategies to reduce customer churn and improve retention?",
            "description": "Contextualized question with specific details"
        },
        "premium": {
            "prompt": "You are a business consultant specializing in customer retention. I'm running a SaaS business with the following metrics: 15% monthly churn rate, $50 average monthly revenue per user, 6-month average customer lifetime. Please provide: 1) Root cause analysis of potential churn drivers, 2) Specific, actionable strategies to reduce churn, 3) Implementation timeline and resource requirements, 4) Expected ROI and success metrics. Prioritize strategies by impact and feasibility.",
            "description": "Expert consultation prompt with detailed context, specific deliverables, and structured output requirements"
        }
    }
]

# Sample text for testing summarization prompts
SAMPLE_TEXT = """
Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century, fundamentally reshaping industries, economies, and societies worldwide. From healthcare and finance to transportation and entertainment, AI applications are revolutionizing how we work, communicate, and solve complex problems.

The current AI landscape is dominated by machine learning techniques, particularly deep learning neural networks that can process vast amounts of data to identify patterns and make predictions. Large Language Models (LLMs) like GPT-4 have demonstrated remarkable capabilities in natural language understanding and generation, enabling applications from automated customer service to creative writing assistance.

However, the rapid advancement of AI also presents significant challenges. Concerns about job displacement, algorithmic bias, privacy, and the concentration of AI capabilities in the hands of a few large corporations have sparked important debates about regulation and governance. Additionally, the environmental impact of training large AI models has raised questions about sustainability in AI development.

Looking forward, the integration of AI into everyday life will likely accelerate, with developments in areas such as autonomous vehicles, personalized medicine, and smart cities. Success in navigating this AI-driven future will require thoughtful consideration of both the tremendous opportunities and the substantial risks that these technologies present.
"""

def call_openai_api(prompt: str, config: Dict) -> Dict:
    """
    Make API call to OpenAI and capture comprehensive response metrics including cost analysis.
    
    This function extends basic API calling to include detailed cost tracking and 
    performance metrics essential for prompt optimization and budget management.
    
    Args:
        prompt (str): The input prompt to send to the model
        config (Dict): Model configuration including model name, temperature, etc.
    
    Returns:
        Dict: Response data including content, performance metrics, cost analysis, and error handling
    """
    print(f"  🔄 Calling {config['model']} (temp: {config['temperature']}, max_tokens: {config['max_tokens']})...")
    start_time = time.time()  # Start timing the API call
    
    try:
        # Initialize OpenAI client with API key
        # Note: In production, use environment variables for API keys
        client = OpenAI(api_key="your-key-here")
        
        # Make the API call with specified configuration
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=config["temperature"],
            max_completion_tokens=config["max_tokens"],
            logprobs=True
        )
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Calculate cost based on token usage and model pricing
        model_name = config["model"]
        if model_name in MODEL_PRICING:
            input_cost = (response.usage.prompt_tokens / 1000) * MODEL_PRICING[model_name]["input"]
            output_cost = (response.usage.completion_tokens / 1000) * MODEL_PRICING[model_name]["output"]
            total_cost = input_cost + output_cost
        else:
            input_cost = output_cost = total_cost = 0.0
        
        # Structure the successful response with all relevant metrics
        result = {
            "response": response.choices[0].message.content,
            "latency_ms": round(latency, 2),
            "tokens_used": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "cost_per_token": round(total_cost / response.usage.total_tokens, 8) if response.usage.total_tokens > 0 else 0,
            "success": True,
            "error": None
        }
        
        print(f"  ✅ Success! Latency: {result['latency_ms']}ms, Tokens: {result['tokens_used']}, Cost: ${result['total_cost']:.6f}")
        return result
        
    except Exception as e:
        # Handle API errors gracefully and return structured error response
        print(f"Error: {str(e)}")
        return {
            "response": None, "latency_ms": None, "tokens_used": None,
            "prompt_tokens": None, "completion_tokens": None,
            "input_cost": 0, "output_cost": 0, "total_cost": 0, "cost_per_token": 0,
            "success": False, "error": str(e)
        }

def score_response_quality(response: str, category: str) -> int:
    """
    Evaluate the quality of responses on a 1-5 scale based on task category.
    
    This function implements category-specific scoring logic to assess how well
    the model performed the requested task, considering factors like completeness,
    accuracy, creativity, and adherence to instructions.
    
    Args:
        response (str): The model's response to evaluate
        category (str): The task category (task_completion, creative_writing, problem_solving)
    
    Returns:
        int: Score from 1-5 where 5 is excellent quality, 1 is poor quality
    """
    if not response:
        return 0  # No response provided
        
    response_lower = response.lower()
    word_count = len(response.split())
    
    # Task completion scoring: Focus on completeness and structure
    if category == "task_completion":
        # Check for summary indicators and structure
        summary_indicators = ["main", "key", "important", "summary", "conclusion", "points"]
        structure_indicators = ["first", "second", "finally", "additionally", "furthermore"]
        
        summary_score = sum(1 for indicator in summary_indicators if indicator in response_lower)
        structure_score = sum(1 for indicator in structure_indicators if indicator in response_lower)
        
        # Base score on content indicators and length appropriateness
        if summary_score >= 3 and word_count >= 50:
            return 5  # Excellent: comprehensive summary with good structure
        elif summary_score >= 2 and word_count >= 30:
            return 4  # Good: adequate summary with some structure
        elif summary_score >= 1 and word_count >= 20:
            return 3  # Fair: basic summary
        elif word_count >= 10:
            return 2  # Poor: minimal content
        else:
            return 1  # Very poor: insufficient content
            
    # Creative writing scoring: Focus on narrative elements and creativity
    elif category == "creative_writing":
        # Check for story elements
        narrative_elements = ["robot", "emotion", "feel", "discover", "experience", "thought"]
        dialogue_indicators = ['"', "'", "said", "asked", "replied", "exclaimed"]
        descriptive_words = ["suddenly", "slowly", "carefully", "bright", "dark", "strange", "wonderful"]
        
        narrative_score = sum(1 for element in narrative_elements if element in response_lower)
        dialogue_score = sum(1 for indicator in dialogue_indicators if indicator in response)
        descriptive_score = sum(1 for word in descriptive_words if word in response_lower)
        
        # Evaluate story completeness and creativity
        if narrative_score >= 4 and dialogue_score >= 2 and word_count >= 150:
            return 5  # Excellent: complete story with dialogue and good length
        elif narrative_score >= 3 and word_count >= 100:
            return 4  # Good: solid story with adequate development
        elif narrative_score >= 2 and word_count >= 50:
            return 3  # Fair: basic story elements present
        elif narrative_score >= 1:
            return 2  # Poor: minimal story development
        else:
            return 1  # Very poor: no clear story structure
            
    # Problem solving scoring: Focus on actionable advice and comprehensiveness
    elif category == "problem_solving":
        # Check for business strategy elements
        strategy_words = ["strategy", "approach", "solution", "recommend", "implement", "improve"]
        analysis_words = ["analyze", "identify", "cause", "reason", "factor", "metric"]
        action_words = ["action", "step", "plan", "timeline", "measure", "track"]
        
        strategy_score = sum(1 for word in strategy_words if word in response_lower)
        analysis_score = sum(1 for word in analysis_words if word in response_lower)
        action_score = sum(1 for word in action_words if word in response_lower)
        
        # Evaluate comprehensiveness and actionability
        total_score = strategy_score + analysis_score + action_score
        if total_score >= 6 and word_count >= 100:
            return 5  # Excellent: comprehensive analysis with actionable strategies
        elif total_score >= 4 and word_count >= 75:
            return 4  # Good: solid advice with some analysis
        elif total_score >= 2 and word_count >= 50:
            return 3  # Fair: basic recommendations
        elif total_score >= 1:
            return 2  # Poor: minimal useful content
        else:
            return 1  # Very poor: no actionable advice
    
    return 3  # Default middle score for unrecognized categories

def calculate_cost_effectiveness(quality_score: int, total_cost: float) -> float:
    """
    Calculate cost-effectiveness ratio for prompt strategies.
    
    This metric helps determine which prompt approach provides the best
    value by balancing quality output with cost efficiency.
    
    Args:
        quality_score (int): Quality score from 1-5
        total_cost (float): Total cost in USD for the API call
    
    Returns:
        float: Cost-effectiveness ratio (higher is better)
    """
    if total_cost == 0:
        return 0.0
    
    # Calculate quality points per dollar spent
    # Multiply by 1000 to get a more readable number
    return (quality_score / total_cost) * 1000

def test_prompt_strategy(category: str, strategy_type: str):
    """
    Test a specific prompt strategy and analyze its cost-effectiveness.
    
    This function demonstrates how to systematically evaluate different prompt
    approaches, measuring both output quality and cost efficiency to optimize
    prompt engineering decisions.
    
    Args:
        category (str): The task category to test
        strategy_type (str): The prompt strategy type (minimal, standard, premium)
    
    Returns:
        dict: Results including quality scores, costs, and effectiveness metrics
    """
    # Find the specified strategy
    strategy_data = None
    for strategy in PROMPT_STRATEGIES:
        if strategy["category"] == category:
            strategy_data = strategy
            break
    
    if not strategy_data:
        print(f"❌ Category '{category}' not found")
        return None
    
    prompt_info = strategy_data[strategy_type]
    config = PROMPT_CONFIGS[strategy_type]
    
    print(f"\n🧪 TESTING PROMPT STRATEGY: {category.upper()} - {strategy_type.upper()}")
    print(f"Description: {prompt_info['description']}")
    print(f"Model: {config['model']} | Temp: {config['temperature']} | Max Tokens: {config['max_tokens']}")
    print("=" * 80)
    
    # Prepare the prompt (replace placeholder if needed)
    prompt = prompt_info["prompt"]
    if "[TEXT_PLACEHOLDER]" in prompt:
        prompt = prompt.replace("[TEXT_PLACEHOLDER]", SAMPLE_TEXT)
    
    # Make API call with current configuration
    result = call_openai_api(prompt, config)
    
    if result['success']:

        print(result)
        # Score the response quality
        quality_score = score_response_quality(result['response'], category)
        
        # Calculate cost-effectiveness
        cost_effectiveness = calculate_cost_effectiveness(quality_score, result['total_cost'])
        
        # Display results with clear formatting
        print(f"\n📝 RESPONSE:")
        print("-" * 60)
        print(result['response'])
        print("-" * 60)
        
        print(f"\n📊 METRICS:")
        print(f"✅ Quality Score: {quality_score}/5")
        print(f"⏱️  Latency: {result['latency_ms']}ms")
        print(f"🔢 Tokens Used: {result['tokens_used']} (Input: {result['prompt_tokens']}, Output: {result['completion_tokens']})")
        print(f"💰 Cost Breakdown:")
        print(f"   Input Cost: ${result['input_cost']:.6f}")
        print(f"   Output Cost: ${result['output_cost']:.6f}")
        print(f"   Total Cost: ${result['total_cost']:.6f}")
        print(f"   Cost per Token: ${result['cost_per_token']:.8f}")
        print(f"📈 Cost-Effectiveness: {cost_effectiveness:.2f} quality points per $1000")
        
        # Store results for comparison
        return {
            'category': category,
            'strategy_type': strategy_type,
            'response': result['response'],
            'quality_score': quality_score,
            'latency_ms': result['latency_ms'],
            'tokens_used': result['tokens_used'],
            'prompt_tokens': result['prompt_tokens'],
            'completion_tokens': result['completion_tokens'],
            'total_cost': result['total_cost'],
            'cost_effectiveness': cost_effectiveness,
            'model': config['model']
        }
    else:
        print(f"❌ Failed: {result['error']}")
        return None

def compare_prompt_strategies(category: str):
    """
    Compare all prompt strategies for a given category and analyze trade-offs.
    
    This function provides a comprehensive comparison of minimal, standard, and premium
    prompt approaches, helping users understand the cost-quality trade-offs and make
    informed decisions about prompt engineering strategies.
    
    Args:
        category (str): The task category to compare strategies for
    
    Returns:
        dict: Comparison results with recommendations
    """
    print(f"\n🔍 COMPARING PROMPT STRATEGIES FOR: {category.upper()}")
    print("=" * 80)
    
    results = {}
    strategy_types = ["minimal", "standard", "premium"]
    
    # Test each strategy type
    for strategy_type in strategy_types:
        result = test_prompt_strategy(category, strategy_type)
        if result:
            results[strategy_type] = result
        
        # Add a small delay between API calls to avoid rate limiting
        time.sleep(1)
    
    # Analyze and display comparison
    if len(results) > 1:
        print(f"\n📊 STRATEGY COMPARISON SUMMARY:")
        print("=" * 80)
        
        # Create comparison table
        comparison_data = []
        for strategy_type, data in results.items():
            comparison_data.append({
                'Strategy': strategy_type.capitalize(),
                'Quality': f"{data['quality_score']}/5",
                'Cost': f"${data['total_cost']:.6f}",
                'Tokens': data['tokens_used'],
                'Cost-Effectiveness': f"{data['cost_effectiveness']:.2f}",
                'Model': data['model']
            })
        
        # Display comparison table
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Provide recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        # Find best cost-effectiveness
        best_ce = max(results.values(), key=lambda x: x['cost_effectiveness'])
        print(f"🏆 Best Cost-Effectiveness: {best_ce['strategy_type'].capitalize()} strategy")
        print(f"   Quality: {best_ce['quality_score']}/5, Cost: ${best_ce['total_cost']:.6f}")
        
        # Find highest quality
        best_quality = max(results.values(), key=lambda x: x['quality_score'])
        print(f"⭐ Highest Quality: {best_quality['strategy_type'].capitalize()} strategy")
        print(f"   Quality: {best_quality['quality_score']}/5, Cost: ${best_quality['total_cost']:.6f}")
        
        # Find lowest cost
        lowest_cost = min(results.values(), key=lambda x: x['total_cost'])
        print(f"💰 Lowest Cost: {lowest_cost['strategy_type'].capitalize()} strategy")
        print(f"   Quality: {lowest_cost['quality_score']}/5, Cost: ${lowest_cost['total_cost']:.6f}")
        
        return results
    else:
        print("❌ Insufficient results for comparison")
        return results

# Example usage: Compare strategies for task completion
# Uncomment the line below to run a comparison
task_completion_results = compare_prompt_strategies("task_completion")

# Additional examples you can run:
# creative_writing_results = compare_prompt_strategies("creative_writing")
# problem_solving_results = compare_prompt_strategies("problem_solving")

# Single strategy tests:
# minimal_summary = test_prompt_strategy("task_completion", "minimal")
# premium_story = test_prompt_strategy("creative_writing", "premium")
# standard_business = test_prompt_strategy("problem_solving", "standard")
