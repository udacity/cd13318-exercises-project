# Prompt Engineering and Cost Evaluation Exercise - Starter Template
# TODO: Complete this script to optimize prompts for effectiveness and cost efficiency

# TODO: Import necessary libraries
# Hint: You'll need openai, pandas, time, json, typing, matplotlib, seaborn, datetime, numpy
import openai
from openai import OpenAI
# TODO: Add remaining imports here


# TODO: Define prompt configurations for different cost-effectiveness strategies
# Create a dictionary with three configurations:
# 1. "minimal" - lowest cost but potentially less effective
# 2. "standard" - balanced cost-performance ratio
# 3. "premium" - highest quality but most expensive
PROMPT_CONFIGS = {
    "minimal": {
        # TODO: Choose most cost-effective model (hint: gpt-4o-mini is cheapest)
        "model": "",
        # TODO: Set temperature for balanced results
        "temperature": 0,
        # TODO: Set low max_tokens to control costs
        "max_tokens": 0,
        "description": "Minimal prompts with cost-effective model for budget optimization"
    },
    "standard": {
        # TODO: Choose balanced model (hint: gpt-4o offers good performance)
        "model": "",
        # TODO: Set temperature for consistent results
        "temperature": 0,
        # TODO: Set moderate max_tokens
        "max_tokens": 0,
        "description": "Standard prompts with balanced cost-performance ratio"
    },
    "premium": {
        # TODO: Choose high-performance model
        "model": "",
        # TODO: Set temperature for consistency (hint: lower for more consistent)
        "temperature": 0,
        # TODO: Set higher max_tokens for detailed responses
        "max_tokens": 0,
        "description": "Detailed prompts with premium model for maximum effectiveness"
    }
}

# TODO: Define pricing information for cost calculations
# Research current OpenAI pricing and fill in the rates per 1K tokens in USD
MODEL_PRICING = {
    "gpt-4o-mini": {
        # TODO: Add input token price (hint: check OpenAI pricing page)
        "input": 0,
        # TODO: Add output token price
        "output": 0
    },
    "gpt-4o": {
        # TODO: Add input token price
        "input": 0,
        # TODO: Add output token price
        "output": 0
    },
    "gpt-4-turbo": {
        # TODO: Add input token price
        "input": 0,
        # TODO: Add output token price
        "output": 0
    }
}

# TODO: Define prompt strategies for different task categories
# Create test cases that demonstrate how prompt complexity affects quality and cost
PROMPT_STRATEGIES = [
    {
        "category": "task_completion",
        "minimal": {
            # TODO: Create a basic summarization prompt
            "prompt": "",
            "description": "Basic instruction without context or examples"
        },
        "standard": {
            # TODO: Create a more detailed summarization prompt with guidance
            "prompt": "",
            "description": "Clear instruction with specific guidance"
        },
        "premium": {
            # TODO: Create a comprehensive prompt with role-playing and detailed requirements
            "prompt": "",
            "description": "Detailed instruction with role, structure, and formatting requirements"
        }
    },
    {
        "category": "creative_writing",
        "minimal": {
            # TODO: Create a simple creative writing prompt
            "prompt": "",
            "description": "Simple creative prompt without constraints"
        },
        "standard": {
            # TODO: Create a structured creative prompt with specifications
            "prompt": "",
            "description": "Structured creative prompt with length and content specifications"
        },
        "premium": {
            # TODO: Create a comprehensive creative prompt with detailed requirements
            "prompt": "",
            "description": "Comprehensive creative prompt with role-playing, detailed requirements, and quality guidelines"
        }
    },
    {
        "category": "problem_solving",
        "minimal": {
            # TODO: Create a direct business question
            "prompt": "",
            "description": "Direct question without context"
        },
        "standard": {
            # TODO: Create a contextualized business question
            "prompt": "",
            "description": "Contextualized question with specific details"
        },
        "premium": {
            # TODO: Create a comprehensive business consultation prompt
            "prompt": "",
            "description": "Expert consultation prompt with detailed context, specific deliverables, and structured output requirements"
        }
    }
]

# TODO: Create sample text for testing summarization prompts
# Write a substantial paragraph (200-300 words) about a relevant topic
SAMPLE_TEXT = """
TODO: Add a comprehensive text sample here that can be used for summarization testing.
This should be 200-300 words about a relevant topic like AI, technology, business, etc.
Make sure it has multiple key points that can be summarized effectively.
"""

def call_openai_api(prompt: str, config: Dict) -> Dict:
    """
    Make API call to OpenAI and capture comprehensive response metrics including cost analysis.
    
    TODO: Complete this function to:
    1. Initialize OpenAI client
    2. Make API call with given configuration
    3. Measure response time
    4. Calculate costs based on token usage
    5. Return structured results with cost metrics
    
    Args:
        prompt (str): The input prompt to send to the model
        config (Dict): Model configuration including model name, temperature, etc.
    
    Returns:
        Dict: Response data including content, performance metrics, cost analysis, and error handling
    """
    print(f"  🔄 Calling {config['model']} (temp: {config['temperature']}, max_tokens: {config['max_tokens']})...")
    
    # TODO: Record start time for latency measurement
    start_time = 0
    
    try:
        # TODO: Initialize OpenAI client with your API key
        # SECURITY NOTE: Use environment variables for API keys in production
        client = OpenAI(api_key="YOUR_API_KEY_HERE")
        
        # TODO: Make the API call
        # Hint: Use client.chat.completions.create() with:
        # - model from config
        # - messages with user role and prompt content
        # - temperature from config
        # - max_completion_tokens from config
        response = None
        
        # TODO: Calculate latency in milliseconds
        end_time = 0
        latency = 0
        
        # TODO: Calculate cost based on token usage and model pricing
        # Hint: cost = (tokens / 1000) * price_per_1k_tokens
        model_name = config["model"]
        input_cost = 0
        output_cost = 0
        total_cost = 0
        
        # TODO: Structure the successful response
        # Include: response content, latency, token usage, cost analysis, success status
        result = {
            "response": "",
            "latency_ms": 0,
            "tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0,
            "cost_per_token": 0,
            "success": True,
            "error": None
        }
        
        print(f"  ✅ Success! Latency: {result['latency_ms']}ms, Tokens: {result['tokens_used']}, Cost: ${result['total_cost']:.6f}")
        return result
        
    except Exception as e:
        # TODO: Handle API errors gracefully
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
    
    TODO: Complete this function to:
    1. Check response content for category-specific indicators
    2. Evaluate completeness and quality
    3. Return appropriate score based on criteria
    
    Args:
        response (str): The model's response to evaluate
        category (str): The task category (task_completion, creative_writing, problem_solving)
    
    Returns:
        int: Score from 1-5 where 5 is excellent quality, 1 is poor quality
    """
    if not response:
        return 0
        
    response_lower = response.lower()
    word_count = len(response.split())
    
    # TODO: Implement scoring for task_completion category
    if category == "task_completion":
        # Hint: Look for summary indicators like "main", "key", "important"
        # Check for appropriate length and structure
        # Return score based on completeness and quality
        pass
            
    # TODO: Implement scoring for creative_writing category
    elif category == "creative_writing":
        # Hint: Look for narrative elements, dialogue, descriptive language
        # Check for story completeness and creativity
        # Consider word count and narrative structure
        pass
            
    # TODO: Implement scoring for problem_solving category
    elif category == "problem_solving":
        # Hint: Look for strategy words, analysis terms, actionable advice
        # Check for comprehensiveness and practical value
        # Consider depth of analysis and actionability
        pass
    
    return 3  # Default middle score

def calculate_cost_effectiveness(quality_score: int, total_cost: float) -> float:
    """
    Calculate cost-effectiveness ratio for prompt strategies.
    
    TODO: Complete this function to:
    1. Handle zero cost cases
    2. Calculate quality points per dollar
    3. Return meaningful ratio for comparison
    
    Args:
        quality_score (int): Quality score from 1-5
        total_cost (float): Total cost in USD for the API call
    
    Returns:
        float: Cost-effectiveness ratio (higher is better)
    """
    # TODO: Handle zero cost case
    if total_cost == 0:
        return 0.0
    
    # TODO: Calculate and return cost-effectiveness ratio
    # Hint: quality_score / total_cost, multiply by 1000 for readability
    return 0

def test_prompt_strategy(category: str, strategy_type: str):
    """
    Test a specific prompt strategy and analyze its cost-effectiveness.
    
    TODO: Complete this function to:
    1. Find the specified strategy from PROMPT_STRATEGIES
    2. Get the appropriate configuration
    3. Make API call and evaluate results
    4. Display comprehensive metrics
    5. Return structured results
    
    Args:
        category (str): The task category to test
        strategy_type (str): The prompt strategy type (minimal, standard, premium)
    
    Returns:
        dict: Results including quality scores, costs, and effectiveness metrics
    """
    # TODO: Find the specified strategy
    strategy_data = None
    # Hint: Loop through PROMPT_STRATEGIES to find matching category
    
    if not strategy_data:
        print(f"❌ Category '{category}' not found")
        return None
    
    # TODO: Get prompt info and config
    prompt_info = None
    config = None
    
    print(f"\n🧪 TESTING PROMPT STRATEGY: {category.upper()} - {strategy_type.upper()}")
    print(f"Description: {prompt_info['description']}")
    print(f"Model: {config['model']} | Temp: {config['temperature']} | Max Tokens: {config['max_tokens']}")
    print("=" * 80)
    
    # TODO: Prepare the prompt (replace placeholder if needed)
    prompt = prompt_info["prompt"]
    # Hint: Replace [TEXT_PLACEHOLDER] with SAMPLE_TEXT if present
    
    # TODO: Make API call with current configuration
    result = None
    
    if result['success']:
        # TODO: Score the response quality
        quality_score = 0
        
        # TODO: Calculate cost-effectiveness
        cost_effectiveness = 0
        
        # TODO: Display results with clear formatting
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
        
        # TODO: Return structured results
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
    
    TODO: Complete this function to:
    1. Test all three strategy types for the given category
    2. Collect and analyze results
    3. Create comparison table using pandas
    4. Provide recommendations based on different criteria
    
    Args:
        category (str): The task category to compare strategies for
    
    Returns:
        dict: Comparison results with recommendations
    """
    print(f"\n🔍 COMPARING PROMPT STRATEGIES FOR: {category.upper()}")
    print("=" * 80)
    
    results = {}
    strategy_types = ["minimal", "standard", "premium"]
    
    # TODO: Test each strategy type
    for strategy_type in strategy_types:
        # TODO: Call test_prompt_strategy and store results
        # Add small delay between calls to avoid rate limiting
        pass
    
    # TODO: Analyze and display comparison
    if len(results) > 1:
        print(f"\n📊 STRATEGY COMPARISON SUMMARY:")
        print("=" * 80)
        
        # TODO: Create comparison table using pandas
        # Include: Strategy, Quality, Cost, Tokens, Cost-Effectiveness, Model
        
        # TODO: Provide recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        # TODO: Find and display best cost-effectiveness
        # TODO: Find and display highest quality
        # TODO: Find and display lowest cost
        
        return results
    else:
        print("❌ Insufficient results for comparison")
        return results

# TODO: Example usage - uncomment and test when ready
# Test a single strategy
# minimal_summary = test_prompt_strategy("task_completion", "minimal")

# TODO: Additional test examples you can run:
# Compare all strategies for a category
# task_completion_results = compare_prompt_strategies("task_completion")
# creative_writing_results = compare_prompt_strategies("creative_writing")
# problem_solving_results = compare_prompt_strategies("problem_solving")

# Single strategy tests:
# standard_story = test_prompt_strategy("creative_writing", "standard")
# premium_business = test_prompt_strategy("problem_solving", "premium")

"""
EXERCISE COMPLETION CHECKLIST:
□ Import all necessary libraries
□ Complete PROMPT_CONFIGS with appropriate models and parameters
□ Fill in MODEL_PRICING with current OpenAI pricing
□ Create comprehensive PROMPT_STRATEGIES for all categories and types
□ Write substantial SAMPLE_TEXT for summarization testing
□ Implement call_openai_api() function with cost calculation
□ Complete score_response_quality() with category-specific logic
□ Implement calculate_cost_effectiveness() function
□ Complete test_prompt_strategy() function
□ Implement compare_prompt_strategies() function
□ Test your implementation with the example usage
□ Add your own API key and test the complete workflow

BONUS CHALLENGES:
□ Add visualization of cost vs quality trade-offs using matplotlib
□ Implement batch testing with multiple runs for statistical analysis
□ Add confidence intervals for quality scores
□ Create a budget optimization function that recommends strategies based on cost constraints
□ Add support for custom prompt templates
□ Implement A/B testing framework for prompt comparison
□ Add export functionality for results (CSV, JSON)
□ Create a cost forecasting tool based on usage patterns
"""
