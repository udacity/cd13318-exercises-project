# Prompt Engineering and Cost Evaluation Exercise - Starter Guide

## Purpose of this Folder

This folder contains the starter code and instructions for the Prompt Engineering and Cost Evaluation exercise. You'll learn how to optimize prompts for both effectiveness and cost efficiency by completing the `prompt_cost_evaluation.py` template. This exercise focuses on real-world business considerations where balancing quality with budget constraints is essential.

## Learning Objectives

By completing this exercise, you will:

1. **Master Prompt Engineering**: Understand how prompt complexity affects output quality and cost
2. **Implement Cost Analysis**: Build comprehensive cost tracking and optimization systems
3. **Evaluate Trade-offs**: Learn to balance quality requirements with budget constraints
4. **Make Data-Driven Decisions**: Use metrics to choose optimal prompt strategies
5. **Apply Business Thinking**: Consider real-world financial implications of LLM usage

## Exercise Overview

You'll complete a Python script that:
- Defines three prompt strategies with different cost-effectiveness profiles (minimal, standard, premium)
- Implements comprehensive cost tracking for different OpenAI models
- Tests prompt strategies across three task categories (summarization, creative writing, problem solving)
- Analyzes cost-effectiveness trade-offs and provides business recommendations
- Creates comparison frameworks for systematic prompt optimization

## Getting Started

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ installed
2. **Required Libraries**: Install the necessary packages:
   ```bash
   pip install openai pandas matplotlib seaborn numpy
   ```
3. **OpenAI API Key**: You'll need an OpenAI API key with credits
   - Sign up at [OpenAI Platform](https://platform.openai.com/)
   - Generate an API key from your dashboard
   - **Important**: Monitor your usage as this exercise involves cost analysis

### File Structure

```
starter/
├── README.md (this file)
└── prompt_cost_evaluation.py (template to complete)
```

## Step-by-Step Instructions

### Step 1: Complete the Imports
Add the missing import statements at the top of `prompt_cost_evaluation.py`:
```python
import pandas as pd
import time
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
```

### Step 2: Configure Prompt Strategies
Fill in the `PROMPT_CONFIGS` dictionary with three cost-effectiveness strategies:

**For minimal strategy:**
- Model: "gpt-4o-mini" (most cost-effective)
- Temperature: 0.7 (balanced for consistency)
- Max tokens: 150 (limited to control costs)

**For standard strategy:**
- Model: "gpt-4o" (balanced performance)
- Temperature: 0.7 (consistent results)
- Max tokens: 300 (moderate length)

**For premium strategy:**
- Model: "gpt-4o" (high performance)
- Temperature: 0.5 (more consistent)
- Max tokens: 500 (detailed responses)

### Step 3: Research and Input Pricing Data
Complete the `MODEL_PRICING` dictionary with current OpenAI pricing:

1. Visit the [OpenAI Pricing Page](https://openai.com/pricing)
2. Find current rates for:
   - gpt-4o-mini (input and output tokens)
   - gpt-4o (input and output tokens)
   - gpt-4-turbo (input and output tokens)
3. Convert prices to per-1K-token rates in USD

**Example format:**
```python
"gpt-4o-mini": {
    "input": 0.00015,   # $0.15 per 1M tokens = $0.00015 per 1K
    "output": 0.0006    # $0.60 per 1M tokens = $0.0006 per 1K
}
```

### Step 4: Design Prompt Strategies
Create comprehensive `PROMPT_STRATEGIES` for three categories:

**Task Completion (Summarization):**
- **Minimal**: "Summarize this text: [TEXT_PLACEHOLDER]"
- **Standard**: "Please provide a concise summary of the following text, focusing on the main points: [TEXT_PLACEHOLDER]"
- **Premium**: "You are an expert analyst. Provide a comprehensive summary with: 1) Main points, 2) Key evidence, 3) Conclusions. Format with headings: [TEXT_PLACEHOLDER]"

**Creative Writing:**
- **Minimal**: "Write a story about a robot."
- **Standard**: "Write a short story (200-300 words) about a robot discovering emotions. Include dialogue."
- **Premium**: "You are a skilled author. Write a compelling story (200-300 words) about a robot discovering emotions. Requirements: 1) Meaningful dialogue, 2) Emotional journey, 3) Narrative arc, 4) Vivid descriptions."

**Problem Solving:**
- **Minimal**: "How do I reduce customer churn?"
- **Standard**: "I'm running a SaaS business with 15% monthly churn. What strategies can reduce churn?"
- **Premium**: "You are a business consultant. My SaaS has 15% churn, $50 ARPU, 6-month LTV. Provide: 1) Root cause analysis, 2) Actionable strategies, 3) Implementation timeline, 4) ROI projections."

### Step 5: Create Sample Content
Write a substantial `SAMPLE_TEXT` (200-300 words) about a relevant topic like AI, technology, or business. This will be used for summarization testing.

### Step 6: Implement Core Functions

**`call_openai_api()` Function:**
1. **Timing**: Record start/end times for latency calculation
2. **API Call**: Use `client.chat.completions.create()` with proper parameters
3. **Cost Calculation**: 
   ```python
   input_cost = (response.usage.prompt_tokens / 1000) * MODEL_PRICING[model_name]["input"]
   output_cost = (response.usage.completion_tokens / 1000) * MODEL_PRICING[model_name]["output"]
   total_cost = input_cost + output_cost
   ```
4. **Error Handling**: Graceful handling of API failures

**`score_response_quality()` Function:**
Implement category-specific scoring:

- **Task Completion**: Look for summary indicators ("main", "key", "important"), check length and structure
- **Creative Writing**: Check for narrative elements, dialogue, descriptive language
- **Problem Solving**: Look for strategy words, analysis terms, actionable advice

**`calculate_cost_effectiveness()` Function:**
```python
if total_cost == 0:
    return 0.0
return (quality_score / total_cost) * 1000  # Quality points per $1000
```

### Step 7: Build Testing Framework

**`test_prompt_strategy()` Function:**
1. Find the strategy from `PROMPT_STRATEGIES`
2. Replace `[TEXT_PLACEHOLDER]` with `SAMPLE_TEXT` if needed
3. Make API call and calculate metrics
4. Display comprehensive results with cost breakdown

**`compare_prompt_strategies()` Function:**
1. Test all three strategies for a category
2. Create pandas DataFrame for comparison
3. Provide recommendations for best cost-effectiveness, quality, and cost

## Expected Behavior

When working correctly, your script should:

### Single Strategy Test Output:
```
🧪 TESTING PROMPT STRATEGY: TASK_COMPLETION - MINIMAL
Description: Basic instruction without context or examples
Model: gpt-4o-mini | Temp: 0.7 | Max Tokens: 150
================================================================================

📝 RESPONSE:
------------------------------------------------------------
[Model's actual response to your prompt]
------------------------------------------------------------

📊 METRICS:
✅ Quality Score: 4/5
⏱️  Latency: 1250ms
🔢 Tokens Used: 85 (Input: 45, Output: 40)
💰 Cost Breakdown:
   Input Cost: $0.000007
   Output Cost: $0.000024
   Total Cost: $0.000031
   Cost per Token: $0.00000036
📈 Cost-Effectiveness: 129032.26 quality points per $1000
```

### Strategy Comparison Output:
```
📊 STRATEGY COMPARISON SUMMARY:
================================================================================
  Strategy  Quality      Cost  Tokens Cost-Effectiveness    Model
0  Minimal      4/5  $0.000031      85           129032.26  gpt-4o-mini
1 Standard      4/5  $0.000156     120            25641.03      gpt-4o
2  Premium      5/5  $0.000298     180            16778.52      gpt-4o

💡 RECOMMENDATIONS:
🏆 Best Cost-Effectiveness: Minimal strategy
⭐ Highest Quality: Premium strategy
💰 Lowest Cost: Minimal strategy
```

## Testing Strategy

### Recommended Testing Order:

1. **Test Imports**: Run the file to verify all imports work
2. **Test Configuration**: Print configurations to verify structure
3. **Test Pricing**: Verify cost calculations with sample data
4. **Test Single API Call**: Start with one simple prompt
5. **Test Scoring**: Verify quality scoring with sample responses
6. **Test Full Strategy**: Run complete single strategy test
7. **Test Comparison**: Run full strategy comparison

### Validation Checklist:

- [ ] All imports successful
- [ ] PROMPT_CONFIGS properly configured with realistic parameters
- [ ] MODEL_PRICING filled with current OpenAI rates
- [ ] All PROMPT_STRATEGIES have meaningful content
- [ ] SAMPLE_TEXT is substantial and relevant
- [ ] API calls return successful responses with cost data
- [ ] Quality scoring returns reasonable scores (1-5)
- [ ] Cost calculations are accurate
- [ ] Comparison function provides clear recommendations

## Business Context and Applications

### Why This Exercise Matters:

1. **Real-world Constraints**: Production LLM applications must balance quality with cost
2. **Scalability**: Small cost differences become significant at scale
3. **ROI Optimization**: Understanding cost-effectiveness helps maximize business value
4. **Budget Management**: Organizations need predictable LLM spending
5. **Strategic Decision Making**: Data-driven prompt optimization

### Industry Applications:

**Content Marketing:**
- Blog post generation with budget constraints
- Social media content at scale
- Email marketing optimization

**Customer Support:**
- Automated response quality vs. cost analysis
- Escalation criteria based on complexity
- Volume-based optimization

**Business Intelligence:**
- Report generation with varying detail levels
- Executive summaries vs. detailed analysis
- Automated insights with cost controls

## Cost Optimization Insights

### Key Learnings to Discover:

1. **Model Selection Impact**: Often more important than prompt complexity
2. **Diminishing Returns**: Premium prompts may not justify additional cost
3. **Task Matching**: Simple tasks often work well with minimal strategies
4. **Volume Considerations**: Cost differences compound at scale
5. **Quality Thresholds**: Define minimum acceptable quality levels

### Business Decision Framework:

- **Budget-Constrained**: Use minimal strategies, focus on cost-effectiveness
- **Quality-Critical**: Use premium strategies, justify with business value
- **Balanced Approach**: Standard strategies often provide optimal value
- **Scale Considerations**: Test at expected production volumes

## Troubleshooting

### Common Issues and Solutions:

1. **API Key Errors**:
   - Verify your API key is correct and active
   - Check that you have sufficient credits
   - Ensure proper key formatting

2. **Cost Calculation Errors**:
   - Verify MODEL_PRICING data is current
   - Check unit conversions (per 1K vs per 1M tokens)
   - Ensure proper decimal precision

3. **Quality Scoring Issues**:
   - Test scoring functions with sample responses
   - Adjust scoring criteria based on actual outputs
   - Consider edge cases (very short/long responses)

4. **Rate Limiting**:
   - Add delays between API calls
   - Monitor your usage limits
   - Implement exponential backoff for retries

5. **Unexpected Costs**:
   - Monitor token usage carefully
   - Set max_tokens appropriately
   - Test with small samples first

## Extension Opportunities

Once you complete the basic exercise, try these enhancements:

### Advanced Features:
1. **Budget Optimization**: Create functions that recommend strategies based on budget constraints
2. **Batch Analysis**: Test multiple prompts simultaneously for statistical significance
3. **Visualization**: Create cost vs. quality scatter plots and trend analysis
4. **A/B Testing**: Framework for comparing custom prompt variations
5. **Cost Forecasting**: Predict monthly costs based on usage patterns
6. **ROI Calculator**: Business value analysis for quality improvements

### Production Features:
1. **Monitoring Dashboard**: Real-time cost and quality tracking
2. **Alert System**: Notifications for budget thresholds
3. **Caching Layer**: Store responses to reduce duplicate costs
4. **Load Balancing**: Distribute requests across models for cost optimization

## Success Criteria

You've successfully completed the exercise when:

- [ ] Your script runs without errors
- [ ] All three prompt strategies make successful API calls
- [ ] Cost calculations are accurate and realistic
- [ ] Quality scoring provides meaningful differentiation
- [ ] Strategy comparison provides clear business recommendations
- [ ] You understand the trade-offs between cost and quality
- [ ] You can explain when to use each strategy type

## Key Takeaways

This exercise teaches essential skills for production LLM applications:

1. **Cost Consciousness**: Always consider financial implications
2. **Quality Measurement**: Develop objective evaluation methods
3. **Business Thinking**: Balance technical capabilities with business needs
4. **Data-Driven Decisions**: Use metrics to guide strategy selection
5. **Scalability Planning**: Consider costs at production volumes

Remember: The goal is not just to make the code work, but to understand how to build cost-effective LLM applications that deliver business value while staying within budget constraints!
