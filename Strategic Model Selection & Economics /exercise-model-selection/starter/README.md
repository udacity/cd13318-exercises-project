# Model Selection Exercise - Starter Guide

## Purpose of this Folder

This folder contains the starter code and instructions for the Model Selection exercise. You'll learn how to compare different Large Language Model (LLM) configurations for specific use cases by completing the `model_selection.py` template.

## Learning Objectives

By completing this exercise, you will:

1. **Understand Model Configuration**: Learn how different parameters (model choice, temperature, tokens) affect LLM behavior
2. **Design Effective Prompts**: Create test prompts that evaluate specific capabilities (reasoning vs. creativity)
3. **Implement API Integration**: Work with the OpenAI API to make model calls and handle responses
4. **Build Evaluation Systems**: Understand how to objectively measure subjective model outputs
5. **Compare Model Performance**: Analyze trade-offs between different model configurations

## Exercise Overview

You'll complete a Python script that:
- Configures two different model setups (reasoning-optimized vs. generation-optimized)
- Tests both configurations on reasoning tasks (math, logic, calculations)
- Tests both configurations on creative tasks (storytelling, marketing, dialogue)
- Scores and compares the results across multiple dimensions

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
   - **Important**: Keep your API key secure and never commit it to version control

### File Structure

```
starter/
├── README.md (this file)
└── model_selection.py (template to complete)
```

## Step-by-Step Instructions

### Step 1: Complete the Imports
Add the missing import statements at the top of `model_selection.py`:
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

### Step 2: Configure Model Settings
Fill in the `MODEL_CONFIGS` dictionary with appropriate values:

**For reasoning_optimized:**
- Model: Choose an o-series model (e.g., "o1-mini" or "o1-preview")
- Temperature: Use a lower value (0.1-0.3) for consistent outputs
- Max tokens: Set to 500-1000 for detailed reasoning

**For generation_optimized:**
- Model: Choose a creative model (e.g., "gpt-4o" or "gpt-4-turbo")
- Temperature: Use a higher value (0.7-1.0) for creative variation
- Max tokens: Set to 500-1000 for longer creative responses
- Top_p: Add nucleus sampling (0.9-0.95)

### Step 3: Design Test Prompts

**REASONING_PROMPTS** - Create three test cases:

1. **Math Word Problem**: 
   - Create a simple arithmetic problem with context
   - Example: "Sarah has X items, gives away Y, buys Z more. How many does she have?"
   - Provide the correct numerical answer

2. **Logical Deduction**:
   - Create an if-then logical sequence
   - Example: "If all A are B, and C is A, what can we conclude about C?"
   - Provide the logical conclusion

3. **Business Calculation**:
   - Create a multi-step percentage problem
   - Example: Revenue changes over multiple quarters
   - Provide the calculated result

**GENERATION_PROMPTS** - Create three creative tasks:

1. **Creative Storytelling**: Write a prompt for imaginative narrative
2. **Marketing Copy**: Request persuasive product description
3. **Creative Dialogue**: Ask for character-based conversation

### Step 4: Implement API Integration
Complete the `call_openai_api()` function:

1. **Timing**: Record start and end times for latency measurement
2. **Client Setup**: Initialize OpenAI client with your API key
3. **API Call**: Use `client.chat.completions.create()` with proper parameters
4. **Response Handling**: Extract content and usage metrics
5. **Error Handling**: Gracefully handle API failures

**Key Implementation Points:**
```python
# Timing
start_time = time.time()

# API Call
response = client.chat.completions.create(
    model=config["model"],
    messages=[{"role": "user", "content": prompt}],
    temperature=config["temperature"],
    max_completion_tokens=config["max_tokens"]
)

# Calculate latency
end_time = time.time()
latency = (end_time - start_time) * 1000  # Convert to milliseconds
```

### Step 5: Test Your Implementation

1. **Add Your API Key**: Replace `"YOUR_API_KEY_HERE"` with your actual OpenAI API key
2. **Start Small**: Test with one task first
3. **Uncomment Test Code**: Enable the example usage at the bottom
4. **Run and Debug**: Execute the script and fix any issues

## Expected Behavior

When working correctly, your script should:

1. **Display Configuration**: Show which model and parameters are being tested
2. **Make API Calls**: Successfully call OpenAI with your prompts
3. **Show Responses**: Display the model's actual responses
4. **Calculate Scores**: Show accuracy scores for reasoning tasks and creativity scores for generation tasks
5. **Report Metrics**: Display latency and token usage for each call
6. **Compare Results**: Allow you to see how different configurations perform

## Sample Output

```
🧠 TESTING REASONING TASK: math_word_problem
Description: Simple arithmetic word problem
Expected Answer: 6 apples
============================================================

🤖 Testing reasoning_optimized:
   Model: o1-mini | Temp: 0.2
  🔄 Calling o1-mini (temp: 0.2)...
  ✅ Success! Latency: 1250.5ms, Tokens: 45

📝 RESPONSE:
----------------------------------------
Sarah starts with 5 apples, gives 2 to Tom (5-2=3), 
then buys 3 more (3+3=6). Sarah has 6 apples.
----------------------------------------
✅ Accuracy Score: 5/5
⏱️  Latency: 1250.5ms
🔢 Tokens: 45
```

## Troubleshooting

### Common Issues and Solutions:

1. **API Key Errors**:
   - Verify your API key is correct
   - Check that you have sufficient credits
   - Ensure the key has proper permissions

2. **Import Errors**:
   - Install missing packages: `pip install package_name`
   - Check Python version compatibility

3. **Model Not Available**:
   - Some models may not be available in all regions
   - Try alternative models (gpt-4o, gpt-4-turbo, etc.)

4. **Rate Limiting**:
   - Add delays between API calls if needed
   - Check your API usage limits

5. **Empty Responses**:
   - Verify your prompts are clear and specific
   - Check that max_tokens is sufficient

## Testing Strategy

### Recommended Testing Order:

1. **Test Imports**: Run the file to check all imports work
2. **Test Configuration**: Print `MODEL_CONFIGS` to verify structure
3. **Test One API Call**: Start with a simple prompt
4. **Test Scoring**: Verify scoring functions work with sample responses
5. **Test Full Workflow**: Run complete reasoning and generation tests

### Validation Checklist:

- [ ] All imports successful
- [ ] MODEL_CONFIGS properly filled
- [ ] All prompts have content and expected answers
- [ ] API calls return successful responses
- [ ] Scoring functions return reasonable scores (1-5)
- [ ] Latency and token metrics are captured
- [ ] Both reasoning and generation tests work

## Extension Ideas

Once you complete the basic exercise, try these enhancements:

1. **Add More Models**: Test additional model configurations
2. **Expand Test Cases**: Create more diverse prompts
3. **Visualize Results**: Create charts comparing performance
4. **Cost Analysis**: Calculate and display API costs
5. **Batch Testing**: Run multiple iterations for statistical analysis
6. **Custom Scoring**: Develop more sophisticated evaluation metrics

## Key Concepts Reinforced

This exercise teaches several important concepts:

- **Model Selection**: Different models excel at different tasks
- **Parameter Tuning**: Temperature and other settings significantly impact output
- **Prompt Engineering**: Well-designed prompts are crucial for good results
- **Evaluation Methodology**: Objective measurement of subjective outputs
- **API Integration**: Proper error handling and metric collection
- **Comparative Analysis**: Systematic approaches to model comparison

## Getting Help

If you encounter issues:

1. **Check the Solution**: Compare with the complete solution in the `solution/` folder
2. **Review Documentation**: Consult OpenAI API documentation
3. **Debug Step by Step**: Use print statements to isolate issues
4. **Test Components**: Verify each function works independently

## Success Criteria

You've successfully completed the exercise when:

- [ ] Your script runs without errors
- [ ] Both model configurations make successful API calls
- [ ] Reasoning tasks show accuracy scores
- [ ] Generation tasks show creativity scores
- [ ] You can compare performance between configurations
- [ ] You understand why different configurations perform differently on different tasks

Remember: The goal is not just to make the code work, but to understand the principles of model selection and evaluation that will help you choose the right LLM configuration for your own projects!
