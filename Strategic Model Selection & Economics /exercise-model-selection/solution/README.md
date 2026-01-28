# Model Selection Exercise - Solution

## Purpose of this Folder

This folder contains the complete solution to the model selection exercise, demonstrating how to compare different LLM configurations for specific use cases. The solution includes comprehensive code examples, evaluation metrics, and best practices for choosing the right model configuration for different types of tasks.

## Solution Overview

The `model_selection.py` script demonstrates a systematic approach to comparing Large Language Models (LLMs) across different task types. This solution teaches students how to:

- Configure models for specific use cases (reasoning vs. generation)
- Implement quantitative evaluation metrics
- Compare model performance across multiple dimensions
- Make data-driven decisions about model selection

## Key Learning Objectives

By studying this solution, students will understand:

1. **Model Configuration Strategy**: How different parameters (temperature, model choice) affect performance
2. **Task-Specific Optimization**: Why certain configurations work better for reasoning vs. creative tasks
3. **Evaluation Methodology**: How to create objective scoring systems for subjective outputs
4. **Performance Metrics**: Measuring latency, token usage, and quality scores
5. **Comparative Analysis**: Systematic approaches to model comparison

## Solution Components

### 1. Model Configurations (`MODEL_CONFIGS`)

The solution defines two distinct configurations:

- **Reasoning Optimized**: Uses o4-mini with lower temperature for consistent, logical outputs
- **Generation Optimized**: Uses gpt-4o with higher temperature for creative, varied responses

### 2. Test Prompts

**Reasoning Prompts** (`REASONING_PROMPTS`):
- Math word problems requiring arithmetic
- Logical deduction tasks
- Multi-step business calculations

**Generation Prompts** (`GENERATION_PROMPTS`):
- Creative storytelling
- Marketing copy creation
- Character dialogue writing

### 3. Evaluation Functions

**`score_reasoning_accuracy()`**: 
- Evaluates logical correctness on a 1-5 scale
- Task-specific scoring criteria
- Checks for both correct answers and proper reasoning

**`score_creativity()`**:
- Assesses creative quality and vocabulary richness
- Task-specific element detection
- Vocabulary diversity analysis

### 4. Testing Framework

**`test_reasoning_task()`** and **`test_generation_task()`**:
- Systematic comparison across model configurations
- Performance metric collection (latency, tokens)
- Formatted output for easy analysis

## How to Use This Solution

### Prerequisites

1. Install required dependencies:
```bash
pip install openai pandas matplotlib seaborn numpy
```

2. Set up your OpenAI API key:
   - Replace the hardcoded API key with your own
   - For production use, store API keys in environment variables

### Running the Solution

1. **Basic Execution**:
```python
python model_selection.py
```

2. **Test Specific Tasks**:
```python
# Test different reasoning tasks
reasoning_results_1 = test_reasoning_task(0)  # Math problem
reasoning_results_2 = test_reasoning_task(1)  # Logic deduction
reasoning_results_3 = test_reasoning_task(2)  # Business calculation

# Test different generation tasks
generation_results_1 = test_generation_task(0)  # Creative story
generation_results_2 = test_generation_task(1)  # Marketing copy
generation_results_3 = test_generation_task(2)  # Dialogue
```

### Expected Output

The solution provides detailed output including:
- Model configuration details
- API response content
- Performance metrics (latency, token usage)
- Quality scores (accuracy for reasoning, creativity for generation)
- Comparative analysis between configurations

## Key Insights from the Solution

### 1. Temperature Impact
- **Lower temperature (1.0)**: More consistent, focused responses ideal for reasoning
- **Higher temperature (1.0)**: More creative variation suitable for generation tasks

### 2. Model Selection
- **O-series models**: Excel at step-by-step reasoning and logical tasks
- **GPT-4o**: Provides excellent creative capabilities and varied outputs

### 3. Evaluation Challenges
- Reasoning tasks can be objectively scored against expected answers
- Creative tasks require more nuanced evaluation criteria
- Vocabulary richness serves as a proxy for creative quality

### 4. Performance Trade-offs
- More capable models may have higher latency
- Token usage varies significantly between tasks and models
- Quality improvements may come at computational cost

## Extension Opportunities

Students can extend this solution by:

1. **Adding More Models**: Test additional model configurations (different temperatures, models)
2. **Expanding Evaluation**: Implement more sophisticated scoring algorithms
3. **Task Variety**: Add new task types (code generation, translation, summarization)
4. **Statistical Analysis**: Add confidence intervals and significance testing
5. **Visualization**: Create charts comparing model performance across dimensions
6. **Cost Analysis**: Include API cost calculations in the comparison

## Best Practices Demonstrated

1. **Structured Configuration**: Using dictionaries to manage model settings
2. **Error Handling**: Graceful handling of API failures
3. **Modular Design**: Separate functions for different concerns
4. **Comprehensive Logging**: Detailed output for debugging and analysis
5. **Quantitative Evaluation**: Objective scoring methods for subjective tasks
6. **Documentation**: Extensive comments explaining the reasoning behind design decisions

## Security Considerations

- **API Key Management**: Never commit API keys to version control
- **Rate Limiting**: Be mindful of API rate limits when scaling tests
- **Cost Monitoring**: Track API usage to avoid unexpected charges
- **Data Privacy**: Ensure test prompts don't contain sensitive information

## Troubleshooting

Common issues and solutions:

1. **API Key Errors**: Verify your OpenAI API key is valid and has sufficient credits
2. **Model Availability**: Some models may not be available in all regions
3. **Rate Limiting**: Add delays between API calls if hitting rate limits
4. **Token Limits**: Adjust max_tokens if responses are being truncated

This solution provides a comprehensive foundation for understanding model selection principles and implementing systematic evaluation approaches in real-world applications.
