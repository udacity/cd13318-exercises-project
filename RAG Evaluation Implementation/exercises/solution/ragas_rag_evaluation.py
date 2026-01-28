# RAGAS RAG System Evaluation Framework
# This script demonstrates how to evaluate RAG systems using the RAGAS framework
# It covers comprehensive evaluation metrics, automated testing, and performance analysis

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# RAGAS imports for evaluation metrics
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)

# Dataset creation and LLM integration
from datasets import Dataset
import openai
from openai import OpenAI

# ChromaDB integration for RAG system
import chromadb
from chromadb.config import Settings

# Configuration for different evaluation strategies and metrics
# Each configuration focuses on different aspects of RAG system performance
EVALUATION_CONFIGS = {
    # Comprehensive evaluation with all RAGAS metrics
    "comprehensive": {
        "metrics": [
            faithfulness,           # How faithful is the answer to the context
            answer_relevancy,       # How relevant is the answer to the question
            context_precision,      # Precision of retrieved context
            context_recall,         # Recall of retrieved context
            answer_correctness,     # Correctness of the answer
            answer_similarity       # Similarity to ground truth answer
        ],
        "description": "Complete evaluation covering all aspects of RAG performance"
    },
    # Focused on retrieval quality
    "retrieval_focused": {
        "metrics": [
            context_precision,
            context_recall
        ],
        "description": "Evaluation focused on retrieval component quality"
    },
    # Focused on generation quality
    "generation_focused": {
        "metrics": [
            faithfulness,
            answer_relevancy,
            answer_correctness,
            answer_similarity
        ],
        "description": "Evaluation focused on generation component quality"
    },
    # Quick evaluation for development
    "quick_eval": {
        "metrics": [
            faithfulness,
            answer_relevancy,
            context_precision
        ],
        "description": "Fast evaluation for iterative development"
    }
}

# Test datasets for different domains and complexity levels
# These represent realistic evaluation scenarios for RAG systems
EVALUATION_DATASETS = {
    "technical_qa": {
        "name": "Technical Documentation Q&A",
        "description": "Questions about technical concepts and implementations",
        "test_cases": [
            {
                "question": "What is ChromaDB and what are its main features?",
                "ground_truth": "ChromaDB is an open-source vector database designed for AI applications. Its main features include efficient storage and retrieval of high-dimensional vectors, support for multiple embedding functions, both in-memory and persistent storage options, and optimization for semantic search and RAG implementations.",
                "contexts": [
                    "ChromaDB is an open-source vector database designed for AI applications. It provides efficient storage and retrieval of high-dimensional vectors, making it ideal for semantic search, recommendation systems, and RAG implementations.",
                    "ChromaDB supports multiple embedding functions and offers both in-memory and persistent storage options. It's optimized for AI workloads and provides excellent performance for similarity search operations."
                ]
            },
            {
                "question": "How does RAG improve upon traditional language model responses?",
                "ground_truth": "RAG (Retrieval-Augmented Generation) improves upon traditional language models by combining external knowledge retrieval with generation. This provides more accurate, up-to-date, and contextually relevant responses while reducing hallucinations and improving factual accuracy.",
                "contexts": [
                    "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval. By retrieving relevant documents before generation, RAG systems can provide more accurate responses.",
                    "RAG systems reduce hallucinations and improve factual accuracy by grounding responses in retrieved context. This makes them more reliable for knowledge-intensive tasks."
                ]
            },
            {
                "question": "What are vector embeddings and how do they enable semantic search?",
                "ground_truth": "Vector embeddings are numerical representations of text that capture semantic meaning. They enable semantic search by converting text into high-dimensional vectors where similar concepts are positioned closer together in the vector space, allowing for meaning-based rather than keyword-based search.",
                "contexts": [
                    "Vector embeddings are numerical representations of text that capture semantic meaning. Modern embedding models can convert text into high-dimensional vectors.",
                    "In vector space, similar concepts are positioned closer together, enabling semantic search capabilities that go beyond simple keyword matching."
                ]
            }
        ]
    },
    "customer_support": {
        "name": "Customer Support FAQ",
        "description": "Common customer service questions and procedures",
        "test_cases": [
            {
                "question": "How do I reset my password?",
                "ground_truth": "To reset your password, click on the 'Forgot Password' link on the login page, enter your email address, and follow the instructions sent to your email. The reset link expires after 24 hours for security purposes.",
                "contexts": [
                    "Password reset process: Click 'Forgot Password' on login page, enter email address, and follow email instructions. Reset links expire after 24 hours.",
                    "For security, password reset links are only valid for 24 hours. If expired, you'll need to request a new reset link."
                ]
            },
            {
                "question": "What are your business hours?",
                "ground_truth": "Our customer support is available Monday through Friday, 9 AM to 6 PM EST. For urgent technical issues, our emergency support line is available 24/7 for premium customers.",
                "contexts": [
                    "Customer support hours: Monday-Friday, 9 AM to 6 PM EST. Emergency support available 24/7 for premium customers.",
                    "Premium customers have access to 24/7 emergency technical support for urgent issues outside regular business hours."
                ]
            },
            {
                "question": "How can I upgrade my subscription?",
                "ground_truth": "You can upgrade your subscription by logging into your account, navigating to the 'Billing' section, and selecting 'Upgrade Plan'. Changes take effect immediately, and you'll be prorated for the current billing period.",
                "contexts": [
                    "Subscription upgrade process: Log into account → Billing section → Upgrade Plan. Changes are immediate with prorated billing.",
                    "Billing is prorated when upgrading subscriptions, so you only pay for the remaining time in your current billing period."
                ]
            }
        ]
    }
}

class RAGSystemEvaluator:
    """
    Comprehensive RAG system evaluation using RAGAS framework.
    
    This class provides a complete evaluation pipeline for RAG systems,
    including automated testing, metric calculation, and performance analysis.
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the RAG system evaluator.
        
        Args:
            openai_api_key (str): OpenAI API key for LLM-based evaluations
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", api_key=openai_api_key))
        langchain_openai_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_openai_embeddings)
        self.evaluation_results = {}
        self.test_datasets = {}
        
        print("🔍 RAG System Evaluator initialized")
        print(f"   Available evaluation configs: {list(EVALUATION_CONFIGS.keys())}")
        print(f"   Available test datasets: {list(EVALUATION_DATASETS.keys())}")

    def create_evaluation_dataset(self, dataset_key: str, rag_system_responses: List[Dict]) -> Dataset:
        """
        Create a RAGAS-compatible dataset from test cases and RAG responses.
        
        This method formats evaluation data into the structure required by RAGAS,
        combining questions, contexts, ground truth answers, and system responses.
        
        Args:
            dataset_key (str): Key identifying the test dataset
            rag_system_responses (List[Dict]): RAG system responses for each test case
            
        Returns:
            Dataset: RAGAS-compatible dataset for evaluation
        """
        if dataset_key not in EVALUATION_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}")
            
        dataset_config = EVALUATION_DATASETS[dataset_key]
        test_cases = dataset_config["test_cases"]
        
        print(f"\n📊 Creating evaluation dataset: {dataset_config['name']}")
        print(f"   Description: {dataset_config['description']}")
        print(f"   Test cases: {len(test_cases)}")
        
        # Prepare data in RAGAS format
        evaluation_data = {
            "question": [],
            "contexts": [],
            "answer": [],
            "ground_truth": []
        }
        
        for i, test_case in enumerate(test_cases):
            if i < len(rag_system_responses):
                evaluation_data["question"].append(test_case["question"])
                evaluation_data["contexts"].append(test_case["contexts"])
                evaluation_data["answer"].append(rag_system_responses[i]["answer"])
                evaluation_data["ground_truth"].append(test_case["ground_truth"])
            else:
                print(f"⚠️  Missing RAG response for test case {i+1}")
        
        # Create RAGAS dataset
        dataset = Dataset.from_dict(evaluation_data)
        
        print(f"✅ Dataset created with {len(dataset)} examples")
        return dataset

    def evaluate_rag_system(self, dataset: Dataset, config_key: str = "comprehensive") -> Dict:
        """
        Evaluate RAG system using specified RAGAS metrics.
        
        This method runs comprehensive evaluation using RAGAS framework,
        calculating multiple metrics and providing detailed analysis.
        
        Args:
            dataset (Dataset): RAGAS-compatible evaluation dataset
            config_key (str): Configuration key for evaluation metrics
            
        Returns:
            Dict: Comprehensive evaluation results with metrics and analysis
        """
        if config_key not in EVALUATION_CONFIGS:
            raise ValueError(f"Unknown evaluation config: {config_key}")
            
        config = EVALUATION_CONFIGS[config_key]
        
        print(f"\n🧪 Evaluating RAG system with {config_key} configuration")
        print(f"   Description: {config['description']}")
        print(f"   Metrics: {[metric.name for metric in config['metrics']]}")
        
        start_time = time.time()
        
        try:
            # Run RAGAS evaluation
            results = evaluate(
                dataset=dataset,
                metrics=config["metrics"],
                llm=self.evaluator_llm,  # Use OpenAI for LLM-based metrics
                embeddings=self.evaluator_embeddings 
            )
            
            evaluation_time = time.time() - start_time
            
            # Format results for analysis
            formatted_results = {
                "config_used": config_key,
                "evaluation_time": round(evaluation_time, 2),
                "dataset_size": len(dataset),
                "metrics": {},
                "overall_score": 0,
                "detailed_scores": results.to_pandas() if hasattr(results, 'to_pandas') else None
            }
            
            # Extract metric scores
            total_score = 0
            metric_count = 0
            
            for metric in config["metrics"]:
                metric_name = metric.name
                if hasattr(results, metric_name):
                    score = getattr(results, metric_name)
                    formatted_results["metrics"][metric_name] = round(score, 4)
                    total_score += score
                    metric_count += 1
            
            # Calculate overall score
            if metric_count > 0:
                formatted_results["overall_score"] = round(total_score / metric_count, 4)
            
            print(f"✅ Evaluation completed successfully")
            print(f"   Evaluation time: {evaluation_time:.2f}s")
            print(f"   Overall score: {formatted_results['overall_score']:.4f}")
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            return {
                "config_used": config_key,
                "evaluation_time": 0,
                "dataset_size": len(dataset),
                "metrics": {},
                "overall_score": 0,
                "error": str(e)
            }

    def display_evaluation_results(self, results: Dict, dataset_name: str = "Unknown") -> None:
        """
        Display evaluation results in a formatted, comprehensive way.
        
        This method provides detailed analysis of RAG system performance
        including metric breakdowns, insights, and recommendations.
        
        Args:
            results (Dict): Evaluation results from evaluate_rag_system
            dataset_name (str): Name of the evaluated dataset
        """
        print(f"\n" + "="*80)
        print(f"📊 RAG SYSTEM EVALUATION RESULTS")
        print(f"="*80)
        
        print(f"\n📋 EVALUATION SUMMARY:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Configuration: {results['config_used']}")
        print(f"   Dataset Size: {results['dataset_size']} examples")
        print(f"   Evaluation Time: {results['evaluation_time']}s")
        print(f"   Overall Score: {results['overall_score']:.4f}")
        
        if "error" in results:
            print(f"\n❌ EVALUATION ERROR:")
            print(f"   {results['error']}")
            return
        
        print(f"\n📈 METRIC BREAKDOWN:")
        for metric_name, score in results["metrics"].items():
            # Provide interpretation for each metric
            interpretation = self._interpret_metric_score(metric_name, score)
            print(f"   {metric_name}: {score:.4f} - {interpretation}")
        
        # Provide overall analysis
        print(f"\n💡 PERFORMANCE ANALYSIS:")
        self._analyze_performance(results)
        
        # Provide recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        self._provide_recommendations(results)

    def _interpret_metric_score(self, metric_name: str, score: float) -> str:
        """
        Provide interpretation for individual metric scores.
        
        Args:
            metric_name (str): Name of the metric
            score (float): Metric score (typically 0-1)
            
        Returns:
            str: Human-readable interpretation of the score
        """
        # Define score thresholds
        if score >= 0.8:
            quality = "Excellent"
        elif score >= 0.6:
            quality = "Good"
        elif score >= 0.4:
            quality = "Fair"
        elif score >= 0.2:
            quality = "Poor"
        else:
            quality = "Very Poor"
        
        # Metric-specific interpretations
        interpretations = {
            "faithfulness": f"{quality} - Generated answers are {'highly' if score >= 0.8 else 'moderately' if score >= 0.6 else 'poorly'} faithful to retrieved context",
            "answer_relevancy": f"{quality} - Answers are {'highly' if score >= 0.8 else 'moderately' if score >= 0.6 else 'poorly'} relevant to questions",
            "context_precision": f"{quality} - Retrieved context has {'high' if score >= 0.8 else 'moderate' if score >= 0.6 else 'low'} precision",
            "context_recall": f"{quality} - Retrieved context has {'high' if score >= 0.8 else 'moderate' if score >= 0.6 else 'low'} recall",
            "answer_correctness": f"{quality} - Answers are {'highly' if score >= 0.8 else 'moderately' if score >= 0.6 else 'poorly'} correct",
            "answer_similarity": f"{quality} - Answers are {'very' if score >= 0.8 else 'moderately' if score >= 0.6 else 'poorly'} similar to ground truth"
        }
        
        return interpretations.get(metric_name, f"{quality} performance")

    def _analyze_performance(self, results: Dict) -> None:
        """
        Provide comprehensive performance analysis based on metric scores.
        
        Args:
            results (Dict): Evaluation results with metric scores
        """
        metrics = results["metrics"]
        
        # Analyze retrieval performance
        retrieval_metrics = ["context_precision", "context_recall"]
        retrieval_scores = [metrics.get(m, 0) for m in retrieval_metrics if m in metrics]
        
        if retrieval_scores:
            avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
            print(f"   Retrieval Performance: {avg_retrieval:.4f} - {'Strong' if avg_retrieval >= 0.7 else 'Moderate' if avg_retrieval >= 0.5 else 'Weak'} retrieval capabilities")
        
        # Analyze generation performance
        generation_metrics = ["faithfulness", "answer_relevancy", "answer_correctness"]
        generation_scores = [metrics.get(m, 0) for m in generation_metrics if m in metrics]
        
        if generation_scores:
            avg_generation = sum(generation_scores) / len(generation_scores)
            print(f"   Generation Performance: {avg_generation:.4f} - {'Strong' if avg_generation >= 0.7 else 'Moderate' if avg_generation >= 0.5 else 'Weak'} generation capabilities")
        
        # Identify strengths and weaknesses
        if metrics:
            best_metric = max(metrics.items(), key=lambda x: x[1])
            worst_metric = min(metrics.items(), key=lambda x: x[1])
            
            print(f"   Strongest Area: {best_metric[0]} ({best_metric[1]:.4f})")
            print(f"   Weakest Area: {worst_metric[0]} ({worst_metric[1]:.4f})")

    def _provide_recommendations(self, results: Dict) -> None:
        """
        Provide actionable recommendations based on evaluation results.
        
        Args:
            results (Dict): Evaluation results with metric scores
        """
        metrics = results["metrics"]
        recommendations = []
        
        # Context-based recommendations
        if "context_precision" in metrics and metrics["context_precision"] < 0.6:
            recommendations.append("Improve retrieval precision by refining similarity thresholds or using re-ranking")
        
        if "context_recall" in metrics and metrics["context_recall"] < 0.6:
            recommendations.append("Enhance context recall by increasing retrieved document count or improving embeddings")
        
        # Generation-based recommendations
        if "faithfulness" in metrics and metrics["faithfulness"] < 0.6:
            recommendations.append("Improve faithfulness by refining prompts to emphasize context adherence")
        
        if "answer_relevancy" in metrics and metrics["answer_relevancy"] < 0.6:
            recommendations.append("Enhance answer relevancy through better prompt engineering and context selection")
        
        if "answer_correctness" in metrics and metrics["answer_correctness"] < 0.6:
            recommendations.append("Improve answer correctness by using higher-quality models or better context")
        
        # General recommendations
        if results["overall_score"] < 0.5:
            recommendations.append("Consider comprehensive system redesign focusing on both retrieval and generation components")
        elif results["overall_score"] < 0.7:
            recommendations.append("Focus on incremental improvements in the weakest performing areas")
        
        if not recommendations:
            recommendations.append("System performance is strong - consider advanced optimizations for specific use cases")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

    def compare_configurations(self, dataset: Dataset, configs: List[str]) -> Dict:
        """
        Compare multiple evaluation configurations on the same dataset.
        
        This method enables systematic comparison of different evaluation approaches
        to understand which metrics are most relevant for specific use cases.
        
        Args:
            dataset (Dataset): RAGAS-compatible evaluation dataset
            configs (List[str]): List of configuration keys to compare
            
        Returns:
            Dict: Comparison results across configurations
        """
        print(f"\n🔄 Comparing {len(configs)} evaluation configurations")
        
        comparison_results = {
            "configurations": configs,
            "results": {},
            "summary": {}
        }
        
        # Evaluate with each configuration
        for config in configs:
            if config in EVALUATION_CONFIGS:
                print(f"\n--- Evaluating with {config} configuration ---")
                result = self.evaluate_rag_system(dataset, config)
                comparison_results["results"][config] = result
                
                # Add small delay to avoid rate limiting
                time.sleep(1)
        
        # Create summary comparison
        self._create_comparison_summary(comparison_results)
        
        return comparison_results

    def _create_comparison_summary(self, comparison_results: Dict) -> None:
        """
        Create summary comparison across different configurations.
        
        Args:
            comparison_results (Dict): Results from compare_configurations
        """
        results = comparison_results["results"]
        
        # Compare overall scores
        overall_scores = {config: result["overall_score"] for config, result in results.items()}
        best_config = max(overall_scores.items(), key=lambda x: x[1])
        
        comparison_results["summary"] = {
            "best_overall": best_config,
            "score_comparison": overall_scores,
            "evaluation_times": {config: result["evaluation_time"] for config, result in results.items()}
        }
        
        print(f"\n📊 CONFIGURATION COMPARISON SUMMARY:")
        print(f"   Best Overall Performance: {best_config[0]} ({best_config[1]:.4f})")
        
        print(f"\n   Score Comparison:")
        for config, score in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"     {config}: {score:.4f}")

    def generate_evaluation_report(self, results: Dict, dataset_name: str, output_file: str = None) -> str:
        """
        Generate comprehensive evaluation report in markdown format.
        
        Args:
            results (Dict): Evaluation results
            dataset_name (str): Name of evaluated dataset
            output_file (str): Optional file path to save report
            
        Returns:
            str: Markdown-formatted evaluation report
        """
        report = f"""# RAG System Evaluation Report

## Executive Summary
- **Dataset**: {dataset_name}
- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Configuration**: {results['config_used']}
- **Overall Score**: {results['overall_score']:.4f}
- **Dataset Size**: {results['dataset_size']} examples
- **Evaluation Time**: {results['evaluation_time']}s

## Metric Breakdown
"""
        
        for metric_name, score in results["metrics"].items():
            interpretation = self._interpret_metric_score(metric_name, score)
            report += f"- **{metric_name}**: {score:.4f} - {interpretation}\n"
        
        report += f"""
## Performance Analysis
{self._get_performance_analysis_text(results)}

## Recommendations
{self._get_recommendations_text(results)}

## Technical Details
- **Evaluation Framework**: RAGAS
- **LLM Provider**: OpenAI
- **Evaluation Configuration**: {results['config_used']}
- **Timestamp**: {datetime.now().isoformat()}
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to: {output_file}")
        
        return report

    def _get_performance_analysis_text(self, results: Dict) -> str:
        """Generate performance analysis text for reports."""
        metrics = results["metrics"]
        analysis = []
        
        # Analyze retrieval vs generation performance
        retrieval_metrics = ["context_precision", "context_recall"]
        generation_metrics = ["faithfulness", "answer_relevancy", "answer_correctness"]
        
        retrieval_scores = [metrics.get(m, 0) for m in retrieval_metrics if m in metrics]
        generation_scores = [metrics.get(m, 0) for m in generation_metrics if m in metrics]
        
        if retrieval_scores:
            avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
            analysis.append(f"Retrieval component shows {'strong' if avg_retrieval >= 0.7 else 'moderate' if avg_retrieval >= 0.5 else 'weak'} performance ({avg_retrieval:.4f})")
        
        if generation_scores:
            avg_generation = sum(generation_scores) / len(generation_scores)
            analysis.append(f"Generation component shows {'strong' if avg_generation >= 0.7 else 'moderate' if avg_generation >= 0.5 else 'weak'} performance ({avg_generation:.4f})")
        
        return "\n".join(analysis) if analysis else "Performance analysis not available"

    def _get_recommendations_text(self, results: Dict) -> str:
        """Generate recommendations text for reports."""
        metrics = results["metrics"]
        recommendations = []
        
        # Generate recommendations based on scores
        if "context_precision" in metrics and metrics["context_precision"] < 0.6:
            recommendations.append("Improve retrieval precision through better similarity thresholds")
        
        if "faithfulness" in metrics and metrics["faithfulness"] < 0.6:
            recommendations.append("Enhance faithfulness by improving prompt engineering")
        
        if "answer_relevancy" in metrics and metrics["answer_relevancy"] < 0.6:
            recommendations.append("Optimize answer relevancy through better context selection")
        
        if not recommendations:
            recommendations.append("System performance is satisfactory - consider advanced optimizations")
        
        return "\n".join(f"- {rec}" for rec in recommendations)

# Mock RAG System for demonstration purposes
class MockRAGSystem:
    """
    Mock RAG system for demonstration and testing purposes.
    
    This class simulates a RAG system to provide realistic responses
    for evaluation without requiring a full RAG implementation.
    """
    
    def __init__(self):
        """Initialize mock RAG system with predefined responses."""
        self.responses = {
            "technical_qa": [
                {
                    "question": "What is ChromaDB and what are its main features?",
                    "answer": "ChromaDB is an open-source vector database specifically designed for AI applications. Its main features include efficient storage and retrieval of high-dimensional vectors, support for multiple embedding functions, persistent and in-memory storage options, and optimization for semantic search and RAG implementations. It provides excellent performance for similarity search operations.",
                    "contexts": ["ChromaDB documentation", "Vector database guide"],
                    "retrieval_time": 0.15,
                    "generation_time": 1.2
                },
                {
                    "question": "How does RAG improve upon traditional language model responses?",
                    "answer": "RAG significantly improves traditional language models by incorporating external knowledge retrieval before generation. This approach provides more accurate, up-to-date, and contextually relevant responses. RAG reduces hallucinations by grounding responses in retrieved context and improves factual accuracy for knowledge-intensive tasks.",
                    "contexts": ["RAG research papers", "AI system architecture"],
                    "retrieval_time": 0.18,
                    "generation_time": 1.4
                },
                {
                    "question": "What are vector embeddings and how do they enable semantic search?",
                    "answer": "Vector embeddings are numerical representations that capture the semantic meaning of text. They convert text into high-dimensional vectors where semantically similar concepts are positioned closer together in the vector space. This enables semantic search by allowing similarity comparisons based on meaning rather than just keyword matching.",
                    "contexts": ["Embedding models guide", "Semantic search principles"],
                    "retrieval_time": 0.12,
                    "generation_time": 1.1
                }
            ],
            "customer_support": [
                {
                    "question": "How do I reset my password?",
                    "answer": "To reset your password, click the 'Forgot Password' link on the login page, enter your email address, and follow the instructions sent to your email. The reset link expires after 24 hours for security. If you don't receive the email, check your spam folder or contact support.",
                    "contexts": ["Password reset procedure", "Security policies"],
                    "retrieval_time": 0.08,
                    "generation_time": 0.9
                },
                {
                    "question": "What are your business hours?",
                    "answer": "Our customer support is available Monday through Friday from 9 AM to 6 PM EST. For urgent technical issues, premium customers have access to our 24/7 emergency support line. Standard customers can submit tickets anytime through our support portal.",
                    "contexts": ["Support hours policy", "Premium customer benefits"],
                    "retrieval_time": 0.06,
                    "generation_time": 0.8
                },
                {
                    "question": "How can I upgrade my subscription?",
                    "answer": "You can upgrade your subscription by logging into your account and navigating to the 'Billing' section, then selecting 'Upgrade Plan'. Changes take effect immediately, and billing is prorated for the current period. You can also contact our sales team for assistance with enterprise upgrades.",
                    "contexts": ["Billing procedures", "Subscription management"],
                    "retrieval_time": 0.10,
                    "generation_time": 1.0
                }
            ]
        }
    
    def get_responses(self, dataset_key: str) -> List[Dict]:
        """
        Get mock RAG responses for a specific dataset.
        
        Args:
            dataset_key (str): Dataset identifier
            
        Returns:
            List[Dict]: Mock RAG responses
        """
        return self.responses.get(dataset_key, [])

def demonstrate_ragas_evaluation():
    """
    Comprehensive demonstration of RAGAS evaluation framework.
    
    This function showcases the complete evaluation workflow from dataset
    preparation through metric calculation to result analysis and reporting.
    """
    print("🚀 RAGAS RAG System Evaluation Demonstration")
    print("="*60)
    
    # Initialize evaluator (replace with your OpenAI API key)
    evaluator = RAGSystemEvaluator(
        openai_api_key="your-key-here"
    )
    
    # Initialize mock RAG system for demonstration
    mock_rag = MockRAGSystem()
    
    # Test different datasets and configurations
    test_scenarios = [
        {
            "dataset": "technical_qa",
            "config": "comprehensive",
            "description": "Technical Q&A with comprehensive evaluation"
        },
        {
            "dataset": "customer_support", 
            "config": "quick_eval",
            "description": "Customer support with quick evaluation"
        }
    ]
    
    all_results = {}
    
    for scenario in test_scenarios:
        dataset_key = scenario["dataset"]
        config_key = scenario["config"]
        
        print(f"\n{'='*20} {scenario['description']} {'='*20}")
        
        # Get mock RAG responses
        rag_responses = mock_rag.get_responses(dataset_key)
        
        if not rag_responses:
            print(f"⚠️  No responses available for dataset: {dataset_key}")
            continue
        
        # Create evaluation dataset
        eval_dataset = evaluator.create_evaluation_dataset(dataset_key, rag_responses)
        
        # Run evaluation
        results = evaluator.evaluate_rag_system(eval_dataset, config_key)
        
        # Display results
        dataset_name = EVALUATION_DATASETS[dataset_key]["name"]
        evaluator.display_evaluation_results(results, dataset_name)
        
        # Store results for comparison
        all_results[f"{dataset_key}_{config_key}"] = results
        
        # Add delay to avoid rate limiting
        time.sleep(2)
    
    # Demonstrate configuration comparison
    if len(all_results) > 1:
        print(f"\n{'='*20} Configuration Comparison {'='*20}")
        
        # Compare different configurations on technical dataset
        tech_responses = mock_rag.get_responses("technical_qa")
        tech_dataset = evaluator.create_evaluation_dataset("technical_qa", tech_responses)
        
        comparison_configs = ["comprehensive", "retrieval_focused", "generation_focused"]
        comparison_results = evaluator.compare_configurations(tech_dataset, comparison_configs)
        
        # Display comparison summary
        print(f"\n📊 COMPARISON SUMMARY:")
        summary = comparison_results["summary"]
        print(f"   Best Configuration: {summary['best_overall'][0]} ({summary['best_overall'][1]:.4f})")
        
        for config, score in summary["score_comparison"].items():
            print(f"   {config}: {score:.4f}")
    
    # Generate sample report
    if all_results:
        sample_result = list(all_results.values())[0]
        sample_dataset = list(EVALUATION_DATASETS.keys())[0]
        
        print(f"\n📄 Generating evaluation report...")
        report = evaluator.generate_evaluation_report(
            sample_result, 
            EVALUATION_DATASETS[sample_dataset]["name"],
            "rag_evaluation_report.md"
        )
    
    print(f"\n🎉 RAGAS evaluation demonstration completed!")
    print(f"   Scenarios tested: {len(test_scenarios)}")
    print(f"   Configurations compared: {len(comparison_configs) if 'comparison_configs' in locals() else 0}")
    print(f"   Total evaluations: {len(all_results)}")

def run_custom_evaluation(dataset_key: str, rag_responses: List[Dict], config_key: str = "comprehensive"):
    """
    Run custom evaluation with user-provided data.
    
    This function allows users to evaluate their own RAG systems with custom
    datasets and responses using the RAGAS framework.
    
    Args:
        dataset_key (str): Key identifying the test dataset
        rag_responses (List[Dict]): RAG system responses to evaluate
        config_key (str): Evaluation configuration to use
        
    Returns:
        Dict: Evaluation results
    """
    print(f"🔧 Running custom evaluation")
    print(f"   Dataset: {dataset_key}")
    print(f"   Configuration: {config_key}")
    print(f"   Responses: {len(rag_responses)}")
    
    # Initialize evaluator (replace with your API key)
    evaluator = RAGSystemEvaluator(
        openai_api_key="YOUR_OPENAI_API_KEY_HERE"
    )
    
    # Create evaluation dataset
    eval_dataset = evaluator.create_evaluation_dataset(dataset_key, rag_responses)
    
    # Run evaluation
    results = evaluator.evaluate_rag_system(eval_dataset, config_key)
    
    # Display results
    dataset_name = EVALUATION_DATASETS.get(dataset_key, {}).get("name", dataset_key)
    evaluator.display_evaluation_results(results, dataset_name)
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Run the comprehensive demonstration
    demonstrate_ragas_evaluation()
    
    # Additional examples for advanced usage:
    
    # Example 1: Custom evaluation with your own RAG system
    # custom_responses = [
    #     {
    #         "question": "Your question here",
    #         "answer": "Your RAG system's answer",
    #         "contexts": ["Retrieved context 1", "Retrieved context 2"],
    #         "retrieval_time": 0.15,
    #         "generation_time": 1.2
    #     }
    # ]
    # custom_results = run_custom_evaluation("technical_qa", custom_responses, "comprehensive")
    
    # Example 2: Batch evaluation with multiple configurations
    # evaluator = RAGSystemEvaluator("your_api_key")
    # mock_rag = MockRAGSystem()
    # responses = mock_rag.get_responses("technical_qa")
    # dataset = evaluator.create_evaluation_dataset("technical_qa", responses)
    # 
    # configs_to_test = ["comprehensive", "retrieval_focused", "generation_focused", "quick_eval"]
    # comparison = evaluator.compare_configurations(dataset, configs_to_test)
    
    # Example 3: Generate detailed reports
    # report = evaluator.generate_evaluation_report(
    #     results, 
    #     "Technical Documentation Q&A",
    #     "detailed_evaluation_report.md"
    # )
