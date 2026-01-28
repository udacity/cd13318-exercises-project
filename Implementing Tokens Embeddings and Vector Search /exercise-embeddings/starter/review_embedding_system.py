"""
Review Embedding and Semantic Search System
Lesson 7: Embeddings for Customer Service

This exercise teaches you how to use embeddings to build semantic search for
product reviews and customer feedback. Embeddings convert text into numerical
vectors that capture meaning, enabling you to find similar content and cluster
feedback automatically.

Learning Objectives:
- Create embeddings using OpenAI's embedding API
- Store embeddings with metadata for retrieval
- Calculate cosine similarity between embeddings
- Implement semantic search for similar reviews
- Cluster feedback to identify common themes
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from datetime import datetime
import json


class ReviewEmbeddingSystem:
    """
    Manages embeddings for product reviews and customer feedback.

    This system helps you find similar customer issues, recommend relevant
    responses, and identify common themes in feedback.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the review embedding system.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use (default: text-embedding-3-small)
        """
        # TODO: Initialize the OpenAI client
        # Hint: For Vocareum keys, add: base_url="https://openai.vocareum.com/v1"
        self.client = None

        # TODO: Store the model name
        self.model = None

        # TODO: Initialize storage for embeddings and metadata
        # Format: List of dicts with keys: "text", "embedding", "metadata"
        self.embeddings_store = []

    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for a text string.

        Embeddings are numerical representations that capture the meaning of text.
        Similar texts will have similar embedding vectors.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as a list of floats
        """
        # TODO: Create an embedding using the OpenAI API
        # 1. Call client.embeddings.create() with model and input parameters
        # 2. Extract the embedding vector from response.data[0].embedding
        # 3. Return the embedding as a list

        # Hint: response = self.client.embeddings.create(model=self.model, input=text)
        # Hint: embedding = response.data[0].embedding
        pass

    def embed_review(self, review_text: str, metadata: Dict) -> Dict:
        """
        Create embedding for a review and store it with metadata.

        Args:
            review_text: The review text
            metadata: Associated metadata (rating, product, date, etc.)

        Returns:
            Dictionary with text, embedding, and metadata
        """
        # TODO: Create embedding and store with metadata
        # 1. Create embedding for the review_text
        # 2. Create a dict with "text", "embedding", and "metadata" keys
        # 3. Add to self.embeddings_store
        # 4. Return the dict
        pass

    def embed_reviews(self, reviews: List[Dict]) -> List[Dict]:
        """
        Batch process multiple reviews to create embeddings.

        Args:
            reviews: List of review dicts with "text" and "metadata" keys

        Returns:
            List of embedded reviews
        """
        # TODO: Process multiple reviews
        # For each review in the list:
        # 1. Extract text and metadata
        # 2. Call embed_review()
        # 3. Collect results in a list

        # Note: Could be optimized to use batch API calls, but
        # individual calls are fine for learning purposes
        pass

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Cosine similarity measures how similar two vectors are, ranging from
        -1 (opposite) to 1 (identical). Values close to 1 indicate similar meanings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1
        """
        # TODO: Calculate cosine similarity
        # Formula: similarity = dot(A, B) / (norm(A) * norm(B))

        # 1. Convert embeddings to numpy arrays
        # 2. Calculate dot product: np.dot(vec1, vec2)
        # 3. Calculate norms: np.linalg.norm(vec1) and np.linalg.norm(vec2)
        # 4. Divide dot product by product of norms
        # 5. Return the similarity score

        # Hint: Use numpy for efficient vector operations
        pass

    def find_similar_reviews(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Tuple[Dict, float]]:
        """
        Find reviews most similar to a query using semantic search.

        This is more powerful than keyword search because it understands meaning.
        For example, "product broke" would match reviews saying "stopped working"
        even though they don't share exact words.

        Args:
            query: The search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (review_dict, similarity_score) tuples, sorted by similarity
        """
        # TODO: Implement semantic search
        # 1. Create embedding for the query
        # 2. Calculate similarity with all stored embeddings
        # 3. Filter results by min_similarity
        # 4. Sort by similarity (highest first)
        # 5. Return top_k results

        # Hint: Store (review, similarity) tuples in a list
        # Hint: Sort with sorted(results, key=lambda x: x[1], reverse=True)
        pass

    def find_similar_to_review(
        self,
        review_index: int,
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Find reviews similar to a specific stored review.

        Useful for finding related customer issues or grouping similar feedback.

        Args:
            review_index: Index of the review in embeddings_store
            top_k: Number of similar reviews to return

        Returns:
            List of (review_dict, similarity_score) tuples
        """
        # TODO: Find similar reviews to a given review
        # 1. Get the embedding of the review at review_index
        # 2. Compare with all other embeddings
        # 3. Sort by similarity and return top_k (excluding the review itself)

        # Hint: Similar to find_similar_reviews but uses existing embedding
        pass

    def cluster_feedback(
        self,
        num_clusters: int = 5,
        method: str = "kmeans"
    ) -> Dict[int, List[Dict]]:
        """
        Cluster reviews into groups based on semantic similarity.

        This helps identify common themes in customer feedback automatically.
        For example, it might group all delivery complaints together, all
        product quality issues together, etc.

        Args:
            num_clusters: Number of clusters to create
            method: Clustering method (only 'kmeans' for now)

        Returns:
            Dictionary mapping cluster_id -> list of reviews in that cluster
        """
        # TODO: Implement clustering
        # This is more advanced - basic implementation:

        # 1. Extract all embeddings into a numpy array
        # 2. Use simple K-means clustering (can use sklearn if available)
        # 3. Assign each review to a cluster
        # 4. Group reviews by cluster ID
        # 5. Return dict of cluster_id -> reviews

        # Hint: If sklearn is not available, you can implement simple
        # clustering by finding reviews closest to random centroids

        # For starter file, you can leave this as a challenge
        pass

    def get_cluster_summary(self, cluster_reviews: List[Dict], client: OpenAI) -> str:
        """
        Generate a summary of common themes in a cluster.

        Uses the LLM to analyze reviews in a cluster and describe the common theme.

        Args:
            cluster_reviews: List of reviews in the cluster
            client: OpenAI client for generating summary

        Returns:
            Summary of the cluster's main theme
        """
        # TODO: Generate cluster summary using LLM
        # 1. Extract review texts from cluster_reviews
        # 2. Create a prompt asking for common themes
        # 3. Call the LLM to generate summary
        # 4. Return the summary

        # Hint: Limit to first 10-15 reviews to avoid token limits
        pass

    def save_embeddings(self, filepath: str):
        """
        Save embeddings to a JSON file for later use.

        Args:
            filepath: Path to save the embeddings
        """
        # TODO: Save embeddings to file
        # Use json.dump() to save self.embeddings_store
        # Note: Embeddings are already lists, so they're JSON-serializable
        pass

    def load_embeddings(self, filepath: str):
        """
        Load embeddings from a JSON file.

        Args:
            filepath: Path to the embeddings file
        """
        # TODO: Load embeddings from file
        # Use json.load() to load into self.embeddings_store
        pass


# Sample product review dataset for testing
SAMPLE_REVIEWS = [
    {
        "text": "The laptop arrived quickly and works great! Very satisfied with the purchase.",
        "metadata": {"product": "laptop", "rating": 5, "date": "2024-01-15"}
    },
    {
        "text": "Terrible experience. The product broke after 2 days and customer service was unhelpful.",
        "metadata": {"product": "headphones", "rating": 1, "date": "2024-01-16"}
    },
    {
        "text": "Good quality but delivery took too long. Item arrived 2 weeks after expected date.",
        "metadata": {"product": "keyboard", "rating": 3, "date": "2024-01-17"}
    },
    {
        "text": "Amazing sound quality! These headphones are worth every penny. Highly recommend.",
        "metadata": {"product": "headphones", "rating": 5, "date": "2024-01-18"}
    },
    {
        "text": "The product stopped working after a week. Very disappointed with the quality.",
        "metadata": {"product": "mouse", "rating": 1, "date": "2024-01-19"}
    },
    {
        "text": "Fast shipping and excellent packaging. The laptop exceeded my expectations!",
        "metadata": {"product": "laptop", "rating": 5, "date": "2024-01-20"}
    },
    {
        "text": "Not impressed. The keyboard feels cheap and some keys stick occasionally.",
        "metadata": {"product": "keyboard", "rating": 2, "date": "2024-01-21"}
    },
    {
        "text": "Shipping was slow but the product quality is excellent. Would buy again.",
        "metadata": {"product": "monitor", "rating": 4, "date": "2024-01-22"}
    },
    {
        "text": "Customer service was amazing when I had an issue. They resolved it immediately.",
        "metadata": {"product": "laptop", "rating": 5, "date": "2024-01-23"}
    },
    {
        "text": "The item arrived damaged and the return process was complicated.",
        "metadata": {"product": "monitor", "rating": 2, "date": "2024-01-24"}
    }
]


def demonstrate_embedding_creation():
    """
    Demonstrate creating embeddings for reviews.
    """
    print("\n" + "="*70)
    print("DEMO 1: Creating Embeddings")
    print("="*70)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: Please set OPENAI_API_KEY environment variable")
        return None

    system = ReviewEmbeddingSystem(api_key)

    print("\nCreating embeddings for sample reviews...")
    print(f"Processing {len(SAMPLE_REVIEWS)} reviews...\n")

    # TODO: Uncomment and complete
    # embedded_reviews = system.embed_reviews(SAMPLE_REVIEWS)
    #
    # print(f"Successfully created {len(embedded_reviews)} embeddings")
    # print(f"\nExample embedding (first 10 dimensions):")
    # print(embedded_reviews[0]["embedding"][:10])
    # print(f"Embedding dimension: {len(embedded_reviews[0]['embedding'])}")
    # print(f"\nReview text: {embedded_reviews[0]['text'][:80]}...")

    return system


def demonstrate_similarity_search():
    """
    Demonstrate semantic search for similar reviews.
    """
    print("\n" + "="*70)
    print("DEMO 2: Semantic Search")
    print("="*70)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: Please set OPENAI_API_KEY environment variable")
        return

    system = ReviewEmbeddingSystem(api_key)
    system.embed_reviews(SAMPLE_REVIEWS)

    # Test queries that use semantic understanding
    queries = [
        "product broke quickly",
        "great customer service",
        "slow delivery"
    ]

    print("\nTesting semantic search with different queries:\n")

    # TODO: Uncomment and complete
    # for query in queries:
    #     print(f"Query: '{query}'")
    #     results = system.find_similar_reviews(query, top_k=3)
    #
    #     print(f"Found {len(results)} similar reviews:")
    #     for i, (review, similarity) in enumerate(results, 1):
    #         print(f"\n  {i}. Similarity: {similarity:.3f}")
    #         print(f"     Text: {review['text'][:70]}...")
    #         print(f"     Rating: {review['metadata']['rating']} stars")
    #     print("\n" + "-"*70)


def demonstrate_similarity_calculation():
    """
    Demonstrate similarity calculation between specific reviews.
    """
    print("\n" + "="*70)
    print("DEMO 3: Review Similarity Calculation")
    print("="*70)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: Please set OPENAI_API_KEY environment variable")
        return

    system = ReviewEmbeddingSystem(api_key)
    system.embed_reviews(SAMPLE_REVIEWS)

    print("\nFinding reviews similar to specific examples:\n")

    # TODO: Uncomment and complete
    # # Find reviews similar to the first negative review (index 1)
    # print("Original review (negative about product breaking):")
    # print(f"  {system.embeddings_store[1]['text']}")
    # print(f"  Rating: {system.embeddings_store[1]['metadata']['rating']}")
    #
    # similar = system.find_similar_to_review(1, top_k=3)
    # print("\nMost similar reviews:")
    # for i, (review, similarity) in enumerate(similar, 1):
    #     print(f"\n  {i}. Similarity: {similarity:.3f}")
    #     print(f"     Text: {review['text'][:70]}...")
    #     print(f"     Rating: {review['metadata']['rating']} stars")


def demonstrate_clustering():
    """
    Demonstrate clustering reviews by topic.
    """
    print("\n" + "="*70)
    print("DEMO 4: Clustering Feedback by Theme")
    print("="*70)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: Please set OPENAI_API_KEY environment variable")
        return

    system = ReviewEmbeddingSystem(api_key)
    system.embed_reviews(SAMPLE_REVIEWS)

    print("\nClustering reviews into thematic groups...\n")

    # TODO: Uncomment and complete
    # clusters = system.cluster_feedback(num_clusters=3)
    #
    # for cluster_id, reviews in clusters.items():
    #     print(f"\nCluster {cluster_id}: {len(reviews)} reviews")
    #     print("Sample reviews:")
    #     for review in reviews[:2]:  # Show first 2 from each cluster
    #         print(f"  - {review['text'][:60]}...")
    #         print(f"    Rating: {review['metadata']['rating']} stars")


def demonstrate_practical_use_cases():
    """
    Demonstrate practical applications of embeddings in customer service.
    """
    print("\n" + "="*70)
    print("DEMO 5: Practical Customer Service Use Cases")
    print("="*70)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: Please set OPENAI_API_KEY environment variable")
        return

    system = ReviewEmbeddingSystem(api_key)
    system.embed_reviews(SAMPLE_REVIEWS)

    print("\nUse Case 1: Finding Similar Customer Issues")
    print("-" * 70)

    # TODO: Uncomment and complete
    # new_complaint = "My order hasn't arrived and it's been 3 weeks"
    # print(f"New customer complaint: '{new_complaint}'")
    # similar_issues = system.find_similar_reviews(new_complaint, top_k=3)
    #
    # print("\nSimilar past issues:")
    # for i, (review, similarity) in enumerate(similar_issues, 1):
    #     print(f"  {i}. [{similarity:.3f}] {review['text'][:60]}...")

    print("\n\nUse Case 2: Recommending Template Responses")
    print("-" * 70)
    print("Based on similar issues, suggest appropriate response templates")

    print("\n\nUse Case 3: Identifying Trending Issues")
    print("-" * 70)
    print("Cluster recent reviews to identify common problems requiring attention")


def main():
    """
    Run all demonstrations of the embedding system.
    """
    print("\n" + "="*70)
    print("REVIEW EMBEDDING AND SEMANTIC SEARCH SYSTEM")
    print("Customer Service Use Case")
    print("="*70)

    print("\nThis demo shows you how to:")
    print("1. Create embeddings for customer reviews")
    print("2. Find semantically similar reviews")
    print("3. Calculate similarity between specific reviews")
    print("4. Cluster feedback to identify themes")
    print("5. Apply embeddings to practical customer service scenarios")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n" + "="*70)
        print("ERROR: OPENAI_API_KEY not set")
        print("="*70)
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nFor Vocareum keys:")
        print("  export OPENAI_API_KEY='voc-...'")
        return

    # Run demonstrations
    # TODO: Uncomment these as you implement the functions
    # demonstrate_embedding_creation()
    # demonstrate_similarity_search()
    # demonstrate_similarity_calculation()
    # demonstrate_clustering()
    # demonstrate_practical_use_cases()

    print("\n" + "="*70)
    print("Key Takeaways:")
    print("- Embeddings capture semantic meaning, not just keywords")
    print("- Cosine similarity measures how similar two texts are")
    print("- Semantic search finds relevant content even with different words")
    print("- Clustering automatically groups similar feedback")
    print("- Embeddings enable smarter customer service automation")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
