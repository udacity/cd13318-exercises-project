"""
Review Embedding and Semantic Search System - SOLUTION
Lesson 7: Embeddings for Customer Service

This is the complete solution for the embeddings exercise.
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
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the review embedding system.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use (default: text-embedding-3-small)
        """
        # Initialize the OpenAI client with Vocareum base URL
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=api_key
        )
        self.model = model

        # Initialize storage for embeddings and metadata
        self.embeddings_store: List[Dict] = []

    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for a text string.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as a list of floats
        """
        try:
            # Create embedding using OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )

            # Extract the embedding vector
            embedding = response.data[0].embedding

            return embedding

        except Exception as e:
            print(f"Error creating embedding: {e}")
            raise

    def embed_review(self, review_text: str, metadata: Dict) -> Dict:
        """
        Create embedding for a review and store it with metadata.

        Args:
            review_text: The review text
            metadata: Associated metadata (rating, product, date, etc.)

        Returns:
            Dictionary with text, embedding, and metadata
        """
        # Create embedding
        embedding = self.create_embedding(review_text)

        # Create review entry
        review_entry = {
            "text": review_text,
            "embedding": embedding,
            "metadata": metadata
        }

        # Store it
        self.embeddings_store.append(review_entry)

        return review_entry

    def embed_reviews(self, reviews: List[Dict]) -> List[Dict]:
        """
        Batch process multiple reviews to create embeddings.

        Args:
            reviews: List of review dicts with "text" and "metadata" keys

        Returns:
            List of embedded reviews
        """
        embedded_reviews = []

        for review in reviews:
            try:
                embedded = self.embed_review(
                    review["text"],
                    review["metadata"]
                )
                embedded_reviews.append(embedded)
            except Exception as e:
                print(f"Error embedding review: {e}")
                continue

        return embedded_reviews

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        # Formula: cos(theta) = dot(A, B) / (norm(A) * norm(B))
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        return float(similarity)

    def find_similar_reviews(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Tuple[Dict, float]]:
        """
        Find reviews most similar to a query using semantic search.

        Args:
            query: The search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (review_dict, similarity_score) tuples, sorted by similarity
        """
        if not self.embeddings_store:
            return []

        # Create embedding for the query
        query_embedding = self.create_embedding(query)

        # Calculate similarity with all stored embeddings
        results = []
        for review in self.embeddings_store:
            similarity = self.calculate_similarity(
                query_embedding,
                review["embedding"]
            )

            # Filter by minimum similarity
            if similarity >= min_similarity:
                results.append((review, similarity))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return results[:top_k]

    def find_similar_to_review(
        self,
        review_index: int,
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Find reviews similar to a specific stored review.

        Args:
            review_index: Index of the review in embeddings_store
            top_k: Number of similar reviews to return

        Returns:
            List of (review_dict, similarity_score) tuples
        """
        if review_index < 0 or review_index >= len(self.embeddings_store):
            raise ValueError(f"Invalid review index: {review_index}")

        # Get the target review's embedding
        target_embedding = self.embeddings_store[review_index]["embedding"]

        # Calculate similarity with all other embeddings
        results = []
        for i, review in enumerate(self.embeddings_store):
            # Skip the review itself
            if i == review_index:
                continue

            similarity = self.calculate_similarity(
                target_embedding,
                review["embedding"]
            )

            results.append((review, similarity))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return results[:top_k]

    def cluster_feedback(
        self,
        num_clusters: int = 5,
        method: str = "kmeans"
    ) -> Dict[int, List[Dict]]:
        """
        Cluster reviews into groups based on semantic similarity.

        Args:
            num_clusters: Number of clusters to create
            method: Clustering method (only 'kmeans' for now)

        Returns:
            Dictionary mapping cluster_id -> list of reviews in that cluster
        """
        if not self.embeddings_store:
            return {}

        # Extract embeddings into numpy array
        embeddings_matrix = np.array([
            review["embedding"] for review in self.embeddings_store
        ])

        # Simple K-means clustering implementation
        # (In production, you'd use sklearn.cluster.KMeans)
        try:
            from sklearn.cluster import KMeans

            # Use sklearn if available
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_matrix)

        except ImportError:
            # Simple fallback clustering if sklearn not available
            print("Note: sklearn not available, using simple clustering")
            cluster_labels = self._simple_kmeans(embeddings_matrix, num_clusters)

        # Group reviews by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.embeddings_store[i])

        return clusters

    def _simple_kmeans(self, embeddings: np.ndarray, k: int, max_iters: int = 10) -> np.ndarray:
        """
        Simple K-means implementation (fallback if sklearn not available).

        Args:
            embeddings: Matrix of embeddings
            k: Number of clusters
            max_iters: Maximum iterations

        Returns:
            Array of cluster labels
        """
        n_samples = len(embeddings)

        # Initialize centroids randomly
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = embeddings[indices]

        for _ in range(max_iters):
            # Assign each point to nearest centroid
            distances = np.array([
                [np.linalg.norm(emb - centroid) for centroid in centroids]
                for emb in embeddings
            ])
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                embeddings[labels == i].mean(axis=0) if np.any(labels == i)
                else centroids[i]
                for i in range(k)
            ])

            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return labels

    def get_cluster_summary(self, cluster_reviews: List[Dict]) -> str:
        """
        Generate a summary of common themes in a cluster.

        Args:
            cluster_reviews: List of reviews in the cluster

        Returns:
            Summary of the cluster's main theme
        """
        if not cluster_reviews:
            return "No reviews in cluster"

        # Extract review texts (limit to avoid token limits)
        review_texts = [r["text"] for r in cluster_reviews[:10]]

        # Create prompt for summarization
        prompt = f"""Analyze these customer reviews and identify the common theme or topic.
Provide a brief 1-2 sentence summary of what these reviews are about.

Reviews:
{chr(10).join(f"- {text}" for text in review_texts)}

Common theme:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Unable to generate summary"

    def save_embeddings(self, filepath: str):
        """
        Save embeddings to a JSON file for later use.

        Args:
            filepath: Path to save the embeddings
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.embeddings_store, f, indent=2)
            print(f"Embeddings saved to {filepath}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self, filepath: str):
        """
        Load embeddings from a JSON file.

        Args:
            filepath: Path to the embeddings file
        """
        try:
            with open(filepath, 'r') as f:
                self.embeddings_store = json.load(f)
            print(f"Loaded {len(self.embeddings_store)} embeddings from {filepath}")
        except Exception as e:
            print(f"Error loading embeddings: {e}")


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

    embedded_reviews = system.embed_reviews(SAMPLE_REVIEWS)

    print(f"Successfully created {len(embedded_reviews)} embeddings")
    print(f"\nExample embedding (first 10 dimensions):")
    print([f"{x:.4f}" for x in embedded_reviews[0]["embedding"][:10]])
    print(f"Embedding dimension: {len(embedded_reviews[0]['embedding'])}")
    print(f"\nReview text: {embedded_reviews[0]['text'][:80]}...")
    print("\nKey insight: Each review is now represented as a vector of numbers")
    print("that captures its semantic meaning!")

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
    print("\nEmbedding reviews...")
    system.embed_reviews(SAMPLE_REVIEWS)

    # Test queries that use semantic understanding
    queries = [
        "product broke quickly",
        "great customer service",
        "slow delivery"
    ]

    print("\nTesting semantic search with different queries:")
    print("Notice how it finds relevant reviews even with different wording!\n")

    for query in queries:
        print("="*70)
        print(f"Query: '{query}'")
        results = system.find_similar_reviews(query, top_k=3)

        print(f"Found {len(results)} similar reviews:")
        for i, (review, similarity) in enumerate(results, 1):
            print(f"\n  {i}. Similarity: {similarity:.3f}")
            print(f"     Text: {review['text'][:70]}...")
            print(f"     Rating: {review['metadata']['rating']} stars")
            print(f"     Product: {review['metadata']['product']}")
        print()


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
    print("\nEmbedding reviews...")
    system.embed_reviews(SAMPLE_REVIEWS)

    print("\nFinding reviews similar to specific examples:\n")

    # Find reviews similar to the first negative review (index 1)
    print("Original review (negative about product breaking):")
    print(f"  Text: {system.embeddings_store[1]['text']}")
    print(f"  Rating: {system.embeddings_store[1]['metadata']['rating']}")

    similar = system.find_similar_to_review(1, top_k=3)
    print("\nMost similar reviews:")
    for i, (review, similarity) in enumerate(similar, 1):
        print(f"\n  {i}. Similarity: {similarity:.3f}")
        print(f"     Text: {review['text'][:70]}...")
        print(f"     Rating: {review['metadata']['rating']} stars")

    print("\nKey insight: The system found other negative reviews about")
    print("product quality issues, showing semantic understanding!")


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
    print("\nEmbedding reviews...")
    system.embed_reviews(SAMPLE_REVIEWS)

    print("\nClustering reviews into thematic groups...\n")

    clusters = system.cluster_feedback(num_clusters=3)

    for cluster_id, reviews in clusters.items():
        print("="*70)
        print(f"Cluster {cluster_id}: {len(reviews)} reviews")

        # Get cluster summary
        summary = system.get_cluster_summary(reviews)
        print(f"Theme: {summary}")

        print("\nSample reviews:")
        for review in reviews[:3]:  # Show first 3 from each cluster
            print(f"  - [{review['metadata']['rating']}⭐] {review['text'][:55]}...")

        print()

    print("Key insight: Clustering automatically groups similar feedback,")
    print("helping identify common issues without manual review!")


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
    print("\nEmbedding reviews...")
    system.embed_reviews(SAMPLE_REVIEWS)

    print("\n" + "-"*70)
    print("Use Case 1: Finding Similar Customer Issues")
    print("-"*70)

    new_complaint = "My order hasn't arrived and it's been 3 weeks"
    print(f"\nNew customer complaint: '{new_complaint}'")
    similar_issues = system.find_similar_reviews(new_complaint, top_k=3)

    print("\nSimilar past issues:")
    for i, (review, similarity) in enumerate(similar_issues, 1):
        print(f"  {i}. [{similarity:.3f}] {review['text'][:60]}...")
        print(f"      Product: {review['metadata']['product']}, Rating: {review['metadata']['rating']}⭐")

    print("\n" + "-"*70)
    print("Use Case 2: Recommending Template Responses")
    print("-"*70)
    print("\nBased on similar issues, here's what worked before:")
    print("  - Offer tracking information and delivery updates")
    print("  - Provide expedited shipping for delayed orders")
    print("  - Proactively contact carrier for status")

    print("\n" + "-"*70)
    print("Use Case 3: Identifying Trending Issues")
    print("-"*70)
    clusters = system.cluster_feedback(num_clusters=3)
    print("\nAutomatically identified issue categories:")
    for cluster_id, reviews in clusters.items():
        summary = system.get_cluster_summary(reviews)
        avg_rating = np.mean([r['metadata']['rating'] for r in reviews])
        print(f"  - {summary} (avg rating: {avg_rating:.1f}⭐, {len(reviews)} reviews)")

    print("\n" + "-"*70)
    print("Use Case 4: Prioritizing Agent Training")
    print("-"*70)
    print("\nBased on clustering analysis:")
    print("  - Most reviews cluster around shipping issues")
    print("  - Train agents on delivery problem resolution")
    print("  - Create FAQ for common shipping questions")


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

    # Run all demonstrations
    demonstrate_embedding_creation()
    demonstrate_similarity_search()
    demonstrate_similarity_calculation()
    demonstrate_clustering()
    demonstrate_practical_use_cases()

    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)

    print("\n1. What are Embeddings?")
    print("   - Numerical representations that capture semantic meaning")
    print("   - Similar texts have similar embedding vectors")
    print("   - Enable mathematical operations on text")

    print("\n2. Semantic Search Benefits:")
    print("   - Finds relevant content beyond keyword matching")
    print("   - Understands synonyms and related concepts")
    print("   - 'product broke' matches 'stopped working'")

    print("\n3. Similarity Measurement:")
    print("   - Cosine similarity ranges from -1 to 1")
    print("   - Higher values mean more similar content")
    print("   - Typical threshold: 0.5-0.7 for related content")

    print("\n4. Clustering Applications:")
    print("   - Automatically groups similar feedback")
    print("   - Identifies common themes without manual review")
    print("   - Helps prioritize issues and training needs")

    print("\n5. Customer Service Use Cases:")
    print("   - Find similar past issues for faster resolution")
    print("   - Recommend response templates based on similarity")
    print("   - Identify trending problems across reviews")
    print("   - Automate feedback categorization")
    print("   - Improve agent training based on common themes")

    print("\n6. Cost Considerations:")
    print("   - text-embedding-3-small: ~$0.00002 per 1K tokens")
    print("   - Much cheaper than chat completions")
    print("   - Embeddings can be cached and reused")
    print("   - One-time cost for existing reviews")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
