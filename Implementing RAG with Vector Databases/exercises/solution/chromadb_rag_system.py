# ChromaDB RAG System Implementation
# This script demonstrates how to build a Retrieval-Augmented Generation system using ChromaDB
# It covers vector database operations, embeddings, and intelligent document retrieval

import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import pandas as pd
import time
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime
import uuid
import os
from pathlib import Path

# Configuration for different embedding strategies and retrieval approaches
# Each configuration optimizes for different use cases and performance characteristics
EMBEDDING_CONFIGS = {
    # OpenAI embeddings - high quality but higher cost
    "openai_embeddings": {
        "provider": "openai",
        "model": "text-embedding-3-small",  # Cost-effective OpenAI embedding model
        "dimensions": 1536,                 # Standard dimension size
        "description": "OpenAI embeddings with excellent semantic understanding"
    },
    # Alternative: sentence-transformers for local processing
    "local_embeddings": {
        "provider": "sentence_transformers", 
        "model": "all-MiniLM-L6-v2",       # Lightweight local model
        "dimensions": 384,                  # Smaller dimension for efficiency
        "description": "Local embeddings for cost-effective processing"
    }
}

# ChromaDB collection configurations for different use cases
# Each collection type is optimized for specific document types and retrieval patterns
COLLECTION_CONFIGS = {
    # Technical documentation collection
    "tech_docs": {
        "name": "technical_documentation",
        "metadata_fields": ["source", "category", "difficulty", "last_updated"],
        "description": "Technical documentation with structured metadata"
    },
    # FAQ collection for customer support
    "faq_support": {
        "name": "faq_customer_support", 
        "metadata_fields": ["category", "priority", "department", "tags"],
        "description": "FAQ database for customer support automation"
    },
    # Knowledge base for general information
    "knowledge_base": {
        "name": "general_knowledge",
        "metadata_fields": ["topic", "source", "confidence", "date_added"],
        "description": "General knowledge base for information retrieval"
    }
}

# Sample documents for testing different types of content and retrieval scenarios
# These represent realistic business documents that would be stored in a vector database
SAMPLE_DOCUMENTS = {
    "tech_docs": [
        {
            "id": "tech_001",
            "content": "ChromaDB is an open-source vector database designed for AI applications. It provides efficient storage and retrieval of high-dimensional vectors, making it ideal for semantic search, recommendation systems, and RAG implementations. ChromaDB supports multiple embedding functions and offers both in-memory and persistent storage options.",
            "metadata": {
                "source": "ChromaDB Documentation",
                "category": "Database",
                "difficulty": "Intermediate",
                "last_updated": "2024-01-15"
            }
        },
        {
            "id": "tech_002", 
            "content": "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval. By retrieving relevant documents before generation, RAG systems can provide more accurate, up-to-date, and contextually relevant responses while reducing hallucinations and improving factual accuracy.",
            "metadata": {
                "source": "AI Research Papers",
                "category": "Machine Learning",
                "difficulty": "Advanced",
                "last_updated": "2024-02-01"
            }
        },
        {
            "id": "tech_003",
            "content": "Vector embeddings are numerical representations of text that capture semantic meaning. Modern embedding models like OpenAI's text-embedding-3-small can convert text into high-dimensional vectors where similar concepts are positioned closer together in the vector space, enabling semantic search capabilities.",
            "metadata": {
                "source": "Embedding Guide",
                "category": "NLP",
                "difficulty": "Intermediate", 
                "last_updated": "2024-01-20"
            }
        }
    ],
    "faq_support": [
        {
            "id": "faq_001",
            "content": "Q: How do I reset my password? A: To reset your password, click on the 'Forgot Password' link on the login page, enter your email address, and follow the instructions sent to your email. The reset link expires after 24 hours for security purposes.",
            "metadata": {
                "category": "Account Management",
                "priority": "High",
                "department": "IT Support",
                "tags": "password, security, login"
            }
        },
        {
            "id": "faq_002",
            "content": "Q: What are your business hours? A: Our customer support is available Monday through Friday, 9 AM to 6 PM EST. For urgent technical issues, our emergency support line is available 24/7 for premium customers.",
            "metadata": {
                "category": "General Information",
                "priority": "Medium",
                "department": "Customer Service",
                "tags": "hours, support, availability"
            }
        },
        {
            "id": "faq_003",
            "content": "Q: How do I upgrade my subscription? A: You can upgrade your subscription by logging into your account, navigating to the 'Billing' section, and selecting 'Upgrade Plan'. Changes take effect immediately, and you'll be prorated for the current billing period.",
            "metadata": {
                "category": "Billing",
                "priority": "High", 
                "department": "Sales",
                "tags": "subscription, billing, upgrade"
            }
        }
    ]
}

class ChromaDBRAGSystem:
    """
    A comprehensive RAG system implementation using ChromaDB for vector storage and retrieval.
    
    This class demonstrates best practices for building production-ready RAG systems,
    including proper error handling, metadata management, and performance optimization.
    """
    
    def __init__(self, embedding_config: str = "openai_embeddings", persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB RAG system with specified configuration.
        
        Args:
            embedding_config (str): Configuration key for embedding strategy
            persist_directory (str): Directory for persistent storage
        """
        self.embedding_config = EMBEDDING_CONFIGS[embedding_config]
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True             # Allow database reset for development
            )
        )
        
        # Initialize OpenAI client for embeddings and generation
        self.openai_client = OpenAI(
            api_key="your-key-here"
        )
        
        # Store active collections for management
        self.collections = {}
        
        print(f"🚀 ChromaDB RAG System initialized")
        print(f"   Embedding Strategy: {self.embedding_config['description']}")
        print(f"   Persist Directory: {persist_directory}")
        print(f"   Available Collections: {len(self.client.list_collections())}")

    def create_collection(self, collection_key: str) -> chromadb.Collection:
        """
        Create a new ChromaDB collection with specified configuration.
        
        This method sets up collections with appropriate embedding functions
        and metadata schemas for different document types.
        
        Args:
            collection_key (str): Key from COLLECTION_CONFIGS
            
        Returns:
            chromadb.Collection: The created collection object
        """
        if collection_key not in COLLECTION_CONFIGS:
            raise ValueError(f"Unknown collection configuration: {collection_key}")
            
        config = COLLECTION_CONFIGS[collection_key]
        collection_name = config["name"]
        
        print(f"\n📁 Creating collection: {collection_name}")
        print(f"   Description: {config['description']}")
        print(f"   Metadata fields: {config['metadata_fields']}")
        
        try:
            # Delete existing collection if it exists (for development)
            try:
                self.client.delete_collection(collection_name)
                print(f"   ♻️  Deleted existing collection")
            except:
                pass
            
            # Create new collection with embedding function
            if self.embedding_config["provider"] == "openai":
                # Use OpenAI embeddings
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=None,  # We'll handle embeddings manually
                    metadata={"description": config["description"]}
                )
            else:
                # Use default ChromaDB embeddings for local processing
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": config["description"]}
                )
            
            self.collections[collection_key] = collection
            print(f"   ✅ Collection created successfully")
            return collection
            
        except Exception as e:
            print(f"   ❌ Error creating collection: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using the configured embedding model.
        
        This method handles both OpenAI and local embedding generation with proper
        error handling and batch processing for efficiency.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        try:
            if self.embedding_config["provider"] == "openai":
                # Use OpenAI embeddings API
                response = self.openai_client.embeddings.create(
                    model=self.embedding_config["model"],
                    input=texts
                )

                embeddings = [embedding.embedding for embedding in response.data]
                print(f"✅ Generated {len(embeddings)} OpenAI embeddings")
                return embeddings
                
            else:
                # Use local sentence transformers (would require additional setup)
                print("⚠️  Local embeddings not implemented in this example")
                # In a real implementation, you would use sentence-transformers here
                # from sentence_transformers import SentenceTransformer
                # model = SentenceTransformer(self.embedding_config["model"])
                # embeddings = model.encode(texts).tolist()
                return []
                
        except Exception as e:
            print(f"❌ Error generating embeddings: {str(e)}")
            raise

    def add_documents(self, collection_key: str, documents: List[Dict]) -> None:
        """
        Add documents to a ChromaDB collection with embeddings and metadata.
        
        This method handles the complete document ingestion pipeline including
        embedding generation, metadata processing, and batch insertion.
        
        Args:
            collection_key (str): Key identifying the target collection
            documents (List[Dict]): List of document dictionaries with content and metadata
        """
        if collection_key not in self.collections:
            raise ValueError(f"Collection {collection_key} not found. Create it first.")
            
        collection = self.collections[collection_key]
        
        print(f"\n📄 Adding {len(documents)} documents to {collection.name}")
        
        # Extract texts and metadata
        texts = [doc["content"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        if not embeddings:
            print("❌ No embeddings generated, skipping document addition")
            return
        
        try:
            # Add documents to collection
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Successfully added {len(documents)} documents")
            print(f"   Collection now contains: {collection.count()} documents")
            
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise

    def search_documents(self, collection_key: str, query: str, n_results: int = 3, 
                        metadata_filter: Optional[Dict] = None) -> Dict:
        """
        Search for relevant documents using semantic similarity.
        
        This method performs vector similarity search with optional metadata filtering
        and returns comprehensive results including similarity scores and metadata.
        
        Args:
            collection_key (str): Key identifying the collection to search
            query (str): Search query text
            n_results (int): Number of results to return
            metadata_filter (Optional[Dict]): Metadata filters to apply
            
        Returns:
            Dict: Search results with documents, distances, and metadata
        """
        if collection_key not in self.collections:
            raise ValueError(f"Collection {collection_key} not found")
            
        collection = self.collections[collection_key]
        
        print(f"\n🔍 Searching collection: {collection.name}")
        print(f"   Query: '{query}'")
        print(f"   Requesting: {n_results} results")
        if metadata_filter:
            print(f"   Filters: {metadata_filter}")
        
        try:
            # Generate embedding for query
            query_embeddings = self.generate_embeddings([query])
            
            if not query_embeddings:
                print("❌ Failed to generate query embedding")
                return {"documents": [], "distances": [], "metadatas": []}
            
            # Perform similarity search
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=metadata_filter,
                include=["documents", "distances", "metadatas"]
            )
            
            print(f"✅ Found {len(results['documents'][0])} relevant documents")
            
            # Format results for better readability
            formatted_results = {
                "query": query,
                "n_results": len(results['documents'][0]),
                "results": []
            }
            
            for i in range(len(results['documents'][0])):
                formatted_results["results"].append({
                    "document": results['documents'][0][i],
                    "similarity_score": 1 - results['distances'][0][i],  # Convert distance to similarity
                    "metadata": results['metadatas'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching documents: {str(e)}")
            return {"documents": [], "distances": [], "metadatas": []}

    def generate_rag_response(self, collection_key: str, query: str, n_context: int = 3,
                            model: str = "gpt-4o-mini") -> Dict:
        """
        Generate a response using Retrieval-Augmented Generation.
        
        This method implements the complete RAG pipeline: retrieval of relevant documents,
        context preparation, and generation of informed responses using the retrieved context.
        
        Args:
            collection_key (str): Collection to search for context
            query (str): User query to answer
            n_context (int): Number of context documents to retrieve
            model (str): OpenAI model to use for generation
            
        Returns:
            Dict: RAG response with context, answer, and metadata
        """
        print(f"\n🤖 Generating RAG response")
        print(f"   Query: '{query}'")
        print(f"   Context documents: {n_context}")
        print(f"   Generation model: {model}")
        
        start_time = time.time()
        
        # Step 1: Retrieve relevant context
        search_results = self.search_documents(collection_key, query, n_context)
        
        if not search_results["results"]:
            return {
                "query": query,
                "answer": "I couldn't find relevant information to answer your question.",
                "context": [],
                "generation_time": 0,
                "context_used": 0
            }
        
        # Step 2: Prepare context for generation
        context_documents = []
        for result in search_results["results"]:
            context_documents.append({
                "content": result["document"],
                "similarity": result["similarity_score"],
                "source": result["metadata"].get("source", "Unknown")
            })
        
        # Step 3: Create prompt with context
        context_text = "\n\n".join([
            f"Document {i+1} (Similarity: {doc['similarity']:.3f}):\n{doc['content']}"
            for i, doc in enumerate(context_documents)
        ])
        
        prompt = f"""Based on the following context documents, please answer the user's question. If the context doesn't contain enough information to answer the question completely, please say so and provide what information you can.

Context Documents:
{context_text}

User Question: {query}

Please provide a comprehensive answer based on the context provided:"""

        try:
            # Step 4: Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=500
            )
            
            generation_time = time.time() - start_time
            
            # Step 5: Format comprehensive response
            rag_response = {
                "query": query,
                "answer": response.choices[0].message.content,
                "context": context_documents,
                "generation_time": round(generation_time, 2),
                "context_used": len(context_documents),
                "model_used": model,
                "tokens_used": response.usage.total_tokens
            }
            
            print(f"✅ RAG response generated successfully")
            print(f"   Generation time: {generation_time:.2f}s")
            print(f"   Context documents used: {len(context_documents)}")
            print(f"   Tokens used: {response.usage.total_tokens}")
            
            return rag_response
            
        except Exception as e:
            print(f"❌ Error generating RAG response: {str(e)}")
            return {
                "query": query,
                "answer": f"Error generating response: {str(e)}",
                "context": context_documents,
                "generation_time": 0,
                "context_used": len(context_documents)
            }

    def display_rag_response(self, rag_response: Dict) -> None:
        """
        Display RAG response in a formatted, readable way.
        
        This method provides a user-friendly presentation of RAG results including
        the generated answer, context sources, and performance metrics.
        
        Args:
            rag_response (Dict): RAG response dictionary from generate_rag_response
        """
        print(f"\n" + "="*80)
        print(f"🤖 RAG RESPONSE")
        print(f"="*80)
        
        print(f"\n❓ QUESTION:")
        print(f"   {rag_response['query']}")
        
        print(f"\n💡 ANSWER:")
        print(f"   {rag_response['answer']}")
        
        print(f"\n📚 CONTEXT SOURCES ({rag_response['context_used']} documents):")
        for i, context in enumerate(rag_response['context']):
            print(f"   {i+1}. Similarity: {context['similarity']:.3f} | Source: {context['source']}")
            print(f"      Preview: {context['content'][:100]}...")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Generation Time: {rag_response['generation_time']}s")
        print(f"   Model Used: {rag_response.get('model_used', 'Unknown')}")
        print(f"   Tokens Used: {rag_response.get('tokens_used', 'Unknown')}")
        print(f"   Context Documents: {rag_response['context_used']}")

def demonstrate_chromadb_rag():
    """
    Comprehensive demonstration of ChromaDB RAG system capabilities.
    
    This function showcases the complete workflow from database setup through
    document ingestion to intelligent query answering using RAG.
    """
    print("🚀 ChromaDB RAG System Demonstration")
    print("="*60)
    
    # Initialize the RAG system
    rag_system = ChromaDBRAGSystem(
        embedding_config="openai_embeddings",
        persist_directory="./demo_chroma_db"
    )
    
    # Create collections for different document types
    print("\n📁 Setting up document collections...")
    rag_system.create_collection("tech_docs")
    rag_system.create_collection("faq_support")
    
    # Add sample documents to collections
    print("\n📄 Adding sample documents...")
    rag_system.add_documents("tech_docs", SAMPLE_DOCUMENTS["tech_docs"])
    rag_system.add_documents("faq_support", SAMPLE_DOCUMENTS["faq_support"])
    
    # Demonstrate different types of queries
    test_queries = [
        {
            "collection": "tech_docs",
            "query": "What is ChromaDB and how does it work?",
            "description": "Technical documentation query"
        },
        {
            "collection": "tech_docs", 
            "query": "How do vector embeddings enable semantic search?",
            "description": "Conceptual understanding query"
        },
        {
            "collection": "faq_support",
            "query": "I forgot my password, how can I reset it?",
            "description": "Customer support query"
        },
        {
            "collection": "faq_support",
            "query": "What are your business hours?",
            "description": "General information query"
        }
    ]
    
    # Execute test queries and display results
    print("\n🔍 Testing RAG system with various queries...")
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*20} TEST QUERY {i}: {test['description']} {'='*20}")
        
        rag_response = rag_system.generate_rag_response(
            collection_key=test["collection"],
            query=test["query"],
            n_context=2
        )
        
        rag_system.display_rag_response(rag_response)
    
    print(f"\n🎉 ChromaDB RAG demonstration completed successfully!")
    print(f"   Collections created: {len(rag_system.collections)}")
    print(f"   Documents processed: {sum(len(docs) for docs in SAMPLE_DOCUMENTS.values())}")
    print(f"   Queries tested: {len(test_queries)}")

# Example usage and testing
if __name__ == "__main__":
    # Run the comprehensive demonstration
    demonstrate_chromadb_rag()
    
    # Additional examples for advanced usage:
    
    # Example 1: Custom metadata filtering
    # rag_system = ChromaDBRAGSystem()
    # rag_system.create_collection("tech_docs")
    # rag_system.add_documents("tech_docs", SAMPLE_DOCUMENTS["tech_docs"])
    # 
    # # Search with metadata filter
    # filtered_results = rag_system.search_documents(
    #     "tech_docs", 
    #     "database information",
    #     metadata_filter={"category": "Database"}
    # )
    
    # Example 2: Batch processing multiple queries
    # queries = ["What is RAG?", "How do embeddings work?", "ChromaDB features"]
    # for query in queries:
    #     response = rag_system.generate_rag_response("tech_docs", query)
    #     rag_system.display_rag_response(response)
