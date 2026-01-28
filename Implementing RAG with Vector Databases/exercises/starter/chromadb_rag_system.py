# ChromaDB RAG System Implementation - Starter Template
# TODO: Complete this script to build a Retrieval-Augmented Generation system using ChromaDB

# TODO: Import necessary libraries
# Hint: You'll need chromadb, openai, pandas, time, json, typing, numpy, datetime, uuid, os, pathlib
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
# TODO: Add remaining imports here


# TODO: Define embedding configurations for different strategies
# Create configurations for OpenAI embeddings and local alternatives
EMBEDDING_CONFIGS = {
    "openai_embeddings": {
        # TODO: Set provider to "openai"
        "provider": "",
        # TODO: Choose OpenAI embedding model (hint: text-embedding-3-small is cost-effective)
        "model": "",
        # TODO: Set dimensions (hint: 1536 for text-embedding-3-small)
        "dimensions": 0,
        "description": "OpenAI embeddings with excellent semantic understanding"
    },
    "local_embeddings": {
        # TODO: Set provider to "sentence_transformers"
        "provider": "",
        # TODO: Choose local model (hint: all-MiniLM-L6-v2 is lightweight)
        "model": "",
        # TODO: Set dimensions (hint: 384 for all-MiniLM-L6-v2)
        "dimensions": 0,
        "description": "Local embeddings for cost-effective processing"
    }
}

# TODO: Define collection configurations for different document types
# Create configurations for technical docs, FAQ support, and knowledge base
COLLECTION_CONFIGS = {
    "tech_docs": {
        # TODO: Set collection name
        "name": "",
        # TODO: Define metadata fields for technical documentation
        "metadata_fields": [],
        "description": "Technical documentation with structured metadata"
    },
    "faq_support": {
        # TODO: Set collection name
        "name": "",
        # TODO: Define metadata fields for FAQ support
        "metadata_fields": [],
        "description": "FAQ database for customer support automation"
    },
    "knowledge_base": {
        # TODO: Set collection name
        "name": "",
        # TODO: Define metadata fields for general knowledge
        "metadata_fields": [],
        "description": "General knowledge base for information retrieval"
    }
}

# TODO: Create sample documents for testing
# Define realistic business documents with content and metadata
SAMPLE_DOCUMENTS = {
    "tech_docs": [
        {
            "id": "tech_001",
            # TODO: Add content about ChromaDB (200-300 words)
            "content": "",
            "metadata": {
                # TODO: Add appropriate metadata
                "source": "",
                "category": "",
                "difficulty": "",
                "last_updated": ""
            }
        },
        {
            "id": "tech_002",
            # TODO: Add content about RAG systems (200-300 words)
            "content": "",
            "metadata": {
                # TODO: Add appropriate metadata
            }
        },
        {
            "id": "tech_003",
            # TODO: Add content about vector embeddings (200-300 words)
            "content": "",
            "metadata": {
                # TODO: Add appropriate metadata
            }
        }
    ],
    "faq_support": [
        {
            "id": "faq_001",
            # TODO: Add FAQ about password reset
            "content": "",
            "metadata": {
                # TODO: Add appropriate metadata for customer support
                "category": "",
                "priority": "",
                "department": "",
                "tags": []
            }
        },
        {
            "id": "faq_002",
            # TODO: Add FAQ about business hours
            "content": "",
            "metadata": {
                # TODO: Add appropriate metadata
            }
        },
        {
            "id": "faq_003",
            # TODO: Add FAQ about subscription upgrade
            "content": "",
            "metadata": {
                # TODO: Add appropriate metadata
            }
        }
    ]
}

class ChromaDBRAGSystem:
    """
    A comprehensive RAG system implementation using ChromaDB for vector storage and retrieval.
    
    TODO: Complete this class to implement a production-ready RAG system with:
    - ChromaDB integration for vector storage
    - Embedding generation and management
    - Document ingestion and retrieval
    - RAG response generation
    """
    
    def __init__(self, embedding_config: str = "openai_embeddings", persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB RAG system with specified configuration.
        
        TODO: Complete this method to:
        1. Store configuration parameters
        2. Initialize ChromaDB client with persistent storage
        3. Initialize OpenAI client for embeddings and generation
        4. Set up collections dictionary
        5. Print initialization status
        
        Args:
            embedding_config (str): Configuration key for embedding strategy
            persist_directory (str): Directory for persistent storage
        """
        # TODO: Store embedding configuration
        self.embedding_config = None
        self.persist_directory = persist_directory
        
        # TODO: Initialize ChromaDB client with persistent storage
        # Hint: Use chromadb.PersistentClient with path and settings
        self.client = None
        
        # TODO: Initialize OpenAI client for embeddings and generation
        # SECURITY NOTE: Use environment variables for API keys in production
        self.openai_client = None
        
        # TODO: Initialize collections dictionary
        self.collections = {}
        
        # TODO: Print initialization status
        print(f"🚀 ChromaDB RAG System initialized")

    def create_collection(self, collection_key: str):
        """
        Create a new ChromaDB collection with specified configuration.
        
        TODO: Complete this method to:
        1. Validate collection_key exists in COLLECTION_CONFIGS
        2. Get collection configuration
        3. Delete existing collection if it exists (for development)
        4. Create new collection with appropriate settings
        5. Store collection in self.collections
        6. Handle errors gracefully
        
        Args:
            collection_key (str): Key from COLLECTION_CONFIGS
            
        Returns:
            chromadb.Collection: The created collection object
        """
        # TODO: Validate collection_key
        if collection_key not in COLLECTION_CONFIGS:
            raise ValueError(f"Unknown collection configuration: {collection_key}")
            
        # TODO: Get configuration and create collection
        config = COLLECTION_CONFIGS[collection_key]
        collection_name = config["name"]
        
        print(f"\n📁 Creating collection: {collection_name}")
        
        try:
            # TODO: Delete existing collection if it exists
            # TODO: Create new collection
            # TODO: Store in self.collections
            # TODO: Print success message
            pass
            
        except Exception as e:
            print(f"   ❌ Error creating collection: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using the configured embedding model.
        
        TODO: Complete this method to:
        1. Handle OpenAI embedding generation
        2. Support local embedding generation (optional)
        3. Include proper error handling
        4. Return list of embedding vectors
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        try:
            if self.embedding_config["provider"] == "openai":
                # TODO: Use OpenAI embeddings API
                # Hint: Use self.openai_client.embeddings.create()
                pass
                
            else:
                # TODO: Handle local embeddings (optional)
                print("⚠️  Local embeddings not implemented in this example")
                return []
                
        except Exception as e:
            print(f"❌ Error generating embeddings: {str(e)}")
            raise

    def add_documents(self, collection_key: str, documents: List[Dict]) -> None:
        """
        Add documents to a ChromaDB collection with embeddings and metadata.
        
        TODO: Complete this method to:
        1. Validate collection exists
        2. Extract texts, IDs, and metadata from documents
        3. Generate embeddings for texts
        4. Add documents to collection with embeddings
        5. Handle errors and provide status updates
        
        Args:
            collection_key (str): Key identifying the target collection
            documents (List[Dict]): List of document dictionaries with content and metadata
        """
        # TODO: Validate collection exists
        if collection_key not in self.collections:
            raise ValueError(f"Collection {collection_key} not found. Create it first.")
            
        # TODO: Get collection and extract document data
        collection = self.collections[collection_key]
        
        print(f"\n📄 Adding {len(documents)} documents to {collection.name}")
        
        # TODO: Extract texts, IDs, and metadata
        # TODO: Generate embeddings
        # TODO: Add documents to collection
        # TODO: Print success status

    def search_documents(self, collection_key: str, query: str, n_results: int = 3, 
                        metadata_filter: Optional[Dict] = None) -> Dict:
        """
        Search for relevant documents using semantic similarity.
        
        TODO: Complete this method to:
        1. Validate collection exists
        2. Generate embedding for query
        3. Perform similarity search with optional metadata filtering
        4. Format and return results with similarity scores
        
        Args:
            collection_key (str): Key identifying the collection to search
            query (str): Search query text
            n_results (int): Number of results to return
            metadata_filter (Optional[Dict]): Metadata filters to apply
            
        Returns:
            Dict: Search results with documents, distances, and metadata
        """
        # TODO: Validate collection exists
        # TODO: Generate query embedding
        # TODO: Perform similarity search
        # TODO: Format and return results
        pass

    def generate_rag_response(self, collection_key: str, query: str, n_context: int = 3,
                            model: str = "gpt-4o-mini") -> Dict:
        """
        Generate a response using Retrieval-Augmented Generation.
        
        TODO: Complete this method to:
        1. Retrieve relevant context documents
        2. Prepare context for generation
        3. Create prompt with context
        4. Generate response using OpenAI
        5. Format comprehensive response with metadata
        
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
        
        # TODO: Retrieve relevant context
        # TODO: Prepare context for generation
        # TODO: Create prompt with context
        # TODO: Generate response using OpenAI
        # TODO: Format and return comprehensive response
        pass

    def display_rag_response(self, rag_response: Dict) -> None:
        """
        Display RAG response in a formatted, readable way.
        
        TODO: Complete this method to:
        1. Display question and answer clearly
        2. Show context sources with similarity scores
        3. Display performance metrics
        4. Format output for readability
        
        Args:
            rag_response (Dict): RAG response dictionary from generate_rag_response
        """
        # TODO: Format and display RAG response
        # Include: question, answer, context sources, performance metrics
        pass

def demonstrate_chromadb_rag():
    """
    Comprehensive demonstration of ChromaDB RAG system capabilities.
    
    TODO: Complete this function to:
    1. Initialize the RAG system
    2. Create collections for different document types
    3. Add sample documents to collections
    4. Test various query types
    5. Display results and performance metrics
    """
    print("🚀 ChromaDB RAG System Demonstration")
    print("="*60)
    
    # TODO: Initialize the RAG system
    # TODO: Create collections
    # TODO: Add sample documents
    # TODO: Test various queries
    # TODO: Display results
    pass

# TODO: Example usage - uncomment and test when ready
# Run the comprehensive demonstration
# demonstrate_chromadb_rag()

# TODO: Additional examples you can implement:
# 
# Example 1: Custom metadata filtering
# rag_system = ChromaDBRAGSystem()
# rag_system.create_collection("tech_docs")
# rag_system.add_documents("tech_docs", SAMPLE_DOCUMENTS["tech_docs"])
# filtered_results = rag_system.search_documents(
#     "tech_docs", 
#     "database information",
#     metadata_filter={"category": "Database"}
# )
#
# Example 2: Batch processing multiple queries
# queries = ["What is RAG?", "How do embeddings work?", "ChromaDB features"]
# for query in queries:
#     response = rag_system.generate_rag_response("tech_docs", query)
#     rag_system.display_rag_response(response)

"""
EXERCISE COMPLETION CHECKLIST:
□ Import all necessary libraries
□ Complete EMBEDDING_CONFIGS with appropriate models and parameters
□ Fill in COLLECTION_CONFIGS with meaningful names and metadata fields
□ Create comprehensive SAMPLE_DOCUMENTS with realistic content
□ Implement ChromaDBRAGSystem.__init__() with proper initialization
□ Complete create_collection() with ChromaDB collection creation
□ Implement generate_embeddings() with OpenAI API integration
□ Complete add_documents() with embedding generation and storage
□ Implement search_documents() with similarity search and filtering
□ Complete generate_rag_response() with full RAG pipeline
□ Implement display_rag_response() with formatted output
□ Complete demonstrate_chromadb_rag() with comprehensive testing
□ Test your implementation with the example usage
□ Add your own API key and test the complete workflow

BONUS CHALLENGES:
□ Add support for local embedding models using sentence-transformers
□ Implement batch processing for large document collections
□ Add metadata-based filtering and advanced search capabilities
□ Create a web interface for the RAG system using Flask or FastAPI
□ Implement document update and deletion functionality
□ Add support for different file formats (PDF, Word, etc.)
□ Create performance monitoring and analytics dashboard
□ Implement user authentication and access control
□ Add support for multi-modal documents (text + images)
□ Create automated document ingestion from external sources
"""
