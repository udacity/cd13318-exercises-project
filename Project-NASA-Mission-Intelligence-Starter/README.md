# NASA RAG Chat Project - Student Learning Version

A hands-on learning project for building a Retrieval-Augmented Generation (RAG) system with real-time evaluation capabilities. This project teaches students to create a complete RAG pipeline from document processing to interactive chat interface.

## 🎯 Learning Objectives

By completing this project, students will learn to:
- Build document embedding pipelines with ChromaDB and OpenAI
- Implement RAG retrieval systems with semantic search
- Create LLM client integrations with conversation management
- Develop real-time evaluation systems using RAGAS metrics
- Build interactive chat interfaces with Streamlit
- Handle error scenarios and edge cases in production systems

## 📁 Project Structure

```
/
├── chat.py                 # Main Streamlit chat application (TODO-based)
├── embedding_pipeline.py   # ChromaDB embedding pipeline (TODO-based)
├── llm_client.py           # OpenAI LLM client wrapper (TODO-based)
├── rag_client.py           # RAG system client (TODO-based)
├── ragas_evaluator.py      # RAGAS evaluation metrics (TODO-based)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Basic understanding of Python, APIs, and vector databases
- Familiarity with machine learning concepts

### Installation

1. **Open the project folder.** The Udacity workspace opens here automatically.
   *(If you cloned the GitHub repo instead, `cd Project-NASA-Mission-Intelligence-Starter` first.)*


2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### RAGAS 0.4.3 — VertexAI Import Compatibility Fix

#### Issue

```bash
python -W ignore -c "import ragas; print('RAGAS:', ragas.__version__)"
```

RAGAS `0.4.3` imports `ChatVertexAI` and `VertexAI` from the older `langchain-community` package paths.

With newer LangChain versions, these integrations are provided by `langchain-google-vertexai`, causing:

```text
ImportError: cannot import name 'ChatVertexAI'
```

#### Fix

Update the imports in:

```text
/opt/venv/lib/python3.10/site-packages/ragas/llms/base.py
```

Replace:

```python
from langchain_community.chat_models.vertexai import ChatVertexAI
from langchain_community.llms import VertexAI
```

with:

```python
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAI
```

Using `sed`:

```bash
sed -i 's/from langchain_community.chat_models.vertexai import ChatVertexAI/from langchain_google_vertexai import ChatVertexAI/' /opt/venv/lib/python3.10/site-packages/ragas/llms/base.py

sed -i 's/from langchain_community.llms import VertexAI/from langchain_google_vertexai import VertexAI/' /opt/venv/lib/python3.10/site-packages/ragas/llms/base.py
```

#### Verification

Verify that the imports were updated:

```bash
grep -n "VertexAI" /opt/venv/lib/python3.10/site-packages/ragas/llms/base.py
```

Expected:

```text
12:from langchain_google_vertexai import ChatVertexAI
13:from langchain_google_vertexai import VertexAI
43:    ChatVertexAI,
44:    VertexAI,
```

Verify the VertexAI integration:

```bash
python -c "from langchain_google_vertexai import ChatVertexAI; print('Vertex import OK')"
```

Finally, verify RAGAS:

```bash
python -W ignore -c "import ragas; print('RAGAS:', ragas.__version__)"
```

Expected:

```text
RAGAS: 0.4.3
```

> **Note:** This modifies the installed RAGAS package under `/opt/venv`. The change may need to be reapplied if the virtual environment or RAGAS package is reinstalled or upgraded.

![RAGAS package fix](fix_ragas.png)

## 📚 Learning Path

This project follows a structured learning approach where each file contains TODO comments guiding you through the implementation. Complete the files in this recommended order:

### **Phase 1: Core Infrastructure**

#### 1. **LLM Client (`llm_client.py`)** - *Estimated Time: 2-3 hours*
**What you'll learn:**
- OpenAI Chat Completions API integration
- System prompt engineering for domain expertise
- Conversation history management
- Context integration strategies
- Model parameter tuning (temperature, max_tokens)

**Key TODOs:**
- Define system prompt for NASA expertise
- Set context in messages
- Add chat history management
- Create OpenAI Client
- Send request to OpenAI and return response

#### 2. **RAG Client (`rag_client.py`)** - *Estimated Time: 3-4 hours*
**What you'll learn:**
- ChromaDB backend discovery and connection
- Semantic search with metadata filtering
- Document retrieval optimization
- Context formatting for LLM consumption

**Key TODOs:**
- Discover available ChromaDB collections
- Initialize RAG system with database connections
- Implement document retrieval with optional filtering
- Format retrieved documents into structured context

#### 3. **Embedding Pipeline (`embedding_pipeline.py`)** - *Estimated Time: 6-8 hours*
**What you'll learn:**
- Document processing and text chunking strategies
- OpenAI embeddings generation
- ChromaDB collection management
- Metadata extraction and organization
- Batch processing and error handling
- Command-line interface development

**Key TODOs:**
- Initialize OpenAI client and ChromaDB
- Implement intelligent text chunking with overlap
- Create document management methods
- Build metadata extraction from file paths
- Implement batch document processing
- Create command-line interface

### **Phase 2: Evaluation and Interface**

#### 4. **RAGAS Evaluator (`ragas_evaluator.py`)** - *Estimated Time: 2-3 hours*
**What you'll learn:**
- Response quality evaluation metrics
- RAGAS framework integration
- Multi-dimensional assessment (relevancy, faithfulness, precision)
- Evaluation data structure management

**Key TODOs:**
- Create evaluator LLM and embeddings
- Define evaluation metrics instances
- Evaluate responses using multiple metrics
- Return comprehensive evaluation results

#### 5. **Chat Application (`chat.py`)** - *Estimated Time: 4-5 hours*
**What you'll learn:**
- Streamlit web application development
- Real-time evaluation integration
- User interface design for RAG systems
- Session state management
- Configuration and settings management

**Key TODOs:**
- Integrate all components (RAG, LLM, evaluation)
- Build interactive chat interface
- Implement real-time quality metrics display
- Handle user configuration and backend selection

## 🛠️ Implementation Guidelines

### **TODO-Based Learning Approach**
Each file contains strategically placed TODO comments that guide you through:
1. **Understanding the purpose** of each function/method
2. **Implementing core logic** step by step
3. **Handling edge cases** and error scenarios
4. **Integrating components** effectively

### **Code Quality Standards**
- Follow Python PEP 8 style guidelines
- Add comprehensive error handling
- Include informative logging statements
- Write clear docstrings for all functions
- Use type hints for better code clarity

### **Testing Strategy**
- Test each component individually before integration
- Use small datasets for initial testing
- Verify API connections before processing large batches
- Test edge cases (empty files, network errors, invalid inputs)

## 📊 Data Requirements

### **Expected Data Structure**
The system expects NASA document data organized in folders:
```
data_text/
├── apollo11/           # Apollo 11 mission documents
│   ├── *.txt          # Text files with mission data
├── apollo13/           # Apollo 13 mission documents
│   ├── *.txt          # Text files with mission data
└── challenger/         # Challenger mission documents
    ├── *.txt          # Text files with mission data
```

### **Supported Document Types**
- Plain text files (.txt)
- Mission transcripts
- Technical documents
- Audio transcriptions
- Flight plans and procedures

## 🧪 Testing Your Implementation

### **Component Testing**

1. **Test LLM Client**:
   ```python
   from llm_client import generate_response
   response = generate_response(api_key, "What was Apollo 11?", "", [])
   print(response)
   ```

2. **Test RAG Client**:
   ```python
   from rag_client import discover_chroma_backends
   backends = discover_chroma_backends()
   print(backends)
   ```

3. **Test Embedding Pipeline**:
   ```bash
   python embedding_pipeline.py --openai-key YOUR_KEY --stats-only
   ```

4. **Test Evaluation**:
   ```python
   from ragas_evaluator import evaluate_response_quality
   scores = evaluate_response_quality("question", "answer", ["context"])
   print(scores)
   ```

### **Integration Testing**

1. **Run the complete pipeline**:
   ```bash
   # Process documents
   python embedding_pipeline.py --openai-key YOUR_KEY --data-path ./data_text
   
   # Launch chat interface
   streamlit run chat.py
   ```

## 🎓 Learning Checkpoints

### **Checkpoint 1: Basic Functionality**
- [ ] LLM client generates responses
- [ ] RAG client discovers ChromaDB backends
- [ ] Embedding pipeline processes sample files
- [ ] Evaluation system calculates basic metrics

### **Checkpoint 2: Integration**
- [ ] Components work together seamlessly
- [ ] Chat interface loads and responds to queries
- [ ] Real-time evaluation displays metrics
- [ ] Error handling works correctly

### **Checkpoint 3: Advanced Features**
- [ ] Mission-specific filtering works
- [ ] Conversation history is maintained
- [ ] Batch processing handles large datasets
- [ ] Performance is acceptable for interactive use

## 🚨 Common Challenges and Solutions

### **API Integration Issues**
- **Problem**: OpenAI API key errors
- **Solution**: Verify key is set correctly and has sufficient credits

### **ChromaDB Connection Issues**
- **Problem**: Collection not found errors
- **Solution**: Run embedding pipeline first to create collections

### **Memory and Performance Issues**
- **Problem**: Out of memory during processing
- **Solution**: Reduce batch sizes and chunk sizes

### **Evaluation Errors**
- **Problem**: RAGAS evaluation fails
- **Solution**: Ensure all dependencies are installed and contexts are properly formatted

## 📈 Success Metrics

Your implementation is successful when:
1. **Functionality**: All components work individually and together
2. **User Experience**: Chat interface is responsive and intuitive
3. **Quality**: Responses are relevant and well-sourced
4. **Evaluation**: Metrics provide meaningful quality assessment
5. **Robustness**: System handles errors gracefully
6. **Performance**: Response times are acceptable for interactive use

## 🔧 Configuration Options

### **Embedding Pipeline**
- Chunk size and overlap settings
- Batch processing parameters
- Update modes for existing documents
- Embedding model selection

### **LLM Client**
- Model selection (gpt-4.1-mini, gpt-4.1, gpt-4o)
- Temperature and creativity settings
- Maximum token limits
- Conversation history length

### **RAG System**
- Number of documents to retrieve
- Mission-specific filtering options
- Similarity thresholds

### **Evaluation System**
- Metric selection and weighting
- Evaluation frequency settings
- Display preferences

## 🏆 Extension Opportunities

Once you complete the basic implementation, consider these enhancements:

1. **Advanced Retrieval**: Implement hybrid search (semantic + keyword)
2. **Multi-modal Support**: Add support for images and audio
3. **Performance Optimization**: Add caching and parallel processing
4. **Advanced Evaluation**: Implement custom metrics for domain-specific quality
5. **Deployment**: Containerize and deploy to cloud platforms
6. **Monitoring**: Add comprehensive logging and monitoring
7. **Security**: Implement authentication and rate limiting

## 📚 Learning Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG System Design Patterns](https://python.langchain.com/docs/use_cases/question_answering/)

## 🤝 Getting Help

If you encounter issues:
1. Check the TODO comments for guidance
2. Review error messages carefully
3. Test components individually
4. Verify API keys and dependencies
5. Check data format and structure

## 📝 Submission Guidelines

When submitting your completed project:
1. Ensure all TODO items are implemented
2. Test the complete workflow end-to-end
3. Include a brief report on challenges faced and solutions found
4. Document any additional features or improvements you added
5. Provide sample queries and expected responses

---

**Good luck with your RAG system implementation!** This project will give you hands-on experience with modern AI application development, from data processing to user interface design. Take your time with each component and don't hesitate to experiment with different approaches and parameters.
