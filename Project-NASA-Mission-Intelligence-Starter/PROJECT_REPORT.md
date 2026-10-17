# NASA Mission Intelligence RAG Project Report

## Project summary

This project provides a retrieval-augmented chat application over the supplied Apollo 11, Apollo 13, and Challenger text documents. It uses OpenAI embeddings, a persistent ChromaDB collection, an OpenAI chat model, and RAGAS response relevancy and faithfulness metrics.

## Implemented components

- `embedding_pipeline.py` chunks source text with configurable chunk size and overlap, stores mission and filepath metadata, and supports `skip`, `update`, and `replace` update modes. It also supports collection statistics and a test query.
- `rag_client.py` discovers persistent ChromaDB collections, embeds questions for semantic retrieval, supports mission filtering, and formats deduplicated, attributed context.
- `llm_client.py` sends conversation history and retrieved context to an OpenAI chat model with instructions to answer from evidence and cite source labels.
- `ragas_evaluator.py` returns response relevancy and faithfulness scores, reports invalid input clearly, and supports reference-based BLEU and ROUGE scores in batch evaluation.
- `chat.py` brings retrieval, chat generation, mission selection, and optional live evaluation together in Streamlit.
- `evaluation_dataset.txt` contains six sample questions and model-written expected answers covering overview, emergency, disaster analysis, crew, technical procedures, and timeline. Each answer includes the source files used as its basis.

## Local verification

- Python syntax compilation passed for the five project Python files.
- The chunker passed a runtime check for its maximum-size limit and exact configured overlap across 35 chunks.
- `evaluation_dataset.txt` loaded as valid JSON with six questions across six categories; malformed evaluator input returned a clear error.
- Live embedding, retrieval, chat, and RAGAS evaluation were not run in this environment. No `OPENAI_API_KEY` was configured, so those steps could not reach the OpenAI API. The default `python` is 3.14.3 (Python 3.12 is also installed); install the pinned project dependencies in the intended Python environment before running the application.

## Running the project

From this project directory, install dependencies and provide an OpenAI API key in your shell. Then run:

```bash
pip install -r requirements.txt
python embedding_pipeline.py --data-path ./data_text --openai-key YOUR_OPENAI_API_KEY --update-mode skip
streamlit run chat.py
python ragas_evaluator.py --dataset evaluation_dataset.txt --chroma-dir ./chroma_db_openai --collection-name nasa_space_missions_text --openai-key YOUR_OPENAI_API_KEY
```

The dataset runner makes OpenAI calls for each question and its evaluations. Configure the Chroma directory and collection name to match the embedding pipeline if you changed their defaults.
