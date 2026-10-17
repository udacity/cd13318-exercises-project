"""RAGAS evaluation helpers and an optional end-to-end dataset runner."""

import argparse
import asyncio
import json
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ragas import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import BleuScore, Faithfulness, ResponseRelevancy, RougeScore
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    RAGAS_AVAILABLE = True
except (ImportError, AttributeError):
    RAGAS_AVAILABLE = False


def _score(metric, sample) -> float:
    """Run a RAGAS single-turn metric from this synchronous application."""
    value = metric.single_turn_ascore(sample)
    if hasattr(value, "__await__"):
        value = asyncio.run(value)
    if hasattr(value, "value"):
        value = value.value
    return float(value)


def evaluate_response_quality(question: str, answer: str,
                              contexts: List[str],
                              reference_answer: Optional[str] = None) -> Dict[str, float]:
    """Return RAGAS quality scores; reference-based metrics are optional."""
    if not isinstance(question, str) or not question.strip():
        return {"error": "Question must be a non-empty string."}
    if not isinstance(answer, str) or not answer.strip():
        return {"error": "Answer must be a non-empty string."}
    if not isinstance(contexts, list) or not contexts or any(not isinstance(c, str) for c in contexts):
        return {"error": "Contexts must be a non-empty list of strings."}
    if reference_answer is not None and (not isinstance(reference_answer, str) or not reference_answer.strip()):
        return {"error": "reference_answer must be a non-empty string when provided."}
    contexts = [context.strip() for context in contexts if context.strip()]
    if not contexts:
        return {"error": "Contexts must contain at least one non-empty string."}
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS evaluation dependencies are unavailable."}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "Set OPENAI_API_KEY to run RAGAS evaluation."}
    try:
        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0))
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key))
        sample_args = {"user_input": question.strip(), "response": answer.strip(),
                       "retrieved_contexts": contexts}
        if reference_answer:
            sample_args["reference"] = reference_answer.strip()
        sample = SingleTurnSample(**sample_args)
        metrics = {
            "response_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
            "faithfulness": Faithfulness(llm=llm),
        }
        if reference_answer:
            metrics["bleu"] = BleuScore()
            metrics["rouge"] = RougeScore()
        return {name: _score(metric, sample) for name, metric in metrics.items()}
    except Exception as exc:
        return {"error": f"RAGAS evaluation failed: {exc}"}


def load_evaluation_dataset(path: str) -> List[Dict[str, str]]:
    """Load a JSON list of question records from evaluation_dataset.txt."""
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not questions:
        raise ValueError("Dataset must contain a non-empty list of questions.")
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Dataset entry {index} must include a non-empty question.")
    return questions


def evaluate_dataset(dataset_path: str, chroma_dir: str, collection_name: str,
                     api_key: str, model: str = "gpt-4o-mini", top_k: int = 4) -> Dict:
    """Retrieve, answer, and evaluate each dataset question, then compute means."""
    from llm_client import generate_response
    from rag_client import format_context, initialize_rag_system, retrieve_documents

    os.environ["OPENAI_API_KEY"] = api_key
    collection = initialize_rag_system(chroma_dir, collection_name)
    results = []
    scores_by_metric: Dict[str, List[float]] = {}
    for item in load_evaluation_dataset(dataset_path):
        question = item["question"]
        retrieved = retrieve_documents(collection, question, top_k, item.get("mission"))
        docs = (retrieved or {}).get("documents", [[]])[0]
        metadata = (retrieved or {}).get("metadatas", [[]])[0]
        context = format_context(docs, metadata)
        answer = generate_response(api_key, question, context, [], model)
        scores = evaluate_response_quality(question, answer, docs, item.get("expected_answer"))
        for name, value in scores.items():
            if isinstance(value, (float, int)):
                scores_by_metric.setdefault(name, []).append(float(value))
        results.append({"question": question, "answer": answer, "scores": scores})
    return {"results": results,
            "aggregate": {name: statistics.mean(values) for name, values in scores_by_metric.items() if values}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS against an evaluation question set")
    parser.add_argument("--dataset", default=str(Path(__file__).with_name("evaluation_dataset.txt")))
    parser.add_argument("--chroma-dir", default="./chroma_db_openai")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    if not args.openai_key:
        parser.error("provide --openai-key or set OPENAI_API_KEY")
    try:
        print(json.dumps(evaluate_dataset(args.dataset, args.chroma_dir, args.collection_name,
                                          args.openai_key, args.model, args.top_k), indent=2))
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
