"""ChromaDB discovery, semantic retrieval, and context formatting."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from openai import OpenAI


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Find Chroma persistent stores below the current project directory."""
    backends: Dict[str, Dict[str, str]] = {}
    current_dir = Path.cwd()
    candidates = set()
    for name in ("chroma_db", "chroma_db_openai", "chroma"):
        candidates.update(p for p in current_dir.glob(f"**/{name}") if p.is_dir())
    candidates.update(p.parent for p in current_dir.glob("**/chroma.sqlite3"))
    for directory in sorted(candidates):
        try:
            client = chromadb.PersistentClient(path=str(directory))
            collections = client.list_collections()
            for item in collections:
                collection_name = item.name if hasattr(item, "name") else str(item)
                collection = client.get_collection(collection_name)
                key = f"{directory.resolve()}::{collection_name}"
                backends[key] = {
                    "directory": str(directory.resolve()),
                    "collection_name": collection_name,
                    "display_name": f"{directory.name} / {collection_name} ({collection.count():,} chunks)",
                }
        except Exception as exc:
            key = f"{directory.resolve()}::unavailable"
            backends[key] = {
                "directory": str(directory.resolve()),
                "collection_name": "",
                "display_name": f"{directory.name} (unavailable: {str(exc)[:80]})",
                "error": str(exc),
            }
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Open an existing persistent collection."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(name=collection_name)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Embed a question and retrieve its nearest chunks, optionally by mission."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if n_results < 1:
        raise ValueError("n_results must be at least 1")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY to create query embeddings")
    metadata = getattr(collection, "metadata", None) or {}
    model = metadata.get("embedding_model") or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding = OpenAI(api_key=api_key).embeddings.create(model=model, input=query).data[0].embedding
    where = None
    if mission_filter and mission_filter.strip().lower() not in {"all", "any"}:
        where = {"mission": mission_filter}
    return collection.query(query_embeddings=[embedding], n_results=n_results, where=where,
                           include=["documents", "metadatas", "distances"])


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Build attributed, deduplicated context for the answer model."""
    if not documents:
        return ""
    parts = ["Retrieved NASA mission sources:"]
    seen = set()
    for i, document in enumerate(documents):
        if not document or document.strip() in seen:
            continue
        seen.add(document.strip())
        metadata = metadatas[i] if i < len(metadatas) else {}
        mission = str(metadata.get("mission", "Unknown mission")).replace("_", " ").title()
        source = metadata.get("file_path") or metadata.get("source", "Unknown source")
        category = str(metadata.get("document_category", "General document")).replace("_", " ").title()
        content = document.strip()
        if len(content) > 6000:
            content = content[:6000].rstrip() + " … [chunk truncated]"
        parts.append(f"\n[Source {len(seen)} | Mission: {mission} | Category: {category} | File: {source}]\n{content}")
    return "\n".join(parts)
