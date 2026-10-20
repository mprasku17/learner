"""MP2 · Mini-RAG
=================

This file implements a complete RAG pipeline over the Sherlock Holmes corpus.

Pipeline you're building:
    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations

Run sequence:
    pip install -r requirements.txt
    source .env                 # exports your OpenAI + Qdrant credentials
    python mp2_rag.py ingest    # builds the collection (run once)
    python mp2_rag.py ask       # interactive Q&A loop
    python mp2_rag.py validate  # runs against data/predefined_questions.jsonl

Tip: get the CORE pipeline working FIRST (Steps 1-7 below), THEN come back to
polish and add your 3 questions. Don't try to perfect each step before moving
on — you'll learn more from a rough end-to-end loop than a polished half.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)  # load .env file, overriding stale shell values

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR        = Path(__file__).parent / "corpus"
DATA_DIR          = Path(__file__).parent / "data"
COLLECTION_NAME   = "mp2_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"    #OPENAI EMBEDDING MODEL
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"               #OPENAI LLM MODEL
TARGET_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP     = 80    # characters

openai = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)


qdrant = QdrantClient(
    #url=os.getenv("QDRANT_URL"),           # not working as of now
    url="http://localhost:6333",            # installed qdrant locally in docker, so using this url
    api_key=os.getenv("QDRANT_API_KEY"),    # though not needed for local qdrant, but used as it is configured
)


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory.

    Returns a list of dicts, each with: source (filename), title (first line),
    and text (full content).

    """
    documents: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        title = next((line.strip() for line in text.splitlines() if line.strip()), path.stem)
        documents.append({
            "source": path.name,
            "title": title,
            "text": text,
        })
    return documents


# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks.

    Each chunk should be a dict with: source, title, section, text.

    Approach (your choice):
      - Simple: fixed-size windows (split text into N-character chunks with overlap)
      - Smarter: split on paragraph boundaries (\\n\\n), then pack paragraphs
        into chunks up to TARGET_CHUNK_SIZE characters

    The reference solution uses the smarter approach, plus heuristic
    section-header detection (short lines without terminal punctuation).
    Either approach is acceptable.

    """
    chunks: list[dict[str, Any]] = []
    current_section = str(doc["title"])
    section_lines: list[str] = []

    def add_section_chunks(section: str, lines: list[str]) -> None:
        section_text = " ".join(line.strip() for line in lines if line.strip())
        if not section_text:
            return

        start = 0
        while start < len(section_text):
            end = min(start + TARGET_CHUNK_SIZE, len(section_text))
            chunk_text = section_text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "source": doc["source"],
                    "title": doc["title"],
                    "section": section,
                    "text": chunk_text,
                })
            if end == len(section_text):
                break
            start = end - CHUNK_OVERLAP

    lines = str(doc["text"]).splitlines()
    title_seen = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not title_seen:
            title_seen = True
            continue

        is_heading = (
            len(stripped) <= 80
            and not stripped.endswith((".", "!", "?", ":", ";"))
        )
        if is_heading:
            add_section_chunks(current_section, section_lines)
            section_lines = []
            current_section = stripped
        else:
            section_lines.append(stripped)

    add_section_chunks(current_section, section_lines)
    return chunks


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs).

    """
    if not texts:
        return []

    print(f"→ Embedding {len(texts)} texts in batches of 100…")
    vectors: list[list[float]] = []
    batch_size = 100  # stay well under the API's per-request input limit
    for start in range(0, len(texts), batch_size):
        # The API rejects empty strings, so substitute a single space
        batch = [text.strip() or " " for text in texts[start:start + batch_size]]
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        # Sort by index so output order matches input order
        print(f"  → Batch {start // batch_size + 1}: {len(batch)} embeddings")
        vectors.extend(
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        )
    print(f"  → {len(vectors)} embeddings generated")
    if len(vectors) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
    return vectors


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection."""

    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        print(f"→ Deleting existing collection '{COLLECTION_NAME}'…")
        qdrant.delete_collection(collection_name=COLLECTION_NAME)


    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"→ Created collection '{COLLECTION_NAME}' with {EMBEDDING_DIM}-dim vectors.")


# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant.

    """
    if not chunks:
        return

    print(f"→ Embedding and upserting {len(chunks)} chunks into Qdrant…")
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=chunk,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  → Upserted {len(points)} points into collection '{COLLECTION_NAME}'.")


# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query.

    """
    if k <= 0:
        return []

    query_vector = embed_texts([query])[0]
    #print(f"  → Query vector: {query_vector}")
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
    )
   
    return [
        {**hit.payload, "score": hit.score}
        for hit in hits.points
        if hit.payload is not None
    ]

    #print(f"  → Retrieved {len(hits)} hits for query: '{query}'")


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result.

    """
    started_at = time.perf_counter()
    retrieved = retrieve(question, k=k)
    context = "\n\n".join(
        f"[Source: {chunk['title']} — {chunk['section']}]\n{chunk['text']}"
        for chunk in retrieved
    )
    if not context:
        context = "(No relevant excerpts were retrieved.)"

    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nExcerpts:\n{context}",
            },
        ],
    )
    answer_text = response.choices[0].message.content
    if answer_text is None:
        raise RuntimeError("The LLM (chat completion) returned no answer content.")

    citations = [
        {
            "source": chunk["source"],
            "title": chunk["title"],
            "section": chunk["section"],
            "score": chunk["score"],
        }
        for chunk in retrieved
    ]
    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "latency_ms": round((time.perf_counter() - started_at) * 1000),
    }


# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
