"""
PatternLibrary: Lightweight RAG store for Prompt DNA artifacts.

Stores and retrieves two artifact types:
- PATTERN: Reusable structural patterns (name + description)
- MUTATION: Effective mutation instructions that led to strong prompts

Primary backend: ChromaDB + sentence-transformers embeddings (local, persisted)
Fallback backend: JSONL append-only store with simple substring matching

Safe to use even when dependencies are missing (degrades gracefully).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import get_logger
from src.models.database import get_database
from src.models.schema import Artifact

logger = get_logger(__name__)


ARTIFACT_TYPE_PATTERN = "PATTERN"
ARTIFACT_TYPE_MUTATION = "MUTATION"


@dataclass
class ArtifactRecord:
    """A single Prompt DNA artifact stored in the library."""

    artifact_id: str
    artifact_type: str  # PATTERN | MUTATION
    content: str  # For PATTERN: canonical label (e.g., JSON_SCHEMA_OUTPUT). For MUTATION: the mutation instruction text
    description: Optional[str]
    source_domain: Optional[str]
    task_description: Optional[str]
    effectiveness_score: Optional[float] = None
    name: Optional[str] = None  # For PATTERN: human-readable name
    extra: Optional[Dict[str, Any]] = None  # free-form (e.g., champion_hash, generation)

    @property
    def embedding_text(self) -> str:
        """Text used for semantic indexing."""
        parts = [
            self.artifact_type or "",
            self.name or "",
            self.content or "",
            self.description or "",
            self.task_description or "",
            self.source_domain or "",
        ]
        return "\n".join([p for p in parts if p])


class PatternLibrary:
    """RAG-backed artifact store with graceful fallback when Chroma is unavailable."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir or Path.cwd() / "pattern_library")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.persist_dir / "artifacts.jsonl"
        # Database (system of record)
        self._db = get_database()

        # Try to initialize Chroma + embedder
        self._use_chroma = False
        self._client = None
        self._collection = None
        self._embedder = None

        try:
            import chromadb  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name="prompt_dna",
                metadata={"hnsw:space": "cosine"},
            )
            # Small, fast local embedder
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_chroma = True
            logger.info("PatternLibrary initialized (ChromaDB)", persist_dir=str(self.persist_dir))
        except Exception as e:
            self._use_chroma = False
            logger.warning(
                f"PatternLibrary running in JSONL fallback mode (no Chroma): {e}",
                persist_dir=str(self.persist_dir),
            )

    # ------------------------- Write API -------------------------
    def upsert(self, artifacts: List[ArtifactRecord]) -> None:
        """Insert or update a list of artifacts."""
        if not artifacts:
            return

        if self._use_chroma:
            try:
                ids = [a.artifact_id or uuid.uuid4().hex for a in artifacts]
                docs = [a.embedding_text for a in artifacts]
                metas = [
                    {
                        "artifact_type": a.artifact_type,
                        "content": a.content,
                        "description": a.description,
                        "source_domain": a.source_domain,
                        "task_description": a.task_description,
                        "effectiveness_score": a.effectiveness_score,
                        "name": a.name,
                        "extra": (json.dumps(a.extra) if isinstance(a.extra, dict) else (a.extra if isinstance(a.extra, (str, int, float, bool)) else None)),
                    }
                    for a in artifacts
                ]
                # Compute embeddings locally
                embeddings = self._embed(a.embedding_text for a in artifacts)
                self._collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
            except Exception as e:
                logger.error(f"Chroma upsert failed, falling back to DB/JSONL: {e}")
        # Write to SQL system-of-record
        try:
            with self._db.session() as session:
                for a in artifacts:
                    existing = (
                        session.query(Artifact)
                        .filter(Artifact.artifact_type == (a.artifact_type or "").lower())
                        .filter(Artifact.content == a.content)
                        .filter(Artifact.task_type == (a.task_description or None))
                        .first()
                    )
                    if existing:
                        # update metadata and effectiveness
                        if a.name and not existing.name:
                            existing.name = a.name
                        if a.description and not existing.description:
                            existing.description = a.description
                        try:
                            if a.effectiveness_score is not None:
                                if existing.effectiveness_score is None:
                                    existing.effectiveness_score = float(a.effectiveness_score)
                                else:
                                    existing.effectiveness_score = max(
                                        float(existing.effectiveness_score), float(a.effectiveness_score)
                                    )
                        except Exception:
                            pass
                        # merge domain tag
                        if a.source_domain:
                            tags = set(existing.domain_tags or [])
                            tags.add(a.source_domain)
                            existing.domain_tags = list(tags)
                        existing.usage_count = (existing.usage_count or 0) + 1
                    else:
                        row = Artifact(
                            artifact_type=(a.artifact_type or "").lower(),
                            content=a.content,
                            name=a.name,
                            description=a.description,
                            domain_tags=[a.source_domain] if a.source_domain else [],
                            task_type=a.task_description,
                            effectiveness_score=a.effectiveness_score,
                            usage_count=0,
                            success_count=0,
                            example_transformations=(a.extra or {}).get("example_transformations"),
                            is_approved=False,
                        )
                        # Best-effort provenance linkage if provided in extra
                        try:
                            src_run = (a.extra or {}).get("source_run_id")
                            if src_run:
                                row.source_run_id = int(src_run)
                        except Exception:
                            pass
                        session.add(row)
                session.flush()
        except Exception as e:
            logger.error(f"SQL upsert failed: {e}")

        # Append to JSONL as audit log (optional)
        self._append_jsonl(artifacts)

    def _append_jsonl(self, artifacts: List[ArtifactRecord]) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            for a in artifacts:
                if not a.artifact_id:
                    a.artifact_id = uuid.uuid4().hex
                f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")

    # ------------------------- Read API -------------------------
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ArtifactRecord]:
        """Retrieve most relevant artifacts for the given query."""
        filters = filters or {}

        if self._use_chroma:
            try:
                query_emb = self._embed([query_text])[0]
                where = {}
                # Map simple filters to metadata fields
                for k in ["artifact_type", "source_domain"]:
                    if k in filters:
                        where[k] = filters[k]
                res = self._collection.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                    where=where or None,
                )
                return self._to_records(res)
            except Exception as e:
                logger.error(f"Chroma query failed, using fallback: {e}")

        # Fallback: try SQL LIKE-based search and rank
        try:
            sql_results = self._sql_fallback_query(query_text, top_k, filters)
            if sql_results:
                return sql_results
        except Exception as e:
            logger.error(f"SQL fallback query failed: {e}")

        # Last resort: naive substring search over JSONL
        return self._jsonl_fallback_query(query_text, top_k, filters)

    def _to_records(self, chroma_res: Dict[str, Any]) -> List[ArtifactRecord]:
        records: List[ArtifactRecord] = []
        ids = chroma_res.get("ids", [[]])[0]
        metas = chroma_res.get("metadatas", [[]])[0]
        docs = chroma_res.get("documents", [[]])[0]

        for i, meta in enumerate(metas):
            extra = meta.get("extra")
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except Exception:
                    pass
            records.append(
                ArtifactRecord(
                    artifact_id=ids[i],
                    artifact_type=meta.get("artifact_type"),
                    content=meta.get("content"),
                    description=meta.get("description"),
                    source_domain=meta.get("source_domain"),
                    task_description=meta.get("task_description"),
                    effectiveness_score=meta.get("effectiveness_score"),
                    name=meta.get("name"),
                    extra=extra if isinstance(extra, dict) else {},
                )
            )
        return records

    def _sql_fallback_query(self, query_text: str, top_k: int, filters: Dict[str, Any]) -> List[ArtifactRecord]:
        q = f"%{query_text}%"
        with self._db.session() as session:
            query = session.query(Artifact)
            if "artifact_type" in filters and filters.get("artifact_type"):
                query = query.filter(Artifact.artifact_type == str(filters["artifact_type"]).lower())
            if "source_domain" in filters and filters.get("source_domain"):
                # JSON contains check
                query = query.filter(Artifact.domain_tags.contains([filters["source_domain"]]))
            query = query.filter(
                (Artifact.content.ilike(q)) |
                (Artifact.name.ilike(q)) |
                (Artifact.description.ilike(q)) |
                (Artifact.task_type.ilike(q))
            ).limit(max(top_k * 3, top_k))
            rows = query.all()

        query_l = query_text.lower()
        ranked: List[Tuple[int, ArtifactRecord]] = []
        for r in rows:
            text = "\n".join([
                r.artifact_type or "",
                r.name or "",
                r.content or "",
                r.description or "",
                r.task_type or "",
                ",".join(r.domain_tags or []),
            ]).lower()
            score = text.count(query_l)
            if score <= 0:
                score = 1
            ranked.append((score, ArtifactRecord(
                artifact_id=str(r.id),
                artifact_type=(r.artifact_type or '').upper(),
                content=r.content,
                description=r.description,
                source_domain=(r.domain_tags[0] if r.domain_tags else None),
                task_description=r.task_type,
                effectiveness_score=r.effectiveness_score,
                name=r.name,
                extra={},
            )))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in ranked[:top_k]]

    def _jsonl_fallback_query(self, query_text: str, top_k: int, filters: Dict[str, Any]) -> List[ArtifactRecord]:
        if not self.jsonl_path.exists():
            return []
        matched: List[Tuple[int, ArtifactRecord]] = []
        q = query_text.lower()
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if filters:
                        if any(data.get(k) != v for k, v in filters.items()):
                            continue
                    text = "\n".join(
                        [
                            str(data.get("artifact_type", "")),
                            str(data.get("name", "")),
                            str(data.get("content", "")),
                            str(data.get("description", "")),
                            str(data.get("task_description", "")),
                            str(data.get("source_domain", "")),
                        ]
                    ).lower()
                    score = text.count(q)  # crude match score
                    if score > 0:
                        matched.append(
                            (
                                score,
                                ArtifactRecord(
                                    artifact_id=data.get("artifact_id", uuid.uuid4().hex),
                                    artifact_type=data.get("artifact_type"),
                                    content=data.get("content"),
                                    description=data.get("description"),
                                    source_domain=data.get("source_domain"),
                                    task_description=data.get("task_description"),
                                    effectiveness_score=data.get("effectiveness_score"),
                                    name=data.get("name"),
                                    extra=data.get("extra", {}),
                                ),
                            )
                        )
                except Exception:
                    continue
        matched.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matched[:top_k]]

    # ------------------------- Utilities -------------------------
    def _embed(self, texts) -> List[List[float]]:
        assert self._embedder is not None
        return self._embedder.encode(list(texts), normalize_embeddings=True).tolist()


def make_artifact_id() -> str:
    return uuid.uuid4().hex
