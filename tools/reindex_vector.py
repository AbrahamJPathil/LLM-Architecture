"""
Backfill existing SQL artifacts into the vector database (Chroma) for semantic retrieval.

Usage:
    # Activate your venv first (preferably Python 3.11+ with Chroma installed)
    python tools/reindex_vector.py --limit 1000

If Chroma is unavailable, this script will log a warning and exit.
"""
from __future__ import annotations

import argparse
from typing import List

from src.models.database import get_database
from src.models.schema import Artifact
from src.patterns.pattern_library import (
    PatternLibrary,
    ArtifactRecord,
    ARTIFACT_TYPE_PATTERN,
    ARTIFACT_TYPE_MUTATION,
)
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def row_to_record(row: Artifact) -> ArtifactRecord:
    atype = (row.artifact_type or "").upper()
    return ArtifactRecord(
        artifact_id=str(row.id),
        artifact_type=atype,
        content=row.content or "",
        description=row.description,
        source_domain=(row.domain_tags[0] if row.domain_tags else None),
        task_description=row.task_type,
        effectiveness_score=row.effectiveness_score,
        name=row.name,
        extra={},
    )


def main():
    parser = argparse.ArgumentParser(description="Reindex SQL artifacts into Chroma vector DB")
    parser.add_argument("--limit", type=int, default=1000, help="Max artifacts to index")
    args = parser.parse_args()

    setup_logging(log_level="INFO", log_format="console")

    # Initialize library (will detect Chroma)
    lib = PatternLibrary()
    if not getattr(lib, "_use_chroma", False):
        logger.warning("Chroma vector DB is not available; install chromadb & sentence-transformers and rerun.")
        return

    # Pull artifacts from SQL
    db = get_database()
    with db.session() as session:
        rows: List[Artifact] = (
            session.query(Artifact)
            .order_by(Artifact.id.desc())
            .limit(args.limit)
            .all()
        )

    if not rows:
        logger.info("No artifacts found to index.")
        return

    # Convert and upsert
    records = [row_to_record(r) for r in rows]
    batch_size = 200
    total = 0
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        lib.upsert(chunk)
        total += len(chunk)
        logger.info("Indexed chunk", size=len(chunk), total=total)

    logger.info("Reindex complete", total_indexed=total)


if __name__ == "__main__":
    main()
