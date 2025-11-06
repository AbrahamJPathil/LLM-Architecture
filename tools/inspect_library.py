"""Inspect the Prompt Pattern Library to see accumulated DNA artifacts."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database
from src.models.schema import Artifact, EvolutionRun

def main():
    db = get_database()
    with db.session() as session:
        total = session.query(Artifact).count()
        artifacts = session.query(Artifact).order_by(Artifact.created_at.desc()).limit(15).all()
        
        print(f"\n{'='*80}")
        print(f"PROMPT PATTERN LIBRARY - Total artifacts: {total}")
        print(f"{'='*80}\n")
        
        if not artifacts:
            print("No artifacts stored yet. Run some optimizations first!\n")
            return
        
        for i, a in enumerate(artifacts, 1):
            score = f"{a.effectiveness_score:.3f}" if a.effectiveness_score else "N/A"
            print(f"{i:2d}. [{a.artifact_type.upper():8s}] Score: {score:5s} | {a.name or 'Unnamed'}")
            print(f"    Content: {(a.content[:100] + '...' if len(a.content) > 100 else a.content)}")
            if a.domain_tags:
                print(f"    Domains: {', '.join(a.domain_tags)}")
            if a.description:
                print(f"    Description: {a.description[:80]}...")
            # Show provenance: which run produced this artifact (if available)
            if a.source_run_id:
                run = session.query(EvolutionRun).filter(EvolutionRun.id == a.source_run_id).first()
                if run:
                    print(f"    From Run: {run.run_id} | Domain: {run.domain} | Started: {run.started_at}")
            print()

if __name__ == "__main__":
    main()
