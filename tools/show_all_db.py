"""Show all data in the WarmStart database."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database
from src.models.schema import Artifact, EvolutionRun, PromptCandidate, Generation

def main():
    db = get_database()
    
    print("\n" + "="*100)
    print("WARMSTART DATABASE - COMPLETE CONTENTS")
    print("="*100)
    
    # ===== ARTIFACTS =====
    with db.session() as session:
        artifacts = session.query(Artifact).order_by(Artifact.created_at.desc()).all()
        print(f"\nARTIFACTS (Prompt DNA Patterns): {len(artifacts)} total")
        print("-" * 100)
        
        for i, a in enumerate(artifacts, 1):
            score = f"{a.effectiveness_score:.3f}" if a.effectiveness_score else "N/A"
            print(f"\n{i:2d}. ID: {a.id} | Type: {a.artifact_type.upper():8s} | Score: {score}")
            print(f"    Name: {a.name or 'N/A'}")
            print(f"    Content: {a.content[:120]}")
            if len(a.content) > 120:
                print(f"             {a.content[120:240]}")
            # Show provenance to the EvolutionRun if available
            if getattr(a, 'source_run_id', None):
                run = session.get(EvolutionRun, a.source_run_id)
                if run:
                    print(f"    From Run: {run.run_id} | Domain: {run.domain or 'N/A'} | Started: {run.started_at}")
            if a.description:
                desc = a.description[:100] + "..." if len(a.description) > 100 else a.description
                print(f"    Description: {desc}")
            if a.domain_tags:
                print(f"    Domains: {', '.join(a.domain_tags)}")
            if a.task_type:
                print(f"    Task Type: {a.task_type[:80]}...")
            print(f"    Usage: {a.usage_count or 0} times | Success: {a.success_count or 0}")
            print(f"    Created: {a.created_at}")
    
    # ===== EVOLUTION RUNS =====
    with db.session() as session:
        runs = session.query(EvolutionRun).order_by(EvolutionRun.started_at.desc()).all()
        print(f"\n\nEVOLUTION RUNS (Optimization Sessions): {len(runs)} total")
        print("-" * 100)
        
        for i, r in enumerate(runs, 1):
            print(f"\n{i:2d}. Run ID: {r.run_id}")
            print(f"    Domain: {r.domain or 'N/A'} | Task: {r.task_description[:80] if r.task_description else 'N/A'}...")
            best_score = f"{r.best_training_score:.3f}" if r.best_training_score else "N/A"
            print(f"    Best Training Score: {best_score} | " +
                  f"Generation: {r.current_generation or 0} | " +
                  f"Candidates: {r.total_candidates_evaluated or 0}")
            print(f"    Status: {r.status or 'N/A'}")
            print(f"    Warm Start: {'Yes' if r.used_warm_start else 'No'}")
            if r.completed_at and r.started_at:
                duration = (r.completed_at - r.started_at).total_seconds()
                print(f"    Duration: {duration:.1f}s")
            print(f"    Started: {r.started_at}")
            if r.completed_at:
                print(f"    Completed: {r.completed_at}")
    
    # ===== PROMPT CANDIDATES =====
    with db.session() as session:
        candidates = session.query(PromptCandidate).order_by(PromptCandidate.created_at.desc()).limit(50).all()
        total_candidates = session.query(PromptCandidate).count()
        print(f"\n\nPROMPT CANDIDATES (Individual Variations): {total_candidates} total (showing last 50)")
        print("-" * 100)
        
        for i, c in enumerate(candidates, 1):
            print(f"\n{i:2d}. ID: {c.id} | Run: {c.run_id}")
            gen_num = c.generation_number if hasattr(c, 'generation_number') else 'N/A'
            fitness = f"{c.fitness:.3f}" if c.fitness else "N/A"
            print(f"    Generation: {gen_num} | Fitness: {fitness}")
            print(f"    Prompt: {c.prompt_text[:100] if c.prompt_text else 'N/A'}...")
            if c.evaluation_feedback:
                print(f"    Feedback: {c.evaluation_feedback[:80]}...")
            if c.is_champion:
                print(f"    ** CHAMPION **")
            print(f"    Created: {c.created_at}")
    
    # ===== SUMMARY STATS =====
    with db.session() as session:
        total_artifacts = session.query(Artifact).count()
        total_runs = session.query(EvolutionRun).count()
        total_candidates = session.query(PromptCandidate).count()
        
        pattern_count = session.query(Artifact).filter(Artifact.artifact_type == 'pattern').count()
        mutation_count = session.query(Artifact).filter(Artifact.artifact_type == 'mutation').count()
        champion_count = session.query(Artifact).filter(Artifact.artifact_type == 'champion').count()
        
        completed_runs = session.query(EvolutionRun).filter(EvolutionRun.status == 'completed').count()
        
        avg_score = session.query(Artifact.effectiveness_score).filter(
            Artifact.effectiveness_score.isnot(None)
        ).all()
        if avg_score:
            avg = sum(s[0] for s in avg_score if s[0]) / len([s for s in avg_score if s[0]])
        else:
            avg = 0
    
    print("\n\n" + "="*100)
    print("DATABASE SUMMARY STATISTICS")
    print("="*100)
    print(f"\nArtifacts:")
    print(f"  • Total: {total_artifacts}")
    print(f"  • Patterns: {pattern_count}")
    print(f"  • Mutations: {mutation_count}")
    print(f"  • Champions: {champion_count}")
    print(f"  • Average Effectiveness Score: {avg:.3f}")
    print(f"\nRuns:")
    print(f"  • Total: {total_runs}")
    print(f"  • Completed: {completed_runs}")
    print(f"\nCandidates:")
    print(f"  • Total Generated: {total_candidates}")
    print(f"  • Average per Run: {total_candidates/total_runs if total_runs > 0 else 0:.1f}")
    
    print("\n" + "="*100)
    print("Database inspection complete!")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
