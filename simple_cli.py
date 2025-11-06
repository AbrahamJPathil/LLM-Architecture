"""
Simple WarmStart CLI - Optimize prompts without test cases.

Usage:
    python simple_cli.py "Your prompt here" --domain legal
    python simple_cli.py "Your prompt here" --context "Additional context"
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from src.evolution.simple_engine import SimpleEvolutionEngine
from src.evolution.results import EvolutionConfig
from src.models.database import init_database
from src.utils.logging import setup_logging, get_logger
from src.utils.text import safe_print, maybe_strip_emoji
from src.utils.config import get_config

logger = get_logger(__name__)


async def optimize_prompt(
    prompt: str,
    domain: str = "general",
    context: str = None,
    population_size: int = 8,
    generations: int = 5,
    output_file: str = None
):
    """
    Optimize a prompt without test cases.
    
    Args:
        prompt: The prompt to optimize
        domain: Domain name (legal, medical, code, general)
        context: Optional context about the use case
        population_size: Population size for evolution
        generations: Number of generations to run
        output_file: Optional file to save results
    """
    # Initialize database
    init_database()
    
    logger.info("\n" + "=" * 80)
    logger.info(f"WarmStart - Quick Prompt Optimization (No Test Cases Required)")
    logger.info(f"Domain: {domain}")
    logger.info("=" * 80)
    
    # Task description
    task_description = f"Optimize prompt for {domain} domain"
    if context:
        task_description += f" - {context}"
    
    # Create evolution config
    config = get_config(domain=domain if domain != "general" else "general")
    evo_config = EvolutionConfig.from_dict(config.to_dict())
    evo_config.population_size = population_size
    evo_config.max_generations = generations
    evo_config.target_score = 0.90  # High quality target
    evo_config.early_stop_on_target = True
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Population: {evo_config.population_size}")
    logger.info(f"  Generations: {evo_config.max_generations}")
    logger.info(f"  Target Score: {evo_config.target_score}")
    logger.info("")
    
    # Create engine
    # RAG flags from environment (set by CLI flags)
    rag_enabled = os.environ.get("WARMSTART_RAG_ENABLED", "1") == "1"
    try:
        rag_top_k = int(os.environ.get("WARMSTART_RAG_TOP_K", "5"))
    except Exception:
        rag_top_k = 5

    engine = SimpleEvolutionEngine(
        domain=domain,
        task_description=task_description,
        initial_prompt=prompt,
        user_context=context,
        config=evo_config,
        rag_enabled=rag_enabled,
        rag_top_k=rag_top_k,
    )
    
    # Run evolution
    logger.info("\n" + "=" * 80)
    logger.info("🚀 STARTING OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(maybe_strip_emoji("\n📝 YOUR INITIAL PROMPT:"))
    logger.info(maybe_strip_emoji(f"   \"{prompt}\""))
    
    if context:
        logger.info(maybe_strip_emoji("\n📋 CONTEXT:"))
        logger.info(maybe_strip_emoji(f"   \"{context}\""))
    
    logger.info("\n" + "=" * 80)
    
    try:
        result = await engine.evolve()
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("✅ OPTIMIZATION COMPLETE!")
        logger.info("=" * 80)
        
        logger.info(maybe_strip_emoji(f"\n📊 Results:"))
        logger.info(f"  Generations: {result.generations_completed}")
        logger.info(f"  Candidates tested: {result.total_candidates_evaluated}")
        logger.info(f"  Duration: {result.duration_seconds:.1f}s")
        logger.info(f"  Total cost: ${result.total_cost_usd:.4f}")
        logger.info(f"  Stopping reason: {result.stopping_reason}")
        
        # Before and After comparison
        logger.info("\n" + "=" * 80)
        logger.info(maybe_strip_emoji("📊 BEFORE vs AFTER"))
        logger.info("=" * 80)
        
        safe_print("\n" + "=" * 80)
        safe_print("🔴 YOUR ORIGINAL PROMPT:")
        safe_print("=" * 80)
        safe_print(f"\n{prompt}\n")
        
        safe_print("=" * 80)
        safe_print(f"🟢 OPTIMIZED PROMPT (Quality Score: {result.champion_fitness:.3f}):")
        safe_print("=" * 80)
        safe_print(f"\n{result.champion_prompt}\n")
        safe_print("=" * 80)
        
        # Show quality breakdown if available
        champion_member = engine.population.get_best()
        if champion_member and hasattr(champion_member, 'metadata') and champion_member.metadata:
            meta = champion_member.metadata
            safe_print("\n📈 Quality Breakdown:")
            safe_print(f"  Clarity:       {meta.get('clarity', 0):.2f}")
            safe_print(f"  Specificity:   {meta.get('specificity', 0):.2f}")
            safe_print(f"  Structure:     {meta.get('structure', 0):.2f}")
            safe_print(f"  Completeness:  {meta.get('completeness', 0):.2f}")
            safe_print(f"  Effectiveness: {meta.get('effectiveness', 0):.2f}")
            
            if meta.get('strengths'):
                safe_print(f"\n✅ Strengths:")
                for strength in meta['strengths']:
                    safe_print(f"  • {strength}")
            
            if meta.get('weaknesses'):
                safe_print(f"\n⚠️  Areas for Improvement:")
                for weakness in meta['weaknesses']:
                    safe_print(f"  • {weakness}")
        
        logger.info(f"\n📈 Evolution Progress:")
        for gen_stat in result.generation_stats:
            logger.info(
                f"  Gen {gen_stat.generation}: "
                f"best={gen_stat.best_fitness:.3f}, "
                f"avg={gen_stat.avg_fitness:.3f}"
            )
        
        # Save to file if requested
        if output_file:
            output_data = {
                "original_prompt": prompt,
                "optimized_prompt": result.champion_prompt,
                "quality_score": result.champion_fitness,
                "context": context,
                "domain": domain,
                "generations": result.generations_completed,
                "cost_usd": result.total_cost_usd,
                "duration_seconds": result.duration_seconds
            }
            
            if champion_member and hasattr(champion_member, 'metadata'):
                output_data["quality_breakdown"] = champion_member.metadata
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(maybe_strip_emoji(f"\n💾 Results saved to: {output_file}"))
        
        logger.info("\n✅ Done!")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}", exc_info=True)
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WarmStart Simple - Fast Prompt Optimization (No Test Cases)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimization
  python simple_cli.py "Extract key contract terms"
  
  # With domain
  python simple_cli.py "Summarize medical reports" --domain medical
  
  # With context
  python simple_cli.py "Analyze code" --context "Python code reviews for security"
  
  # Larger run
  python simple_cli.py "Your prompt" --population 15 --generations 8
  
  # Save results
  python simple_cli.py "Your prompt" --output results.json
        """
    )
    
    parser.add_argument('prompt', type=str, help='Prompt to optimize')
    parser.add_argument('--domain', type=str, default='general',
                       choices=['legal', 'medical', 'code', 'general'],
                       help='Domain for optimization (default: general)')
    parser.add_argument('--context', type=str, default=None,
                       help='Additional context about your use case')
    parser.add_argument('--population', type=int, default=8,
                       help='Population size (default: 8)')
    parser.add_argument('--generations', type=int, default=5,
                       help='Number of generations (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON)')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock LLM for testing')
    parser.add_argument('--no-emoji', action='store_true',
                       help='Strip emojis from console/log output (Windows-safe)')
    parser.add_argument('--rag-off', action='store_true',
                       help='Disable RAG warm-start retrieval (always cold start)')
    parser.add_argument('--rag-top-k', type=int, default=5,
                       help='Max number of warm-start artifacts to retrieve (default: 5)')
    
    args = parser.parse_args()
    
    # Configure no-emoji mode before logging if requested
    if args.no_emoji:
        os.environ["WARMSTART_NO_EMOJI"] = "1"
    else:
        os.environ.setdefault("WARMSTART_NO_EMOJI", "0")

    # Setup logging
    setup_logging(log_level="INFO", log_format="console")
    
    # Set mock mode if requested
    if args.mock:
        os.environ["WARMSTART_MOCK_LLM"] = "1"
        logger.info("🔧 Mock mode enabled (no real API calls)")
    else:
        os.environ.setdefault("WARMSTART_MOCK_LLM", "0")
    
    # Run optimization
    # Pass RAG flags via environment for simplicity
    if args.rag_off:
        os.environ["WARMSTART_RAG_ENABLED"] = "0"
    else:
        os.environ["WARMSTART_RAG_ENABLED"] = "1"
    os.environ["WARMSTART_RAG_TOP_K"] = str(max(1, args.rag_top_k))

    asyncio.run(optimize_prompt(
        prompt=args.prompt,
        domain=args.domain,
        context=args.context,
        population_size=args.population,
        generations=args.generations,
        output_file=args.output
    ))


if __name__ == "__main__":
    main()
