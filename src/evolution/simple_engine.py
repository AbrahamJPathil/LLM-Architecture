"""
SimpleEvolutionEngine: Evolve prompts without test cases.
Uses direct quality evaluation instead of test case execution.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from src.core.prompt_challenger import PromptChallenger
from src.core.prompt_quality_evaluator import PromptQualityEvaluator
from src.evolution.population import Population, PopulationMember
from src.evolution.results import EvolutionConfig, EvolutionResult, GenerationStats
from src.utils.cost_tracker import CostTracker
from src.utils.logging import get_logger, bind_context, clear_context
from src.patterns.pattern_library import (
    PatternLibrary,
    ArtifactRecord,
    ARTIFACT_TYPE_MUTATION,
    ARTIFACT_TYPE_PATTERN,
    make_artifact_id,
)
from src.patterns.extractor import PatternExtractor
from src.models.database import get_database
from src.models.schema import EvolutionRun, RunStatus

logger = get_logger(__name__)


class SimpleEvolutionEngine:
    """
    Simplified evolution engine that works without test cases.
    Evaluates prompt quality directly using LLM judge.
    """
    
    def __init__(
        self,
        domain: str,
        task_description: str,
        initial_prompt: str,
        user_context: Optional[str],
        config: EvolutionConfig,
        rag_enabled: bool = True,
        rag_top_k: int = 5,
    ):
        """
        Initialize SimpleEvolutionEngine.
        
        Args:
            domain: Domain name (e.g., 'legal', 'medical')
            task_description: What the prompt should accomplish
            initial_prompt: Starting prompt to optimize
            user_context: Optional context about the use case
            config: Evolution configuration
        """
        self.domain = domain
        self.task_description = task_description
        self.initial_prompt = initial_prompt
        self.user_context = user_context
        self.config = config
        self.rag_enabled = rag_enabled
        self.rag_top_k = rag_top_k
        
        # Generate run ID
        self.run_id = f"run_{domain}_{uuid.uuid4().hex[:8]}"
        
        # Initialize components
        self.evaluator = PromptQualityEvaluator(
            domain=domain,
            tier="tier1",  # Use good LLM for evaluation
            task_description=task_description,
            user_context=user_context
        )
        
        self.challenger = PromptChallenger(
            tier="tier3",
            domain=domain,
            task_description=task_description,
            user_context=user_context
        )  # Use fast LLM for mutations
        
        # API rate limiting
        self.api_semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent_mutations))
        
        # Cost tracking
        self.cost_tracker = CostTracker(
            name=self.run_id,
            max_cost_limit=config.max_cost_usd
        )
        
        # Initialize population
        self.population = Population(
            size=config.population_size,
            elite_percentage=config.elite_percentage
        )
        
        # Add initial prompt
        self.population.create_member(initial_prompt)
        
        # Tracking
        self.generation_stats: List[GenerationStats] = []
        self.best_fitness_history: List[float] = []
        self.total_candidates_evaluated = 0
        
        # Bind logging context
        bind_context(run_id=self.run_id, domain=domain)
        
        # RAG Warm-start components
        self.pattern_library = PatternLibrary()
        self.pattern_extractor = PatternExtractor()
        self.rag_artifacts: List[ArtifactRecord] = []

        # Database handle for provenance (best-effort)
        self._db = get_database()
        self._run_db_id: Optional[int] = None
        try:
            # Create a minimal EvolutionRun record to link artifacts later
            with self._db.session() as session:
                run_row = EvolutionRun(
                    run_id=self.run_id,
                    domain=self.domain,
                    task_description=self.task_description,
                    config=(self.config.model_dump() if hasattr(self.config, "model_dump") else getattr(self.config, "__dict__", {})),
                    status=RunStatus.PENDING,
                    current_generation=0,
                    total_candidates_evaluated=0,
                    total_tokens_used=0,
                    total_cost_usd=0.0,
                    used_warm_start=False,
                )
                session.add(run_row)
                session.flush()
                self._run_db_id = run_row.id
        except Exception:
            # Non-fatal: proceed without DB provenance if initialization fails
            self._run_db_id = None

        logger.info(
            "SimpleEvolutionEngine initialized",
            task=task_description,
            population_size=1
        )
    
    async def _fill_initial_population_async(self):
        """Fill population to target size with parallel mutations."""
        current_size = len(self.population.members)
        needed = self.config.population_size - current_size
        
        if needed <= 0:
            return
        
        # Retrieve warm-start artifacts (RAG) once
        if self.rag_enabled:
            try:
                # Keep query focused to improve SQL LIKE fallback hit-rate
                query = f"{self.task_description}"
                self.rag_artifacts = self.pattern_library.query(
                    query_text=query,
                    top_k=min(self.rag_top_k, max(1, needed * 2)),
                    filters={}
                )
            except Exception as e:
                logger.warning(f"RAG retrieval failed, proceeding cold: {e}")
                self.rag_artifacts = []
        else:
            self.rag_artifacts = []

        logger.info("=" * 80)
        if self.rag_artifacts:
            logger.info(f"🧬 MUTATION PHASE: Warm-start seeding with {len(self.rag_artifacts)} artifacts; generating {needed} initial mutations")
            try:
                preview = ", ".join(
                    [
                        f"{a.artifact_type}:{(a.name or a.content)[:40]}" for a in self.rag_artifacts
                    ][:10]
                )
                if preview:
                    logger.info("Warm-start artifacts", artifacts=preview)
            except Exception:
                pass
        else:
            logger.info(f"🧬 MUTATION PHASE: Generating {needed} initial mutations in parallel (cold)")
        logger.info("=" * 80)
        
        async def generate_one_mutation(i):
            async with self.api_semaphore:
                # If we have RAG artifacts, try to drive the first few mutations explicitly
                mutation_instruction = None
                rag_slice = None
                if i < len(self.rag_artifacts):
                    a = self.rag_artifacts[i]
                    if a.artifact_type == ARTIFACT_TYPE_MUTATION:
                        mutation_instruction = a.content
                    elif a.artifact_type == ARTIFACT_TYPE_PATTERN:
                        # Convert pattern to a concrete instruction
                        name = a.name or a.content
                        desc = a.description or ""
                        mutation_instruction = f"Apply the pattern '{name}'. {desc} Ensure the improved prompt incorporates this pattern explicitly."
                    rag_slice = [
                        {
                            "artifact_type": a.artifact_type,
                            "name": a.name or a.content,
                            "description": a.description,
                        }
                    ]
                mutation = self.challenger.mutate(
                    self.initial_prompt,
                    mutation_instruction=mutation_instruction,
                    rag_artifacts=rag_slice,
                )
            return mutation
        
        # Generate mutations in parallel
        mutation_tasks = [generate_one_mutation(i) for i in range(needed)]
        mutations = await asyncio.gather(*mutation_tasks)
        
        # Add to population
        for mutation in mutations:
            self.population.create_member(
                mutation.new_prompt,
                parent_hashes=[self.population._compute_hash(self.initial_prompt)],
                mutation_strategy=mutation.strategy
            )
        
        logger.info(f"Population filled to {len(self.population.members)} members")
    
    async def evolve(self) -> EvolutionResult:
        """
        Run the evolution loop.
        
        Returns:
            EvolutionResult with champion and statistics
        """
        start_time = time.time()
        
        # Fill initial population
        await self._fill_initial_population_async()
        
        logger.info("=" * 60)
        logger.info(f"Starting evolution: {self.run_id}")
        logger.info(f"Task: {self.task_description}")
        logger.info("=" * 60)
        
        stopping_reason = "max_generations_reached"
        
        try:
            for gen in range(self.config.max_generations):
                logger.info("\n" + "=" * 80)
                logger.info(f"📊 GENERATION {gen + 1}/{self.config.max_generations}")
                logger.info("=" * 80)
                
                gen_start = time.time()
                
                # Evaluate unevaluated members
                await self._evaluate_population()
                
                # Mark elite members
                self.population.mark_elite()
                
                # Check stopping criteria
                stop, reason = self._check_stopping_criteria(gen)
                if stop:
                    stopping_reason = reason
                    logger.info(f"Stopping: {reason}")
                    break
                
                # Record generation stats
                stats = self.population.get_statistics()
                gen_duration = time.time() - gen_start
                
                gen_stats = GenerationStats(
                    generation=gen,
                    population_size=stats["size"],
                    avg_fitness=stats["avg_fitness"],
                    best_fitness=stats["best_fitness"],
                    worst_fitness=stats["worst_fitness"],
                    diversity=stats["diversity"],
                    tokens_used=self.cost_tracker.total_tokens,
                    cost_usd=self.cost_tracker.total_cost,
                    duration_seconds=gen_duration,
                    prod_check_performed=False,
                    prod_mismatch_rate=None
                )
                
                self.generation_stats.append(gen_stats)
                self.best_fitness_history.append(stats["best_fitness"])
                
                logger.info("=" * 80)
                logger.info(
                    f"📈 Generation {gen} Summary: "
                    f"avg={stats['avg_fitness']:.3f}, "
                    f"best={stats['best_fitness']:.3f}, "
                    f"diversity={stats['diversity']:.3f}, "
                    f"cost=${self.cost_tracker.total_cost:.4f}"
                )
                logger.info("=" * 80)
                
                # Generate next generation (if not last)
                if gen < self.config.max_generations - 1:
                    await self._generate_next_generation()
        
        finally:
            clear_context()
        
        # Get champion
        champion = self.population.get_best()
        
        duration = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info(f"Evolution complete: {self.run_id}")
        logger.info(f"Champion fitness: {champion.fitness_score:.3f}")
        logger.info(f"Total cost: ${self.cost_tracker.total_cost:.4f}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info("=" * 60)
        
        # Create result object
        result = EvolutionResult(
            run_id=self.run_id,
            domain=self.domain,
            task_description=self.task_description,
            champion_prompt=champion.prompt_text if champion else "",
            champion_fitness=champion.fitness_score if champion else 0.0,
            generations_completed=len(self.generation_stats),
            total_candidates_evaluated=self.total_candidates_evaluated,
            total_tokens_used=self.cost_tracker.total_tokens,
            total_cost_usd=self.cost_tracker.total_cost,
            duration_seconds=duration,
            generation_stats=self.generation_stats,
            stopping_reason=stopping_reason,
            config=self.config
        )
        
        # Persist Prompt DNA (best-effort)
        if champion:
            self._persist_prompt_dna(champion)

        # Update EvolutionRun record (best-effort)
        try:
            with self._db.session() as session:
                row = session.query(EvolutionRun).filter(EvolutionRun.run_id == self.run_id).first()
                if row:
                    row.status = RunStatus.COMPLETED
                    row.current_generation = len(self.generation_stats)
                    row.best_training_score = (champion.fitness_score if champion else None)
                    row.total_candidates_evaluated = self.total_candidates_evaluated
                    row.total_tokens_used = self.cost_tracker.total_tokens
                    row.total_cost_usd = float(self.cost_tracker.total_cost)
                    row.used_warm_start = bool(self.rag_artifacts)
                    row.completed_at = None  # keep default; DB default handles timestamps if configured
        except Exception:
            pass

        return result
    
    async def _evaluate_population(self):
        """Evaluate all unevaluated members with controlled concurrency."""
        unevaluated = self.population.get_unevaluated()
        
        if not unevaluated:
            logger.info("All members already evaluated")
            return
        
        logger.info("=" * 80)
        logger.info(f"⚖️  JUDGE EVALUATION PHASE: Evaluating {len(unevaluated)} prompts")
        logger.info("=" * 80)
        
        # Evaluate in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_evals)
        
        async def evaluate_with_limit(member):
            async with semaphore:
                await self._evaluate_single_member(member)
                
                # Check if we hit target
                if (self.config.early_stop_on_target and 
                    member.fitness_score >= self.config.target_score):
                    logger.info(
                        f"🎯 Target score {self.config.target_score} achieved! "
                        f"Fitness: {member.fitness_score:.3f}"
                    )
                    return True
                return False
        
        # Evaluate all
        tasks = [evaluate_with_limit(member) for member in unevaluated]
        results = await asyncio.gather(*tasks)
        
        # Check if target was hit
        if any(results):
            logger.info("Target score reached, stopping evaluations")
        
        # Log best
        best = self.population.get_best()
        if best:
            logger.info(f"Best fitness: {best.fitness_score:.3f}")
    
    async def _evaluate_single_member(self, member: PopulationMember):
        """Evaluate a single population member."""
        
        # Evaluate prompt quality directly
        quality_score = await self.evaluator.evaluate_prompt_async(
            prompt=member.prompt_text,
            reference_prompt=self.initial_prompt
        )
        
        # Update member
        member.fitness_score = quality_score.overall_score
        
        # Store detailed feedback in metadata
        member.metadata = {
            "clarity": quality_score.clarity,
            "specificity": quality_score.specificity,
            "structure": quality_score.structure,
            "completeness": quality_score.completeness,
            "effectiveness": quality_score.effectiveness,
            "strengths": quality_score.strengths,
            "weaknesses": quality_score.weaknesses,
            "suggestions": quality_score.suggestions
        }
        
        self.total_candidates_evaluated += 1
        
        logger.debug(
            f"Evaluated member",
            hash=member.prompt_hash[:8],
            fitness=member.fitness_score
        )
    
    def _check_stopping_criteria(self, generation: int) -> tuple[bool, str]:
        """Check if we should stop evolution."""
        
        # Check max generations
        if generation >= self.config.max_generations - 1:
            return True, "max_generations_reached"
        
        # Check target score
        best = self.population.get_best()
        if best and best.fitness_score >= self.config.target_score:
            return True, f"target_score_achieved ({best.fitness_score:.3f})"
        
        # Check cost budget
        if self.cost_tracker.is_over_budget():
            return True, f"cost_budget_exceeded (${self.cost_tracker.total_cost:.2f})"
        
        # Check plateau
        if len(self.best_fitness_history) >= self.config.plateau_generations:
            recent = self.best_fitness_history[-self.config.plateau_generations:]
            if max(recent) - min(recent) < 0.01:
                return True, f"plateau_detected ({self.config.plateau_generations} gens)"
        
        return False, ""
    
    async def _generate_next_generation(self):
        """Generate next generation through selection and mutation."""
        logger.info("=" * 80)
        logger.info("🧬 MUTATION PHASE: Generating next generation")
        logger.info("=" * 80)
        
        # Get elite to preserve
        elite = self.population.get_elite()
        
        # Calculate how many new members we need
        needed = self.config.population_size - len(elite)
        
        # Select parents
        import random
        selected_parents = []
        for i in range(needed):
            parent = self.population.tournament_selection(self.config.tournament_size)
            selected_parents.append(parent)
        
        # Get best prompt for reference
        best_member = self.population.get_best()
        
        # Generate mutations in parallel with feedback
        async def generate_mutation(parent):
            async with self.api_semaphore:
                # Build feedback string from metadata
                linguistic_feedback = None
                if hasattr(parent, 'metadata') and parent.metadata:
                    meta = parent.metadata
                    feedback_parts = []
                    
                    # Add score info
                    feedback_parts.append(f"Current score: {parent.fitness_score:.2f}")
                    
                    # Add weaknesses
                    if meta.get('weaknesses'):
                        feedback_parts.append("Weaknesses: " + "; ".join(meta['weaknesses']))
                    
                    # Add suggestions
                    if meta.get('suggestions'):
                        feedback_parts.append("Suggestions: " + "; ".join(meta['suggestions']))
                    
                    # Compare to best if not the best
                    if best_member and parent.prompt_hash != best_member.prompt_hash:
                        feedback_parts.append(f"Best prompt score: {best_member.fitness_score:.2f}")
                        feedback_parts.append(f"Best prompt: {best_member.prompt_text}")
                    
                    linguistic_feedback = "\n".join(feedback_parts)
                
                mutation = self.challenger.mutate(
                    parent.prompt_text,
                    linguistic_feedback=linguistic_feedback,
                    # Provide a small sample of top RAG artifacts as guidance
                    rag_artifacts=[
                        {
                            "artifact_type": a.artifact_type,
                            "name": (a.name or a.content),
                            "description": a.description,
                        }
                        for a in (self.rag_artifacts[:3] if self.rag_artifacts else [])
                    ]
                )
            return mutation, parent
        
        mutation_tasks = [generate_mutation(parent) for parent in selected_parents]
        mutation_results = await asyncio.gather(*mutation_tasks)
        
        # Create new members using create_member to compute hash
        new_members = []
        for mutation, parent in mutation_results:
            new_member = self.population.create_member(
                prompt_text=mutation.new_prompt,
                parent_hashes=[parent.prompt_hash],
                mutation_strategy=mutation.strategy
            )
            new_member.generation = parent.generation + 1
            # Store the generator prompt as mutation DNA on the member for lineage tracking
            try:
                if not hasattr(new_member, 'metadata') or new_member.metadata is None:
                    new_member.metadata = {}
                new_member.metadata["mutation_prompt"] = mutation.generator_prompt
                new_member.metadata["rag_used"] = bool(self.rag_artifacts)
            except Exception:
                pass
            new_members.append(new_member)
        
        # Replace population (keep elite + new members)
        self.population.members = elite + new_members
        
        logger.info(f"Next generation: {len(elite)} elite + {len(new_members)} new = {len(self.population.members)} total")

    def _persist_prompt_dna(self, champion: PopulationMember):
        """Store extracted patterns and the winning mutation-prompt (if available) to the PatternLibrary."""
        try:
            artifacts: List[ArtifactRecord] = []

            # 1) Extract PATTERN artifacts from the champion prompt
            patterns = self.pattern_extractor.extract(
                champion_prompt=champion.prompt_text,
                domain=self.domain,
                task_description=self.task_description,
            )
            for p in patterns:
                artifacts.append(
                    ArtifactRecord(
                        artifact_id=make_artifact_id(),
                        artifact_type=ARTIFACT_TYPE_PATTERN,
                        content=(p.get("pattern_name") or "").upper().replace(" ", "_"),
                        name=p.get("pattern_name"),
                        description=p.get("description"),
                        source_domain=self.domain,
                        task_description=self.task_description,
                        effectiveness_score=champion.fitness_score or 0.0,
                        extra={
                            "champion_hash": champion.prompt_hash,
                            "source_run_id": self._run_db_id,
                            "run_id": self.run_id,
                        },
                    )
                )

            # 2) Store MUTATION artifact if we have the generator prompt from the last mutation
            mutation_prompt = None
            if isinstance(champion.metadata, dict):
                mutation_prompt = champion.metadata.get("mutation_prompt")
            if mutation_prompt:
                artifacts.append(
                    ArtifactRecord(
                        artifact_id=make_artifact_id(),
                        artifact_type=ARTIFACT_TYPE_MUTATION,
                        content=str(mutation_prompt),
                        name=None,
                        description="Mutation instruction that produced the champion (or its parent)",
                        source_domain=self.domain,
                        task_description=self.task_description,
                        effectiveness_score=champion.fitness_score or 0.0,
                        extra={
                            "champion_hash": champion.prompt_hash,
                            "source_run_id": self._run_db_id,
                            "run_id": self.run_id,
                        },
                    )
                )

            if artifacts:
                self.pattern_library.upsert(artifacts)
                logger.info("Stored Prompt DNA artifacts", count=len(artifacts))
        except Exception as e:
            logger.error(f"Failed to persist Prompt DNA: {e}")
