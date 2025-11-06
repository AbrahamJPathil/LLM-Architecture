"""
Data classes for evolution results.
Separated from engine to avoid circular imports.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    
    generation: int
    population_size: int
    avg_fitness: float
    best_fitness: float
    worst_fitness: float
    diversity: float
    
    tokens_used: int
    cost_usd: float
    duration_seconds: float
    
    prod_check_performed: bool = False
    prod_mismatch_rate: Optional[float] = None


@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""
    
    # Population settings
    population_size: int = 50
    elite_percentage: float = 0.05
    tournament_size: int = 2
    
    # Mutation settings
    mutation_rate: float = 0.8
    crossover_enabled: bool = False
    
    # Stopping criteria
    max_generations: int = 50
    plateau_generations: int = 10
    target_score: float = 0.95
    max_cost_usd: float = 100.0
    early_stop_on_target: bool = True  # Stop immediately when target is hit
    
    # Evaluation
    batch_eval_size: int = 50  # Number of test cases per evaluation
    max_concurrent_evals: int = 5  # Max parallel evaluations to avoid rate limits
    max_concurrent_mutations: int = 3  # Max parallel mutation generations
    max_concurrent_judge: int = 3  # Max parallel judge evaluations
    
    # Cost control
    prod_check_cadence: int = 10  # Check production LLM every N generations
    prod_check_top_n: int = 5
    
    @classmethod
    def from_dict(cls, config: dict) -> "EvolutionConfig":
        """Create from configuration dictionary."""
        evolution_config = config.get("evolution", {})
        cost_control = config.get("cost_control", {})
        performance_config = config.get("performance", {})
        
        return cls(
            population_size=evolution_config.get("population_size", 50),
            elite_percentage=evolution_config.get("elite_percentage", 0.05),
            tournament_size=evolution_config.get("tournament_size", 2),
            mutation_rate=evolution_config.get("mutation_rate", 0.8),
            crossover_enabled=evolution_config.get("crossover_enabled", False),
            max_generations=evolution_config.get("stopping", {}).get("max_generations", 50),
            plateau_generations=evolution_config.get("stopping", {}).get("plateau_generations", 10),
            target_score=evolution_config.get("stopping", {}).get("target_score", 0.95),
            max_cost_usd=evolution_config.get("stopping", {}).get("max_cost_usd", 100.0),
            batch_eval_size=config.get("evaluation", {}).get("batch_eval_size", 50),
            max_concurrent_evals=performance_config.get("max_concurrent_evaluations", 5),
            max_concurrent_mutations=performance_config.get("max_concurrent_mutations", 3),
            max_concurrent_judge=performance_config.get("max_concurrent_judge", 3),
            prod_check_cadence=cost_control.get("prod_check_cadence", 10),
            prod_check_top_n=cost_control.get("prod_check_top_n", 5),
        )


@dataclass
class EvolutionResult:
    """Results of an evolution run."""
    
    run_id: str
    domain: str
    task_description: str
    
    # Best result
    champion_prompt: str
    champion_fitness: float
    
    # Run statistics
    generations_completed: int
    total_candidates_evaluated: int
    total_tokens_used: int
    total_cost_usd: float
    duration_seconds: float
    
    # Per-generation stats
    generation_stats: List[GenerationStats]
    
    # Stopping reason
    stopping_reason: str
    
    # Configuration used
    config: EvolutionConfig
