"""
Population management for genetic algorithm.
Handles candidate tracking, selection, and elitism.
"""

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.prompt_verifier import BatchEvaluationResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PopulationMember:
    """A single member of the population (prompt candidate)."""
    
    prompt_text: str
    prompt_hash: str
    
    # Fitness scores
    fitness_score: Optional[float] = None
    evaluation_result: Optional[BatchEvaluationResult] = None
    
    # Lineage tracking
    parent_hashes: List[str] = field(default_factory=list)
    generation: int = 0
    mutation_strategy: Optional[str] = None
    
    # Selection tracking
    is_elite: bool = False
    times_selected: int = 0
    
    # Additional metadata (for judge feedback, quality scores, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_evaluated(self) -> bool:
        """Whether this member has been evaluated."""
        return self.fitness_score is not None
    
    def to_dict(self):
        """Convert to dictionary for storage."""
        return {
            "prompt_text": self.prompt_text,
            "prompt_hash": self.prompt_hash,
            "fitness_score": self.fitness_score,
            "parent_hashes": self.parent_hashes,
            "generation": self.generation,
            "mutation_strategy": self.mutation_strategy,
            "is_elite": self.is_elite,
            "times_selected": self.times_selected
        }


class Population:
    """
    Manages the population of prompt candidates.
    Handles fitness tracking, selection, and elitism.
    """
    
    def __init__(
        self,
        size: int,
        elite_percentage: float = 0.05,
        generation: int = 0
    ):
        """
        Initialize population.
        
        Args:
            size: Population size
            elite_percentage: Percentage of population to preserve as elite
            generation: Starting generation number
        """
        self.size = size
        self.elite_percentage = elite_percentage
        self.elite_count = max(1, int(size * elite_percentage))
        self.generation = generation
        
        self.members: List[PopulationMember] = []
        self._hash_to_member: dict = {}
        
        logger.info(
            "Population initialized",
            size=size,
            elite_count=self.elite_count,
            generation=generation
        )
    
    def add_member(self, member: PopulationMember) -> bool:
        """
        Add a member to the population.
        Returns False if duplicate (same hash exists).
        """
        if member.prompt_hash in self._hash_to_member:
            logger.debug(f"Duplicate member rejected: {member.prompt_hash[:8]}")
            return False
        
        self.members.append(member)
        self._hash_to_member[member.prompt_hash] = member
        member.generation = self.generation
        
        return True
    
    def create_member(
        self,
        prompt_text: str,
        parent_hashes: Optional[List[str]] = None,
        mutation_strategy: Optional[str] = None
    ) -> PopulationMember:
        """Create and add a new member."""
        prompt_hash = self._compute_hash(prompt_text)
        
        member = PopulationMember(
            prompt_text=prompt_text,
            prompt_hash=prompt_hash,
            parent_hashes=parent_hashes or [],
            generation=self.generation,
            mutation_strategy=mutation_strategy
        )
        
        self.add_member(member)
        return member
    
    def _compute_hash(self, prompt_text: str) -> str:
        """Compute hash for prompt deduplication."""
        return hashlib.sha256(prompt_text.encode()).hexdigest()
    
    def get_unevaluated(self) -> List[PopulationMember]:
        """Get all members that haven't been evaluated yet."""
        return [m for m in self.members if not m.is_evaluated]
    
    def get_evaluated(self) -> List[PopulationMember]:
        """Get all members that have been evaluated."""
        return [m for m in self.members if m.is_evaluated]
    
    def get_by_hash(self, prompt_hash: str) -> Optional[PopulationMember]:
        """Get member by hash."""
        return self._hash_to_member.get(prompt_hash)
    
    def sort_by_fitness(self) -> List[PopulationMember]:
        """Get members sorted by fitness (best first)."""
        evaluated = self.get_evaluated()
        return sorted(evaluated, key=lambda m: m.fitness_score or 0.0, reverse=True)
    
    def mark_elite(self):
        """Mark top N members as elite."""
        sorted_members = self.sort_by_fitness()
        
        # Clear existing elite status
        for member in self.members:
            member.is_elite = False
        
        # Mark top N as elite
        for i, member in enumerate(sorted_members[:self.elite_count]):
            member.is_elite = True
            logger.debug(
                f"Elite member {i+1}",
                fitness=member.fitness_score,
                hash=member.prompt_hash[:8]
            )
    
    def get_elite(self) -> List[PopulationMember]:
        """Get elite members."""
        return [m for m in self.members if m.is_elite]
    
    def get_best(self) -> Optional[PopulationMember]:
        """Get best member by fitness."""
        sorted_members = self.sort_by_fitness()
        return sorted_members[0] if sorted_members else None
    
    def get_worst(self) -> Optional[PopulationMember]:
        """Get worst member by fitness."""
        sorted_members = self.sort_by_fitness()
        return sorted_members[-1] if sorted_members else None
    
    def tournament_selection(self, tournament_size: int = 2) -> PopulationMember:
        """
        Select a member using tournament selection.
        
        Args:
            tournament_size: Number of candidates in tournament
            
        Returns:
            Selected member
        """
        evaluated = self.get_evaluated()
        
        if len(evaluated) < tournament_size:
            # Not enough evaluated members, return random
            return random.choice(evaluated if evaluated else self.members)
        
        # Select random tournament candidates
        tournament = random.sample(evaluated, tournament_size)
        
        # Select best from tournament
        winner = max(tournament, key=lambda m: m.fitness_score or 0.0)
        winner.times_selected += 1
        
        return winner
    
    def get_statistics(self) -> dict:
        """Get population statistics."""
        evaluated = self.get_evaluated()
        
        if not evaluated:
            return {
                "size": len(self.members),
                "evaluated_count": 0,
                "avg_fitness": 0.0,
                "best_fitness": 0.0,
                "worst_fitness": 0.0,
                "diversity": 0.0
            }
        
        fitnesses = [m.fitness_score for m in evaluated]
        
        # Calculate diversity (unique prompts / total prompts)
        unique_hashes = len(set(m.prompt_hash for m in self.members))
        diversity = unique_hashes / len(self.members)
        
        return {
            "size": len(self.members),
            "evaluated_count": len(evaluated),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "best_fitness": max(fitnesses),
            "worst_fitness": min(fitnesses),
            "std_fitness": self._std_dev(fitnesses),
            "diversity": diversity,
            "elite_count": len(self.get_elite())
        }
    
    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def trim_to_size(self):
        """
        Trim population to target size, keeping best members.
        Preserves all elite members.
        """
        if len(self.members) <= self.size:
            return
        
        logger.info(
            f"Trimming population from {len(self.members)} to {self.size}"
        )
        
        # Separate elite and non-elite
        elite = self.get_elite()
        non_elite = [m for m in self.members if not m.is_elite]
        
        # Sort non-elite by fitness
        non_elite_sorted = sorted(
            non_elite,
            key=lambda m: m.fitness_score or 0.0,
            reverse=True
        )
        
        # Keep elite + best non-elite up to size
        to_keep = elite + non_elite_sorted[:self.size - len(elite)]
        
        # Update members list
        self.members = to_keep
        self._hash_to_member = {m.prompt_hash: m for m in to_keep}
    
    def next_generation(self):
        """Advance to next generation."""
        self.generation += 1
        logger.info(f"Advanced to generation {self.generation}")
