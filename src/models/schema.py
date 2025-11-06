"""
Database models and schema for WarmStart.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ArtifactType(str, Enum):
    """Types of artifacts that can be stored."""
    PATTERN = "pattern"
    MUTATION = "mutation"
    CHAMPION = "champion"
    TEMPLATE = "template"


class RunStatus(str, Enum):
    """Status of an evolution run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Artifact(Base):
    """
    Reusable patterns, mutations, and champion prompts.
    """
    __tablename__ = "artifacts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    artifact_type = Column(String(50), nullable=False, index=True)
    
    # Content
    content = Column(Text, nullable=False)
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Metadata
    domain_tags = Column(JSON, nullable=False, default=list)  # e.g., ["legal", "contract"]
    task_type = Column(String(100), nullable=True, index=True)
    
    # Effectiveness tracking
    effectiveness_score = Column(Float, nullable=True, index=True)
    usage_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    
    # Source tracking
    source_run_id = Column(Integer, ForeignKey("evolution_runs.id"), nullable=True)
    source_prompt_id = Column(Integer, ForeignKey("prompt_candidates.id"), nullable=True)
    parent_artifact_id = Column(Integer, ForeignKey("artifacts.id"), nullable=True)
    
    # Example transformations (JSON list)
    example_transformations = Column(JSON, nullable=True)
    
    # Approval status
    is_approved = Column(Boolean, default=False)
    is_quarantined = Column(Boolean, default=False)
    reviewed_by = Column(String(100), nullable=True)
    review_notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_run = relationship("EvolutionRun", back_populates="artifacts", foreign_keys=[source_run_id])
    source_prompt = relationship("PromptCandidate", back_populates="artifacts", foreign_keys=[source_prompt_id])
    
    __table_args__ = (
        Index('idx_artifact_domain_score', 'domain_tags', 'effectiveness_score'),
        Index('idx_artifact_type_approved', 'artifact_type', 'is_approved'),
    )


class EvolutionRun(Base):
    """
    A complete evolution experiment.
    """
    __tablename__ = "evolution_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Configuration
    domain = Column(String(100), nullable=False, index=True)
    task_description = Column(Text, nullable=False)
    config = Column(JSON, nullable=False)  # Full config snapshot
    
    # Status
    status = Column(String(50), default=RunStatus.PENDING, index=True)
    current_generation = Column(Integer, default=0)
    
    # Results
    champion_prompt_id = Column(Integer, ForeignKey("prompt_candidates.id"), nullable=True)
    final_validation_score = Column(Float, nullable=True)
    best_training_score = Column(Float, nullable=True)
    
    # Metrics
    total_candidates_evaluated = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    
    # Warm start info
    used_warm_start = Column(Boolean, default=False)
    warm_start_pattern_ids = Column(JSON, nullable=True)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Metadata
    created_by = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relationships
    generations = relationship("Generation", back_populates="run", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="source_run", foreign_keys=[Artifact.source_run_id])
    champion = relationship("PromptCandidate", foreign_keys=[champion_prompt_id])


class Generation(Base):
    """
    A single generation in an evolution run.
    """
    __tablename__ = "generations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("evolution_runs.id"), nullable=False, index=True)
    generation_number = Column(Integer, nullable=False)
    
    # Population metrics
    population_size = Column(Integer, nullable=False)
    avg_score = Column(Float, nullable=True)
    best_score = Column(Float, nullable=True)
    worst_score = Column(Float, nullable=True)
    
    # Diversity metrics
    diversity_score = Column(Float, nullable=True)
    unique_prompts = Column(Integer, nullable=True)
    
    # Cost metrics
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    
    # Production check (if applicable)
    prod_check_performed = Column(Boolean, default=False)
    prod_check_mismatch_rate = Column(Float, nullable=True)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    run = relationship("EvolutionRun", back_populates="generations")
    prompt_candidates = relationship("PromptCandidate", back_populates="generation", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_gen_run_number', 'run_id', 'generation_number'),
    )


class PromptCandidate(Base):
    """
    A single prompt candidate in the population.
    """
    __tablename__ = "prompt_candidates"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(Integer, ForeignKey("generations.id"), nullable=False, index=True)
    
    # Prompt content
    prompt_text = Column(Text, nullable=False)
    prompt_hash = Column(String(64), index=True)  # For deduplication
    
    # Lineage
    parent_ids = Column(JSON, nullable=True)  # List of parent prompt IDs
    mutation_strategy = Column(String(100), nullable=True)
    mutation_description = Column(Text, nullable=True)
    
    # Evaluation scores
    fitness_score = Column(Float, nullable=True, index=True)
    accuracy_score = Column(Float, nullable=True)
    consistency_score = Column(Float, nullable=True)
    efficiency_score = Column(Float, nullable=True)
    cost_penalty = Column(Float, nullable=True)
    
    # Evaluation details
    evaluated_on_tier3 = Column(Boolean, default=True)
    evaluated_on_tier1 = Column(Boolean, default=False)
    tier1_score = Column(Float, nullable=True)
    tier3_score = Column(Float, nullable=True)
    
    # Performance
    avg_latency_ms = Column(Float, nullable=True)
    avg_tokens_per_output = Column(Float, nullable=True)
    
    # Selection status
    is_elite = Column(Boolean, default=False)
    times_selected = Column(Integer, default=0)
    
    # Linguistic feedback from Judge
    judge_feedback = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    evaluated_at = Column(DateTime, nullable=True)
    
    # Relationships
    generation = relationship("Generation", back_populates="prompt_candidates")
    evaluation_results = relationship("EvaluationResult", back_populates="prompt_candidate", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="source_prompt", foreign_keys=[Artifact.source_prompt_id])
    
    __table_args__ = (
        Index('idx_candidate_gen_fitness', 'generation_id', 'fitness_score'),
    )


class EvaluationResult(Base):
    """
    Individual test case evaluation result.
    """
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_candidate_id = Column(Integer, ForeignKey("prompt_candidates.id"), nullable=False, index=True)
    
    # Test case info
    test_case_id = Column(String(100), nullable=True)
    test_input = Column(Text, nullable=False)
    expected_output = Column(Text, nullable=True)
    actual_output = Column(Text, nullable=False)
    
    # Scores
    passed = Column(Boolean, nullable=False)
    score = Column(Float, nullable=False)
    
    # Details
    deterministic_checks = Column(JSON, nullable=True)  # Which checks passed/failed
    semantic_feedback = Column(Text, nullable=True)
    
    # Performance
    latency_ms = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    
    # Timestamps
    evaluated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prompt_candidate = relationship("PromptCandidate", back_populates="evaluation_results")


class Dataset(Base):
    """
    Golden and synthetic datasets.
    """
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identification
    name = Column(String(255), nullable=False)
    domain = Column(String(100), nullable=False, index=True)
    task_type = Column(String(100), nullable=True)
    
    # Type and source
    dataset_type = Column(String(50), nullable=False)  # golden, synthetic, validation
    source = Column(String(100), nullable=True)  # human, llm_generated, hitl_approved
    
    # Content
    test_cases = Column(JSON, nullable=False)  # List of test case objects
    rubric = Column(JSON, nullable=True)  # Evaluation rubric
    
    # Metadata
    size = Column(Integer, nullable=False)
    difficulty_distribution = Column(JSON, nullable=True)
    
    # HITL approval
    hitl_reviewed = Column(Boolean, default=False)
    approval_rate = Column(Float, nullable=True)
    reviewed_by = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_dataset_domain_type', 'domain', 'dataset_type'),
    )


class MonitoringEvent(Base):
    """
    Production monitoring events and metrics.
    """
    __tablename__ = "monitoring_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Event info
    event_type = Column(String(100), nullable=False, index=True)  # drift, failure, alert
    severity = Column(String(50), nullable=False)  # info, warning, error, critical
    
    # Context
    run_id = Column(Integer, ForeignKey("evolution_runs.id"), nullable=True)
    prompt_candidate_id = Column(Integer, ForeignKey("prompt_candidates.id"), nullable=True)
    domain = Column(String(100), nullable=True)
    
    # Event data
    message = Column(Text, nullable=False)
    metrics = Column(JSON, nullable=True)
    details = Column(JSON, nullable=True)
    
    # Resolution
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Timestamps
    occurred_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_event_type_time', 'event_type', 'occurred_at'),
        Index('idx_event_severity_resolved', 'severity', 'resolved'),
    )
