"""
PromptEvolve: Self-Improving Prompt Engineering Agent
Core implementation of the R-Zero co-evolutionary framework for automated prompt optimization.

This module implements a closed-loop system that continuously tests, evaluates, reflects upon,
and iteratively improves prompts using evolutionary principles.
"""

import json
import logging
import os
import time
import hashlib
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

import yaml
from pydantic import BaseModel, Field, validator, root_validator
from openai import OpenAI
import google.generativeai as genai

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Ensure directories exist
for directory in [CONFIG_DIR, DATA_DIR, LOGS_DIR, RESULTS_DIR, PROMPTS_DIR]:
    directory.mkdir(exist_ok=True)


# Helper function for unified LLM calls
def call_llm(
    client: Any,
    provider: str,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """
    Unified function to call either OpenAI or Gemini API.
    
    Args:
        client: OpenAI client or genai module
        provider: "openai" or "gemini"
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        
    Returns:
        Generated text response
    """
    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens  # Changed from max_tokens for gpt-5-mini compatibility
        )
        return response.choices[0].message.content
    elif provider == "gemini":
        # Convert messages to Gemini format
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted_parts.append(f"Instructions: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(formatted_parts)
        
        # Create model and generate
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Configure logging
def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up logging configuration from config file.
    
    Args:
        config: Configuration dictionary containing logging settings
        
    Returns:
        Configured logger instance
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', str(LOGS_DIR / 'prompt_evolution.log'))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if log_config.get('console', True) else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES (PYDANTIC MODELS)
# ============================================================================

class TestScenario(BaseModel):
    """
    Defines a single test case for prompt evaluation.
    
    Attributes:
        input_message: The user input or query to test
        existing_memories: Context or previous interactions (if applicable)
        desired_output: The expected/ideal response
        bad_output: A counter-example showing what NOT to produce
        metadata: Additional domain-specific metadata
    """
    input_message: str = Field(..., min_length=1, description="User input or query")
    existing_memories: str = Field(default="", description="Contextual memory or history")
    desired_output: str = Field(..., min_length=1, description="Expected response")
    bad_output: str = Field(default="", description="Counter-example of incorrect output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Domain-specific metadata")
    
    @validator('input_message', 'desired_output')
    def validate_non_empty(cls, v):
        """Ensure critical fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "input_message": "What is the statute of limitations for breach of contract in California?",
                "existing_memories": "User is a paralegal working on civil litigation cases.",
                "desired_output": "In California, the statute of limitations for breach of written contract is 4 years (CCP § 337)...",
                "bad_output": "The statute of limitations is 2 years.",
                "metadata": {"domain": "legal", "difficulty": "medium"}
            }
        }


class PromptResult(BaseModel):
    """
    Tracks the performance metrics for a single prompt variant.
    
    Attributes:
        prompt: The prompt text that was tested
        response: Example or aggregated response from the LLM
        success_rate: Percentage of test cases that passed (0.0 to 1.0)
        quality_score: Composite quality metric (0.0 to 1.0)
        consistency_score: How consistent outputs are across test runs (0.0 to 1.0)
        efficiency_score: Based on execution time and prompt length (0.0 to 1.0)
        execution_time: Average execution time in seconds
        domain_metrics: Domain-specific evaluation metrics
        test_count: Number of test scenarios evaluated
        timestamp: When this result was generated
    """
    prompt: str = Field(..., min_length=10, description="The prompt text")
    response: str = Field(default="", description="Sample or aggregated response")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0-1)")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Composite quality score (0-1)")
    consistency_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Consistency score (0-1)")
    efficiency_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Efficiency score (0-1)")
    execution_time: float = Field(..., gt=0.0, description="Avg execution time in seconds")
    domain_metrics: Dict[str, Any] = Field(default_factory=dict, description="Domain-specific metrics")
    test_count: int = Field(..., gt=0, description="Number of test scenarios")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @validator('prompt')
    def validate_prompt_length(cls, v):
        """Ensure prompt is not too short."""
        if len(v.strip()) < 10:
            raise ValueError("Prompt must be at least 10 characters")
        return v
    
    def get_composite_score(self) -> float:
        """
        Calculate a weighted composite score combining all metrics.
        
        Returns:
            Composite score (0.0 to 1.0)
        """
        return (
            self.success_rate * 0.4 +
            self.quality_score * 0.3 +
            self.consistency_score * 0.2 +
            self.efficiency_score * 0.1
        )


class EvolutionHistory(BaseModel):
    """
    Logs the outcome of a single evolution loop iteration.
    
    Attributes:
        iteration: Iteration number
        prompts_tested: Number of prompt variants tested
        max_success_rate: Best success rate achieved
        avg_success_rate: Average success rate across all prompts
        max_quality_score: Best quality score achieved
        avg_quality_score: Average quality score
        best_prompt_preview: Preview of the best-performing prompt
        learnings: Key insights from this iteration
        timestamp: When this iteration completed
    """
    iteration: int = Field(..., ge=0, description="Iteration number")
    prompts_tested: int = Field(..., gt=0, description="Number of prompts tested")
    max_success_rate: float = Field(..., ge=0.0, le=1.0)
    avg_success_rate: float = Field(..., ge=0.0, le=1.0)
    max_quality_score: float = Field(..., ge=0.0, le=1.0)
    avg_quality_score: float = Field(..., ge=0.0, le=1.0)
    best_prompt_preview: str = Field(..., max_length=200, description="Preview of best prompt")
    learnings: List[str] = Field(default_factory=list, description="Key insights")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class PromptState(BaseModel):
    """
    Tracks the current state of a prompt in the evolution process.
    
    Attributes:
        current_prompt: The current prompt text
        generation: Which generation/iteration this prompt belongs to
        changelog: History of modifications made to this prompt
        results: Performance results for this prompt
        should_continue: Whether to continue iterating
        termination_reason: Why evolution stopped (if applicable)
    """
    current_prompt: str = Field(..., min_length=10)
    generation: int = Field(default=0, ge=0)
    changelog: List[str] = Field(default_factory=list, description="Modification history")
    results: Optional[PromptResult] = None
    should_continue: bool = Field(default=True)
    termination_reason: Optional[str] = None
    
    def add_change(self, change_description: str):
        """Add a change to the changelog with timestamp."""
        timestamp = datetime.now().isoformat()
        self.changelog.append(f"[{timestamp}] {change_description}")


class DomainConfig(BaseModel):
    """
    Domain-specific configuration for prompt optimization.
    
    Attributes:
        domain_name: Name of the domain (e.g., "legal", "ontology")
        base_prompt_template: Starting prompt template for this domain
        evaluation_criteria: Weights for domain-specific metrics
        thresholds: Performance thresholds specific to this domain
        test_scenario_count: Number of test scenarios for this domain
    """
    domain_name: str = Field(..., min_length=1)
    base_prompt_template: str = Field(..., min_length=10)
    evaluation_criteria: Dict[str, float] = Field(..., description="Metric weights (sum to 1.0)")
    thresholds: Dict[str, float] = Field(default_factory=dict)
    test_scenario_count: int = Field(default=20, gt=0)
    
    @validator('evaluation_criteria')
    def validate_weights_sum(cls, v):
        """Ensure evaluation criteria weights sum to approximately 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Evaluation criteria weights must sum to 1.0, got {total}")
        return v


class SelfDebateOutput(BaseModel):
    """
    Output from the self-debate process.
    
    Attributes:
        proposed_prompt: The proposed new prompt variant
        skeptic_concerns: Concerns raised by the skeptic
        final_prompt: The final prompt after debate
        confidence_score: Confidence in the final prompt (0.0 to 1.0)
        reasoning: Explanation of the decision
    """
    proposed_prompt: str
    skeptic_concerns: List[str] = Field(default_factory=list)
    final_prompt: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


# ============================================================================
# PROMPT CHALLENGER (GENERATION)
# ============================================================================

class PromptChallenger:
    """
    Generates variations and enhancements of base prompts using various techniques.
    
    This component applies different prompt engineering strategies and uses a
    self-debate mechanism (Proposer/Skeptic/Judge) to refine prompt variants.
    """
    
    def __init__(self, client: Any, config: Dict[str, Any], logger: logging.Logger, provider: str = "openai"):
        """
        Initialize the Prompt Challenger.
        
        Args:
            client: API client instance (OpenAI or genai)
            config: Configuration dictionary
            logger: Logger instance
            provider: API provider ("openai" or "gemini")
        """
        self.client = client
        self.config = config
        self.logger = logger
        self.provider = provider
        
        # Get model name based on provider
        model_config = config['models']['prompt_writer']
        self.model = model_config.get(provider, model_config.get('name', 'gpt-4-turbo-preview'))
        self.temperature = model_config.get('temperature', 0.7)
        self.max_tokens = model_config.get('max_tokens', 2000)
        
        self.self_debate_enabled = config.get('self_debate', {}).get('enabled', True)
        self.prompt_techniques = config.get('prompt_techniques', {})
    
    def generate_variations(
        self,
        base_prompt: str,
        learnings: List[str],
        count: int = 5
    ) -> List[str]:
        """
        Generate multiple prompt variations using different techniques.
        
        Args:
            base_prompt: The starting prompt to vary
            learnings: Feedback from previous iterations
            count: Number of variations to generate
            
        Returns:
            List of prompt variations
        """
        self.logger.info(f"{__name__}:{self.generate_variations.__code__.co_firstlineno} - "
                        f"Generating {count} prompt variations")
        
        variations = []
        
        try:
            # Generate base variation with learnings incorporated
            if learnings:
                variations.append(self._incorporate_learnings(base_prompt, learnings))
            
            # Apply different techniques
            techniques = [
                self._add_chain_of_thought,
                self._add_few_shot_examples,
                self._add_role_specification,
                self._add_step_by_step_instructions,
            ]
            
            for technique in techniques[:count]:
                try:
                    variant = technique(base_prompt, learnings)
                    
                    # Apply self-debate if enabled
                    if self.self_debate_enabled:
                        debate_result = self._self_debate(variant, learnings)
                        variations.append(debate_result.final_prompt)
                    else:
                        variations.append(variant)
                        
                except Exception as e:
                    self.logger.error(f"{__name__}:{self.generate_variations.__code__.co_firstlineno} - "
                                    f"Error applying technique: {e}")
                    continue
            
            # Ensure we have at least one variation
            if not variations:
                variations.append(base_prompt)
            
            self.logger.info(f"{__name__}:{self.generate_variations.__code__.co_firstlineno} - "
                           f"Successfully generated {len(variations)} variations")
            
            return variations[:count]
            
        except Exception as e:
            self.logger.error(f"{__name__}:{self.generate_variations.__code__.co_firstlineno} - "
                            f"Critical error in variation generation: {e}")
            return [base_prompt]
    
    def _incorporate_learnings(self, base_prompt: str, learnings: List[str]) -> str:
        """Incorporate learnings from previous iterations into the prompt."""
        try:
            learning_summary = "\n".join([f"- {learning}" for learning in learnings[-5:]])
            
            system_message = """You are an expert prompt engineer. Improve the given prompt 
            by incorporating the learnings from previous iterations. Maintain the core intent 
            but address the identified issues."""
            
            user_message = f"""Base Prompt:
{base_prompt}

Learnings from previous iterations:
{learning_summary}

Generate an improved version of the prompt that addresses these learnings."""
            
            response_text = call_llm(
                self.client,
                self.provider,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response_text.strip()
            
        except Exception as e:
            self.logger.error(f"{__name__}:{self._incorporate_learnings.__code__.co_firstlineno} - "
                            f"Error incorporating learnings: {e}")
            return base_prompt
    
    def _add_chain_of_thought(self, base_prompt: str, learnings: List[str]) -> str:
        """Add chain-of-thought reasoning instructions to the prompt."""
        if not self.prompt_techniques.get('chain_of_thought', {}).get('enabled', True):
            return base_prompt
        
        cot_template = self.prompt_techniques.get('chain_of_thought', {}).get(
            'template',
            "Let's approach this step by step:\n1. First, analyze the input\n2. Then, identify key requirements\n3. Finally, formulate a comprehensive response"
        )
        
        return f"{base_prompt}\n\n{cot_template}"
    
    def _add_few_shot_examples(self, base_prompt: str, learnings: List[str]) -> str:
        """Add few-shot examples to the prompt."""
        if not self.prompt_techniques.get('few_shot', {}).get('enabled', True):
            return base_prompt
        
        # In a real implementation, you would retrieve relevant examples
        # For now, we'll use the LLM to generate appropriate examples
        try:
            system_message = """Generate 2-3 high-quality few-shot examples for the given prompt.
            Include both positive examples and counter-examples (what NOT to do) if requested."""
            
            user_message = f"""Prompt: {base_prompt}

Generate few-shot examples that demonstrate the desired behavior."""
            
            examples = call_llm(
                self.client,
                self.provider,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=1000
            )
            
            return f"{base_prompt}\n\nExamples:\n{examples.strip()}"
            
        except Exception as e:
            self.logger.error(f"{__name__}:{self._add_few_shot_examples.__code__.co_firstlineno} - "
                            f"Error adding few-shot examples: {e}")
            return base_prompt
    
    def _add_role_specification(self, base_prompt: str, learnings: List[str]) -> str:
        """Add a specific role specification to the prompt."""
        if not self.prompt_techniques.get('role_specification', {}).get('enabled', True):
            return base_prompt
        
        roles = self.prompt_techniques.get('role_specification', {}).get('roles', [])
        if roles:
            role = roles[0]  # Use the first role for now
            return f"You are a {role}. {base_prompt}"
        
        return base_prompt
    
    def _add_step_by_step_instructions(self, base_prompt: str, learnings: List[str]) -> str:
        """Restructure the prompt with step-by-step instructions."""
        if not self.prompt_techniques.get('step_by_step', {}).get('enabled', True):
            return base_prompt
        
        try:
            system_message = """Restructure the given prompt as clear, step-by-step instructions.
            Use a numbered list format."""
            
            user_message = f"""Original prompt: {base_prompt}

Rewrite this as step-by-step instructions."""
            
            response_text = call_llm(
                self.client,
                self.provider,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=1000
            )
            
            return response_text.strip()
            
        except Exception as e:
            self.logger.error(f"{__name__}:{self._add_step_by_step_instructions.__code__.co_firstlineno} - "
                            f"Error adding step-by-step instructions: {e}")
            return base_prompt
    
    def _self_debate(self, proposed_prompt: str, learnings: List[str]) -> SelfDebateOutput:
        """
        Run a self-debate process (Proposer/Skeptic/Judge) on a prompt variant.
        
        Args:
            proposed_prompt: The prompt variant to debate
            learnings: Previous learnings to inform the debate
            
        Returns:
            SelfDebateOutput containing the final prompt and reasoning
        """
        try:
            max_rounds = self.config.get('self_debate', {}).get('max_rounds', 2)
            rubric = self.config.get('self_debate', {}).get('rubric', {})
            
            # Proposer's case
            proposer_message = f"""As a Proposer, present the case for this prompt:

{proposed_prompt}

Explain why this prompt will work well and what improvements it offers."""
            
            # Skeptic's critique
            learning_context = "\n".join([f"- {l}" for l in learnings[-3:]]) if learnings else "No previous learnings."
            
            skeptic_message = f"""As a Skeptic, identify potential risks and failure modes for this prompt:

{proposed_prompt}

Previous learnings to consider:
{learning_context}

List specific concerns about correctness, simplicity, safety, and rollback capability."""
            
            # Get skeptic's concerns
            concerns = call_llm(
                self.client,
                self.provider,
                messages=[
                    {"role": "system", "content": "You are a critical skeptic evaluating prompt quality."},
                    {"role": "user", "content": skeptic_message}
                ],
                model=self.model,
                temperature=1.0,  # GPT-5 mini only supports temperature=1
                max_tokens=800
            )
            
            # Judge's final decision
            judge_message = f"""As a Judge, evaluate this prompt using the following rubric priorities:
- Correctness: {rubric.get('correctness', 10)}/10
- Simplicity: {rubric.get('simplicity', 7)}/10
- Safety: {rubric.get('safety', 9)}/10
- Rollback Plan: {rubric.get('rollback_plan', 8)}/10

Proposed Prompt:
{proposed_prompt}

Skeptic's Concerns:
{concerns}

Provide:
1. Final refined prompt (or keep original if it's good)
2. Confidence score (0.0 to 1.0)
3. Brief reasoning

Format your response as JSON:
{{
    "final_prompt": "...",
    "confidence_score": 0.85,
    "reasoning": "..."
}}"""
            
            judge_content = call_llm(
                self.client,
                self.provider,
                messages=[
                    {"role": "system", "content": "You are an impartial judge evaluating prompt quality."},
                    {"role": "user", "content": judge_message}
                ],
                model=self.model,
                temperature=1.0,  # GPT-5 mini only supports temperature=1
                max_tokens=1000
            )
            
            # Parse judge's response
            judge_content = judge_content.strip()
            
            # Extract JSON from response
            if "```json" in judge_content:
                json_start = judge_content.find("```json") + 7
                json_end = judge_content.find("```", json_start)
                judge_content = judge_content[json_start:json_end].strip()
            elif "```" in judge_content:
                json_start = judge_content.find("```") + 3
                json_end = judge_content.find("```", json_start)
                judge_content = judge_content[json_start:json_end].strip()
            
            judge_decision = json.loads(judge_content)
            
            return SelfDebateOutput(
                proposed_prompt=proposed_prompt,
                skeptic_concerns=[concerns],
                final_prompt=judge_decision.get('final_prompt', proposed_prompt),
                confidence_score=float(judge_decision.get('confidence_score', 0.7)),
                reasoning=judge_decision.get('reasoning', 'No reasoning provided')
            )
            
        except Exception as e:
            self.logger.error(f"{__name__}:{self._self_debate.__code__.co_firstlineno} - "
                            f"Error in self-debate: {e}")
            # Fallback to the proposed prompt
            return SelfDebateOutput(
                proposed_prompt=proposed_prompt,
                skeptic_concerns=[],
                final_prompt=proposed_prompt,
                confidence_score=0.5,
                reasoning="Self-debate failed, using proposed prompt"
            )


# ============================================================================
# PROMPT SOLVER (EXECUTION)
# ============================================================================

class PromptSolver:
    """
    Executes prompt variants against test scenarios using the production model.
    
    This component runs the actual LLM calls and aggregates results across
    multiple test cases.
    """
    
    def __init__(self, client: Any, config: Dict[str, Any], logger: logging.Logger, provider: str = "openai"):
        """
        Initialize the Prompt Solver.
        
        Args:
            client: API client instance (OpenAI or genai)
            config: Configuration dictionary
            logger: Logger instance
            provider: API provider ("openai" or "gemini")
        """
        self.client = client
        self.config = config
        self.logger = logger
        self.provider = provider
        
        # Get model name based on provider
        model_config = config['models']['prompt_solver']
        self.model = model_config.get(provider, model_config.get('name', 'gpt-3.5-turbo'))
        self.temperature = model_config.get('temperature', 0.3)
        self.max_tokens = model_config.get('max_tokens', 1500)
        
        self.timeout = config.get('safety', {}).get('llm_timeout', 30)
    
    def execute_prompt(
        self,
        prompt: str,
        test_scenarios: List[TestScenario]
    ) -> PromptResult:
        """
        Execute a prompt against multiple test scenarios and aggregate results.
        
        Args:
            prompt: The prompt to test
            test_scenarios: List of test scenarios to evaluate against
            
        Returns:
            PromptResult containing aggregated performance metrics
        """
        self.logger.info(f"{__name__}:{self.execute_prompt.__code__.co_firstlineno} - "
                        f"Executing prompt against {len(test_scenarios)} scenarios")
        
        if not test_scenarios:
            raise ValueError("test_scenarios cannot be empty")
        
        results = []
        execution_times = []
        
        for idx, scenario in enumerate(test_scenarios):
            try:
                start_time = time.time()
                
                # Construct the full message
                user_message = scenario.input_message
                if scenario.existing_memories:
                    user_message = f"Context: {scenario.existing_memories}\n\nQuery: {user_message}"
                
                # Execute the LLM call
                response_text = call_llm(
                    self.client,
                    self.provider,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                # Store the result for this scenario
                results.append({
                    'scenario_idx': idx,
                    'response': response_text.strip(),
                    'execution_time': execution_time,
                    'desired_output': scenario.desired_output,
                    'bad_output': scenario.bad_output
                })
                
            except Exception as e:
                self.logger.error(f"{__name__}:{self.execute_prompt.__code__.co_firstlineno} - "
                                f"Error executing scenario {idx}: {e}")
                # Record failure
                results.append({
                    'scenario_idx': idx,
                    'response': "",
                    'execution_time': 0.0,
                    'error': str(e)
                })
        
        # Aggregate results (will be refined by PromptVerifier)
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0
        
        # For now, return a preliminary result
        # The PromptVerifier will calculate the actual scores
        return PromptResult(
            prompt=prompt,
            response=results[0]['response'] if results else "",
            success_rate=0.0,  # Will be calculated by verifier
            quality_score=0.0,  # Will be calculated by verifier
            execution_time=avg_execution_time,
            test_count=len(test_scenarios),
            domain_metrics={'raw_results': results}
        )


# ============================================================================
# PROMPT VERIFIER (EVALUATION)
# ============================================================================

class PromptVerifier:
    """
    Evaluates prompt effectiveness and generates actionable feedback.
    
    This component calculates comprehensive metrics and produces "learnings"
    that explain why prompts succeeded or failed.
    """
    
    def __init__(self, client: Any, config: Dict[str, Any], logger: logging.Logger, provider: str = "openai"):
        """
        Initialize the Prompt Verifier.
        
        Args:
            client: API client instance (OpenAI or genai)
            config: Configuration dictionary
            logger: Logger instance
            provider: API provider ("openai" or "gemini")
        """
        self.client = client
        self.config = config
        self.logger = logger
        self.provider = provider
        
        # Get model name based on provider
        model_config = config['models']['prompt_verifier']
        self.model = model_config.get(provider, model_config.get('name', 'gpt-4-turbo-preview'))
        self.temperature = model_config.get('temperature', 0.2)
        self.max_tokens = model_config.get('max_tokens', 1000)
    
    def evaluate_results(
        self,
        prompt_result: PromptResult,
        test_scenarios: List[TestScenario],
        domain_config: Optional[DomainConfig] = None
    ) -> Tuple[PromptResult, List[str]]:
        """
        Evaluate prompt results and generate comprehensive metrics and learnings.
        
        Args:
            prompt_result: Preliminary results from PromptSolver
            test_scenarios: The test scenarios that were used
            domain_config: Optional domain-specific configuration
            
        Returns:
            Tuple of (updated PromptResult with scores, list of learnings)
            
        Pre-conditions:
            - test_scenarios is not empty
            - prompt_result.test_count > 0
            
        Post-conditions:
            - returned PromptResult has success_rate between 0.0 and 1.0
            - returned PromptResult has quality_score between 0.0 and 1.0
        """
        self.logger.info(f"{__name__}:{self.evaluate_results.__code__.co_firstlineno} - "
                        "Evaluating prompt results")
        
        raw_results = prompt_result.domain_metrics.get('raw_results', [])
        
        if not raw_results:
            return prompt_result, ["No results to evaluate"]
        
        # Calculate success rate using LLM-based evaluation
        success_count = 0
        quality_scores = []
        learnings = []
        
        for result in raw_results:
            if 'error' in result:
                quality_scores.append(0.0)
                continue
            
            # Use LLM to judge if response matches desired output
            try:
                judgment = self._judge_response(
                    response=result['response'],
                    desired=result['desired_output'],
                    bad_example=result.get('bad_output', ''),
                    domain_config=domain_config
                )
                
                if judgment['is_success']:
                    success_count += 1
                
                quality_scores.append(judgment['quality_score'])
                
                if judgment.get('feedback'):
                    learnings.append(judgment['feedback'])
                    
            except Exception as e:
                self.logger.error(f"{__name__}:{self.evaluate_results.__code__.co_firstlineno} - "
                                f"Error judging response: {e}")
                quality_scores.append(0.0)
        
        # Calculate metrics
        success_rate = success_count / len(raw_results)
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0
        consistency_score = 1.0 - (statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0)
        
        # Calculate efficiency score (inverse of execution time and prompt length)
        time_score = max(0.0, 1.0 - (prompt_result.execution_time / 10.0))  # Normalize to 10 seconds
        length_score = max(0.0, 1.0 - (len(prompt_result.prompt) / 4000.0))  # Normalize to 4000 chars
        efficiency_score = (time_score + length_score) / 2.0
        
        # Update prompt result
        updated_result = PromptResult(
            prompt=prompt_result.prompt,
            response=prompt_result.response,
            success_rate=success_rate,
            quality_score=avg_quality_score,
            consistency_score=max(0.0, min(1.0, consistency_score)),
            efficiency_score=efficiency_score,
            execution_time=prompt_result.execution_time,
            domain_metrics=prompt_result.domain_metrics,
            test_count=prompt_result.test_count
        )
        
        # Generate summary learnings
        summary_learnings = self._generate_learnings_summary(
            success_rate=success_rate,
            quality_score=avg_quality_score,
            individual_learnings=learnings
        )
        
        return updated_result, summary_learnings
    
    def _judge_response(
        self,
        response: str,
        desired: str,
        bad_example: str,
        domain_config: Optional[DomainConfig]
    ) -> Dict[str, Any]:
        """
        Use LLM to judge if a response meets quality criteria.
        
        Args:
            response: The actual LLM response
            desired: The desired output
            bad_example: Example of what NOT to produce
            domain_config: Optional domain-specific configuration
            
        Returns:
            Dictionary with is_success, quality_score, and feedback
        """
        try:
            evaluation_criteria = ""
            if domain_config:
                criteria_list = [f"- {k}: {v*100}%" for k, v in domain_config.evaluation_criteria.items()]
                evaluation_criteria = "\n".join(criteria_list)
            
            judge_message = f"""Evaluate the following LLM response against the desired output.

Desired Output:
{desired}

Actual Response:
{response}

{f'Bad Example (what NOT to produce):{bad_example}' if bad_example else ''}

{f'Evaluation Criteria:{evaluation_criteria}' if evaluation_criteria else ''}

Provide your evaluation in JSON format:
{{
    "is_success": true/false,
    "quality_score": 0.0-1.0,
    "feedback": "Brief explanation of strengths/weaknesses"
}}"""
            
            content = call_llm(
                self.client,
                self.provider,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator judging LLM outputs."},
                    {"role": "user", "content": judge_message}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=500
            )
            
            content = content.strip()
            
            # Extract JSON
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            judgment = json.loads(content)
            return judgment
            
        except Exception as e:
            self.logger.error(f"{__name__}:{self._judge_response.__code__.co_firstlineno} - "
                            f"Error in judgment: {e}")
            return {
                'is_success': False,
                'quality_score': 0.0,
                'feedback': f"Judgment failed: {str(e)}"
            }
    
    def _generate_learnings_summary(
        self,
        success_rate: float,
        quality_score: float,
        individual_learnings: List[str]
    ) -> List[str]:
        """
        Generate a summary of learnings from the evaluation.
        
        Args:
            success_rate: Overall success rate
            quality_score: Overall quality score
            individual_learnings: Feedback from individual test cases
            
        Returns:
            List of summary learnings
        """
        summaries = []
        
        if success_rate < 0.5:
            summaries.append(f"Low success rate ({success_rate:.1%}): The prompt is failing on most test cases.")
        elif success_rate < 0.8:
            summaries.append(f"Moderate success rate ({success_rate:.1%}): The prompt works in some cases but needs improvement.")
        else:
            summaries.append(f"Good success rate ({success_rate:.1%}): The prompt is generally effective.")
        
        if quality_score < 0.6:
            summaries.append(f"Low quality score ({quality_score:.2f}): Responses lack accuracy or completeness.")
        
        # Add top individual learnings
        if individual_learnings:
            summaries.extend(individual_learnings[:3])
        
        return summaries


# ============================================================================
# EVOLUTION ENGINE (SELECTION/ITERATION)
# ============================================================================

class EvolutionEngine:
    """
    Implements genetic algorithm-style selection and combination of prompts.
    
    This component manages the iterative improvement process, including
    selection of top performers and combination of successful elements.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Evolution Engine.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.evolution_config = config.get('evolution', {})
        self.thresholds = config.get('thresholds', {})
    
    def select_top_performers(
        self,
        results: List[PromptResult],
        selection_percentage: Optional[float] = None
    ) -> List[PromptResult]:
        """
        Select the top-performing prompts based on composite scores.
        
        Args:
            results: List of prompt results to select from
            selection_percentage: Percentage of top results to select (0.0 to 1.0)
            
        Returns:
            List of selected top-performing prompts
            
        Pre-conditions:
            - results is not empty
            - selection_percentage is None or between 0.0 and 1.0
            
        Post-conditions:
            - returned list is not empty
            - all returned results have quality_score >= 0.0
        """
        self.logger.info(f"{__name__}:{self.select_top_performers.__code__.co_firstlineno} - "
                        f"Selecting top performers from {len(results)} results")
        
        if not results:
            raise ValueError("results cannot be empty")
        
        if selection_percentage is None:
            selection_percentage = self.evolution_config.get('selection_top_percentage', 0.30)
        
        # Sort by composite score
        sorted_results = sorted(
            results,
            key=lambda r: r.get_composite_score(),
            reverse=True
        )
        
        # Calculate selection count
        selection_count = max(
            self.evolution_config.get('min_selection_count', 2),
            int(len(sorted_results) * selection_percentage)
        )
        
        selected = sorted_results[:selection_count]
        
        self.logger.info(f"{__name__}:{self.select_top_performers.__code__.co_firstlineno} - "
                        f"Selected {len(selected)} top performers")
        
        return selected
    
    def combine_prompts(
        self,
        prompt1: str,
        prompt2: str,
        learnings: List[str]
    ) -> str:
        """
        Combine successful elements from two high-performing prompts.
        
        Args:
            prompt1: First prompt to combine
            prompt2: Second prompt to combine
            learnings: Context from previous iterations
            
        Returns:
            Combined prompt incorporating elements from both
        """
        self.logger.info(f"{__name__}:{self.combine_prompts.__code__.co_firstlineno} - "
                        "Combining two prompts")
        
        # Simple combination strategy: identify unique elements and merge
        # In a more sophisticated implementation, you could use an LLM to do this
        
        # For now, we'll use a simple concatenation with deduplication
        elements1 = set(prompt1.split('\n'))
        elements2 = set(prompt2.split('\n'))
        
        # Keep unique elements from both
        combined_elements = elements1.union(elements2)
        
        # Reconstruct prompt
        combined = '\n'.join(sorted(combined_elements, key=len, reverse=True))
        
        return combined
    
    def should_continue(
        self,
        iteration: int,
        best_result: PromptResult,
        history: List[EvolutionHistory]
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if the evolution loop should continue.
        
        Args:
            iteration: Current iteration number
            best_result: Best result from current iteration
            history: Evolution history from previous iterations
            
        Returns:
            Tuple of (should_continue, termination_reason)
        """
        max_iterations = self.evolution_config.get('max_iterations', 10)
        min_iterations = self.evolution_config.get('min_iterations', 3)
        
        # Check max iterations
        if iteration >= max_iterations:
            return False, f"Reached maximum iterations ({max_iterations})"
        
        # Don't stop before min iterations
        if iteration < min_iterations:
            return True, None
        
        # Check if thresholds are met
        success_threshold = self.thresholds.get('success_rate', 0.85)
        quality_threshold = self.thresholds.get('quality_score', 0.80)
        
        if (best_result.success_rate >= success_threshold and 
            best_result.quality_score >= quality_threshold):
            return False, f"Thresholds met (success: {best_result.success_rate:.1%}, quality: {best_result.quality_score:.2f})"
        
        # Check for improvement plateau
        if len(history) >= 3:
            recent_improvements = [
                history[i].max_quality_score - history[i-1].max_quality_score
                for i in range(-2, 0)
            ]
            
            min_improvement = self.thresholds.get('min_improvement_threshold', 0.02)
            if all(imp < min_improvement for imp in recent_improvements):
                return False, "Improvement has plateaued"
        
        return True, None


# ============================================================================
# MAIN PROMPT EVOLUTION ORCHESTRATOR
# ============================================================================

class PromptEvolution:
    """
    Main orchestrator for the R-Zero prompt evolution loop.
    
    This class coordinates all components to continuously improve prompts
    through testing, evaluation, and iteration.
    """
    
    def __init__(self, config_path: str = None, api_key: Optional[str] = None):
        """
        Initialize the PromptEvolution system.
        
        Args:
            config_path: Path to YAML configuration file (defaults to config/config.yaml)
            api_key: API key (or uses OPENAI_API_KEY/GEMINI_API_KEY env var)
        """
        # Default config path
        if config_path is None:
            config_path = str(CONFIG_DIR / "config.yaml")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up logging
        self.logger = setup_logging(self.config)
        self.logger.info(f"{__name__}:{self.__init__.__code__.co_firstlineno} - "
                        "Initializing PromptEvolution system")
        
        # Determine API provider
        self.provider = self.config.get('api_provider', 'openai').lower()
        
        # Initialize API client based on provider
        if self.provider == "openai":
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "gemini":
            api_key = api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key must be provided or set in GEMINI_API_KEY environment variable")
            genai.configure(api_key=api_key)
            self.client = genai
        else:
            raise ValueError(f"Unsupported API provider: {self.provider}. Use 'openai' or 'gemini'")
        
        self.logger.info(f"Using {self.provider} API provider")
        
        # Initialize components
        self.challenger = PromptChallenger(self.client, self.config, self.logger, self.provider)
        self.solver = PromptSolver(self.client, self.config, self.logger, self.provider)
        self.verifier = PromptVerifier(self.client, self.config, self.logger, self.provider)
        self.engine = EvolutionEngine(self.config, self.logger)
        
        # Create necessary directories
        self._create_directories()
        
        # Evolution state
        self.history: List[EvolutionHistory] = []
    
    def _create_directories(self):
        """Create necessary directories for storing data."""
        paths = self.config.get('paths', {})
        for path_key, path_value in paths.items():
            os.makedirs(path_value, exist_ok=True)
    
    def evolve_prompt(
        self,
        base_prompt: str,
        test_scenarios: List[TestScenario],
        domain_config: Optional[DomainConfig] = None
    ) -> PromptState:
        """
        Run the complete evolution loop to optimize a prompt.
        
        Args:
            base_prompt: Starting prompt to optimize
            test_scenarios: Test scenarios for evaluation
            domain_config: Optional domain-specific configuration
            
        Returns:
            Final PromptState with the optimized prompt
        """
        self.logger.info(f"{__name__}:{self.evolve_prompt.__code__.co_firstlineno} - "
                        "Starting prompt evolution loop")
        
        # Initialize state
        state = PromptState(current_prompt=base_prompt, generation=0)
        current_prompts = [base_prompt]
        all_learnings: List[str] = []
        
        try:
            for iteration in range(self.config['evolution']['max_iterations']):
                self.logger.info(f"{__name__}:{self.evolve_prompt.__code__.co_firstlineno} - "
                                f"Starting iteration {iteration + 1}")
                
                # STEP 1: Execute all prompt variants
                results = []
                for prompt in current_prompts:
                    try:
                        result = self.solver.execute_prompt(prompt, test_scenarios)
                        evaluated_result, learnings = self.verifier.evaluate_results(
                            result, test_scenarios, domain_config
                        )
                        results.append(evaluated_result)
                        all_learnings.extend(learnings)
                    except Exception as e:
                        self.logger.error(f"{__name__}:{self.evolve_prompt.__code__.co_firstlineno} - "
                                        f"Error processing prompt: {e}")
                        continue
                
                if not results:
                    self.logger.error(f"{__name__}:{self.evolve_prompt.__code__.co_firstlineno} - "
                                    "No valid results in this iteration")
                    break
                
                # STEP 2: Select top performers
                top_results = self.engine.select_top_performers(results)
                best_result = top_results[0]
                
                # STEP 3: Log history
                history_entry = EvolutionHistory(
                    iteration=iteration + 1,
                    prompts_tested=len(results),
                    max_success_rate=best_result.success_rate,
                    avg_success_rate=statistics.mean([r.success_rate for r in results]),
                    max_quality_score=best_result.quality_score,
                    avg_quality_score=statistics.mean([r.quality_score for r in results]),
                    best_prompt_preview=best_result.prompt[:200],
                    learnings=all_learnings[-5:]
                )
                self.history.append(history_entry)
                
                # STEP 4: Check termination
                should_continue, reason = self.engine.should_continue(
                    iteration + 1, best_result, self.history
                )
                
                if not should_continue:
                    self.logger.info(f"{__name__}:{self.evolve_prompt.__code__.co_firstlineno} - "
                                    f"Evolution complete: {reason}")
                    state.current_prompt = best_result.prompt
                    state.results = best_result
                    state.should_continue = False
                    state.termination_reason = reason
                    state.generation = iteration + 1
                    break
                
                # STEP 5: Generate new variations for next iteration
                state.current_prompt = best_result.prompt
                state.results = best_result
                state.generation = iteration + 1
                state.add_change(f"Iteration {iteration + 1}: Best score = {best_result.get_composite_score():.3f}")
                
                # Generate variations from top performers
                new_prompts = []
                for top_result in top_results[:2]:  # Use top 2
                    variations = self.challenger.generate_variations(
                        top_result.prompt,
                        all_learnings[-10:],  # Use recent learnings
                        count=2
                    )
                    new_prompts.extend(variations)
                
                # Add crossover if enabled
                if (self.config['evolution'].get('enable_crossover', True) and 
                    len(top_results) >= 2):
                    combined = self.engine.combine_prompts(
                        top_results[0].prompt,
                        top_results[1].prompt,
                        all_learnings[-5:]
                    )
                    new_prompts.append(combined)
                
                current_prompts = new_prompts[:self.config['evolution']['population_size']]
            
            # Save final state
            self._save_evolution_results(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"{__name__}:{self.evolve_prompt.__code__.co_firstlineno} - "
                            f"Critical error in evolution loop: {e}")
            state.should_continue = False
            state.termination_reason = f"Error: {str(e)}"
            return state
    
    def _save_evolution_results(self, state: PromptState):
        """Save evolution results to file."""
        try:
            results_dir = self.config['paths'].get('results_directory', str(RESULTS_DIR))
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(results_dir, f"evolution_result_{timestamp}.json")
            
            result_data = {
                'final_prompt': state.current_prompt,
                'generation': state.generation,
                'changelog': state.changelog,
                'results': state.results.dict() if state.results else None,
                'termination_reason': state.termination_reason,
                'history': [h.dict() for h in self.history]
            }
            
            with open(filename, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            self.logger.info(f"{__name__}:{self._save_evolution_results.__code__.co_firstlineno} - "
                           f"Saved results to {filename}")
            
        except Exception as e:
            self.logger.error(f"{__name__}:{self._save_evolution_results.__code__.co_firstlineno} - "
                            f"Error saving results: {e}")


# ============================================================================
# DOMAIN PROMPT OPTIMIZER
# ============================================================================

class DomainPromptOptimizer:
    """
    Manages domain-specific prompt optimization.
    
    This class handles configuration and optimization for different domains
    (Legal, Ontology, Admin) with domain-specific evaluation criteria.
    """
    
    def __init__(self, config_path: str = None, api_key: Optional[str] = None):
        """
        Initialize the Domain Prompt Optimizer.
        
        Args:
            config_path: Path to YAML configuration file (defaults to config/config.yaml)
            api_key: OpenAI API key
        """
        if config_path is None:
            config_path = str(CONFIG_DIR / "config.yaml")
        self.evolution_system = PromptEvolution(config_path, api_key)
        self.config = self.evolution_system.config
        self.logger = self.evolution_system.logger
        
        # Load domain configurations
        self.domain_configs = self._load_domain_configs()
    
    def _load_domain_configs(self) -> Dict[str, DomainConfig]:
        """Load domain-specific configurations from config file."""
        domains = {}
        
        for domain_name, domain_data in self.config.get('domains', {}).items():
            try:
                config = DomainConfig(
                    domain_name=domain_name,
                    base_prompt_template=domain_data['base_prompt_template'],
                    evaluation_criteria=domain_data['evaluation_criteria'],
                    thresholds=domain_data.get('thresholds', {}),
                    test_scenario_count=domain_data.get('test_scenario_count', 20)
                )
                domains[domain_name] = config
            except Exception as e:
                self.logger.error(f"{__name__}:{self._load_domain_configs.__code__.co_firstlineno} - "
                                f"Error loading config for domain {domain_name}: {e}")
        
        return domains
    
    def optimize_for_domain(
        self,
        domain_name: str,
        test_scenarios: List[TestScenario],
        custom_base_prompt: Optional[str] = None
    ) -> PromptState:
        """
        Optimize a prompt for a specific domain.
        
        Args:
            domain_name: Name of the domain (legal, ontology, admin)
            test_scenarios: Domain-specific test scenarios
            custom_base_prompt: Optional custom starting prompt (uses domain default if None)
            
        Returns:
            Final optimized PromptState
        """
        if domain_name not in self.domain_configs:
            raise ValueError(f"Unknown domain: {domain_name}. Available: {list(self.domain_configs.keys())}")
        
        domain_config = self.domain_configs[domain_name]
        
        base_prompt = custom_base_prompt or domain_config.base_prompt_template
        
        self.logger.info(f"{__name__}:{self.optimize_for_domain.__code__.co_firstlineno} - "
                        f"Optimizing prompt for domain: {domain_name}")
        
        return self.evolution_system.evolve_prompt(
            base_prompt=base_prompt,
            test_scenarios=test_scenarios,
            domain_config=domain_config
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of the PromptEvolution system."""
    
    # Initialize the system (config_path defaults to ../config/config.yaml from project root)
    system = PromptEvolution()
    
    # Create sample test scenarios
    test_scenarios = [
        TestScenario(
            input_message="What is the capital of France?",
            existing_memories="",
            desired_output="The capital of France is Paris.",
            bad_output="The capital is London.",
            metadata={"difficulty": "easy"}
        ),
        TestScenario(
            input_message="Explain quantum entanglement",
            existing_memories="User has basic physics knowledge",
            desired_output="Quantum entanglement is a phenomenon where particles become correlated...",
            bad_output="Entanglement is magic.",
            metadata={"difficulty": "hard"}
        )
    ]
    
    # Base prompt to optimize
    base_prompt = "You are a helpful assistant. Answer questions accurately and concisely."
    
    # Run evolution
    final_state = system.evolve_prompt(
        base_prompt=base_prompt,
        test_scenarios=test_scenarios
    )
    
    print(f"Evolution completed after {final_state.generation} generations")
    print(f"Termination reason: {final_state.termination_reason}")
    print(f"\nFinal prompt:\n{final_state.current_prompt}")
    
    if final_state.results:
        print(f"\nFinal metrics:")
        print(f"  Success rate: {final_state.results.success_rate:.1%}")
        print(f"  Quality score: {final_state.results.quality_score:.2f}")
        print(f"  Composite score: {final_state.results.get_composite_score():.2f}")


if __name__ == "__main__":
    main()
