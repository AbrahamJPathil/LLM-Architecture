"""
PromptChallenger: LLM-driven prompt mutation generator.
Implements multiple mutation strategies for genetic algorithm.
"""

import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.core.llm_client import create_llm_client
from src.core.prompt_verifier import BatchEvaluationResult
from src.utils.cost_tracker import count_tokens
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MutationStrategy(str, Enum):
    """Available mutation strategies."""
    REPHRASE = "rephrase"
    ADD_CONSTRAINTS = "add_constraints"
    SIMPLIFY = "simplify"
    ADD_EXAMPLES = "add_examples"
    STRUCTURAL = "structural"


@dataclass
class MutationResult:
    """Result of a prompt mutation."""
    
    new_prompt: str
    strategy: MutationStrategy
    description: str
    diff_summary: str
    
    # Estimated impact
    estimated_tokens: int
    expected_improvement: str  # What we expect to improve
    
    # Metadata
    parent_prompt: str
    linguistic_feedback: Optional[str] = None
    # The actual generator prompt used to instruct the LLM (useful to store as MUTATION DNA)
    generator_prompt: Optional[str] = None


class PromptChallenger:
    """
    Generates prompt mutations using LLM-driven strategies.
    Uses linguistic feedback to guide improvements.
    """
    
    def __init__(
        self,
        tier: str = "tier3",  # Use cheap model for mutations
        strategy_weights: Optional[Dict[MutationStrategy, float]] = None,
        domain: Optional[str] = None,
        task_description: Optional[str] = None,
        user_context: Optional[str] = None
    ):
        """
        Initialize PromptChallenger.
        
        Args:
            tier: LLM tier to use for generating mutations
            strategy_weights: Weights for selecting strategies
            domain: Domain for context-aware mutations
            task_description: Task description for context
            user_context: User context for domain-specific mutations
        """
        self.tier = tier
        self.llm_client = create_llm_client(tier)
        self.domain = domain
        self.task_description = task_description
        self.user_context = user_context
        
        # Strategy selection weights
        self.strategy_weights = strategy_weights or {
            MutationStrategy.REPHRASE: 0.25,
            MutationStrategy.ADD_CONSTRAINTS: 0.20,
            MutationStrategy.SIMPLIFY: 0.15,
            MutationStrategy.ADD_EXAMPLES: 0.20,
            MutationStrategy.STRUCTURAL: 0.20,
        }
        
        logger.info(
            "PromptChallenger initialized",
            tier=tier,
            model=self.llm_client.model
        )
    
    def select_strategy(self) -> MutationStrategy:
        """Select a mutation strategy based on weights."""
        strategies = list(self.strategy_weights.keys())
        weights = [self.strategy_weights[s] for s in strategies]
        return random.choices(strategies, weights=weights, k=1)[0]
    
    def mutate(
        self,
        prompt: str,
        strategy: Optional[MutationStrategy] = None,
        linguistic_feedback: Optional[str] = None,
        evaluation_result: Optional[BatchEvaluationResult] = None,
        # Warm-start inputs
        mutation_instruction: Optional[str] = None,  # explicit instruction from RAG (for MUTATION artifacts or applying a PATTERN)
        rag_artifacts: Optional[List[dict]] = None,  # a small list of retrieved artifacts to provide guidance
    ) -> MutationResult:
        """
        Generate a mutated version of the prompt.
        
        Args:
            prompt: Base prompt to mutate
            strategy: Specific strategy to use (or random if None)
            linguistic_feedback: Feedback from previous evaluations
            evaluation_result: Detailed evaluation results for context
            
        Returns:
            MutationResult with new prompt and metadata
        """
        if strategy is None:
            strategy = self.select_strategy()
        
        logger.info(
            "Generating mutation",
            strategy=strategy,
            has_feedback=linguistic_feedback is not None
        )
        
        # Create mutation prompt based on strategy
        mutation_prompt = self._create_mutation_prompt(
            prompt,
            strategy,
            linguistic_feedback,
            evaluation_result,
            mutation_instruction=mutation_instruction,
            rag_artifacts=rag_artifacts or []
        )
        
        # Generate mutation
        try:
            response = self.llm_client.complete(
                mutation_prompt,
                temperature=0.7,  # Balanced creativity
                max_tokens=500  # Short response for speed
            )
            
            # Parse structured output
            mutation_data = self._parse_mutation_output(response.content)
            
            if not mutation_data:
                logger.warning("Failed to parse mutation output, using fallback")
                mutation_data = {
                    "new_prompt": prompt,  # Fallback to original
                    "explanation": "Mutation failed",
                    "diff_summary": "No changes",
                    "expected_improvement": "None"
                }
            
            # Estimate tokens
            estimated_tokens = count_tokens(mutation_data["new_prompt"], self.llm_client.model)
            
            return MutationResult(
                new_prompt=mutation_data["new_prompt"],
                strategy=strategy,
                description=mutation_data.get("explanation", "Mutation applied"),
                diff_summary=mutation_data.get("diff_summary", "Prompt modified"),
                estimated_tokens=estimated_tokens,
                expected_improvement=mutation_data.get("expected_improvement", "General improvement"),
                parent_prompt=prompt,
                linguistic_feedback=linguistic_feedback,
                generator_prompt=mutation_prompt
            )
        
        except Exception as e:
            logger.error(f"Mutation generation failed: {e}")
            # Return unchanged prompt as fallback
            return MutationResult(
                new_prompt=prompt,
                strategy=strategy,
                description=f"Mutation failed: {e}",
                diff_summary="No changes (error)",
                estimated_tokens=count_tokens(prompt, self.llm_client.model),
                expected_improvement="None",
                parent_prompt=prompt,
                linguistic_feedback=linguistic_feedback,
                generator_prompt=mutation_prompt if 'mutation_prompt' in locals() else None
            )
    
    def _create_mutation_prompt(
        self,
        prompt: str,
        strategy: MutationStrategy,
        linguistic_feedback: Optional[str],
        evaluation_result: Optional[BatchEvaluationResult],
        *,
        mutation_instruction: Optional[str] = None,
        rag_artifacts: Optional[List[dict]] = None,
    ) -> str:
        """Create FAST mutation prompt (simplified for speed)."""
        
        # Ultra-simplified strategy instructions
        strategy_hints = {
            MutationStrategy.REPHRASE: "clearer wording",
            MutationStrategy.ADD_CONSTRAINTS: "add specific requirements",
            MutationStrategy.SIMPLIFY: "remove verbosity",
            MutationStrategy.ADD_EXAMPLES: "add 1-2 examples",
            MutationStrategy.STRUCTURAL: "use bullets/sections",
        }
        
        hint = strategy_hints.get(strategy, "improve")
        
        # Build context section
        context_parts = []
        if self.user_context:
            context_parts.append(f"Use Case: {self.user_context}")
        if self.task_description:
            context_parts.append(f"Task: {self.task_description}")
        if self.domain and self.domain != "general":
            context_parts.append(f"Domain: {self.domain}")
        
        context_section = "\n".join(context_parts) if context_parts else ""
        
        # Build feedback section
        feedback_section = ""
        if linguistic_feedback:
            feedback_section = f"\nEVALUATION FEEDBACK:\n{linguistic_feedback}\n"
        
        # RAG artifacts section (brief)
        rag_section = ""
        if rag_artifacts:
            top_lines = []
            for a in rag_artifacts[:3]:
                atype = a.get("artifact_type", "")
                name = a.get("name") or a.get("content") or ""
                desc = a.get("description") or ""
                top_lines.append(f"- {atype}: {name} — {desc}")
            if top_lines:
                rag_section = "\nRAG ARTIFACTS (use when helpful):\n" + "\n".join(top_lines) + "\n"

        # If explicit mutation instruction is provided, use it to drive the change
        if mutation_instruction:
            mutation_prompt = f"""You are an expert prompt engineer. Apply the following instruction to improve the prompt.

IMPORTANT CONTEXT:
{context_section}
{feedback_section}{rag_section}
MUTATION INSTRUCTION:
{mutation_instruction}

CURRENT PROMPT:
"{prompt}"

Return JSON:
{{"new_prompt": "improved version following the instruction", "explanation": "brief reason", "diff_summary": "what changed", "expected_improvement": "clarity/accuracy/etc"}}"""
            return mutation_prompt

        # Build mutation prompt with context and feedback (generic)
        if context_section or feedback_section or rag_section:
            mutation_prompt = f"""Improve this prompt ({hint}).

IMPORTANT CONTEXT:
{context_section}
{feedback_section}{rag_section}
CURRENT PROMPT:
"{prompt}"

Make sure the improved prompt addresses the weaknesses and is SPECIFIC to the context above.

Return JSON:
{{"new_prompt": "improved version tailored to context and addressing feedback", "explanation": "brief reason", "diff_summary": "what changed", "expected_improvement": "clarity/accuracy/etc"}}"""
        else:
            # Fallback if no context
            mutation_prompt = f"""Improve this prompt ({hint}):

"{prompt}"

Return JSON:
{{"new_prompt": "improved version", "explanation": "brief reason", "diff_summary": "what changed", "expected_improvement": "clarity/accuracy/etc"}}"""
        
        return mutation_prompt
    
    def _get_strategy_instructions(self, strategy: MutationStrategy) -> str:
        """Get detailed instructions for each strategy."""
        instructions = {
            MutationStrategy.REPHRASE: """
Rephrase the prompt using different wording while maintaining the same intent.
Focus on:
- Using clearer, more precise language
- Improving tone and professionalism
- Making instructions more explicit
- Enhancing readability
""",
            MutationStrategy.ADD_CONSTRAINTS: """
Add constraints or requirements to make the prompt stricter and more specific.
Focus on:
- Adding output format requirements
- Specifying length limits
- Adding quality criteria
- Including edge case handling
- Adding "do not" constraints
""",
            MutationStrategy.SIMPLIFY: """
Simplify the prompt by removing unnecessary complexity.
Focus on:
- Removing verbose language
- Combining redundant instructions
- Using simpler words
- Shortening sentences
- Keeping only essential requirements
""",
            MutationStrategy.ADD_EXAMPLES: """
Add examples (few-shot learning) to clarify expectations.
Focus on:
- Adding 1-3 concrete examples
- Showing input → output pairs
- Demonstrating edge cases
- Illustrating the desired format
""",
            MutationStrategy.STRUCTURAL: """
Modify the structure and format of the prompt.
Focus on:
- Using bullets or numbered lists
- Adding clear sections (e.g., "Task:", "Requirements:", "Output Format:")
- Using markdown formatting
- Reorganizing information flow
- Adding visual structure (headers, spacing)
"""
        }
        
        return instructions.get(strategy, "Improve the prompt.")
    
    def _parse_mutation_output(self, output: str) -> Optional[Dict[str, str]]:
        """Parse structured output from mutation LLM."""
        try:
            # Extract JSON from output
            if "```json" in output:
                start = output.find("```json") + 7
                end = output.find("```", start)
                json_str = output[start:end].strip()
            elif "```" in output:
                start = output.find("```") + 3
                end = output.find("```", start)
                json_str = output[start:end].strip()
            else:
                # Try to find JSON object
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = output.strip()
            
            data = json.loads(json_str)
            
            # Validate required fields
            if "new_prompt" not in data:
                logger.warning("Mutation output missing 'new_prompt' field")
                return None
            
            return data
        
        except Exception as e:
            logger.error(f"Failed to parse mutation output: {e}")
            return None
    
    def mutate_batch(
        self,
        prompts: List[str],
        evaluation_results: Optional[List[BatchEvaluationResult]] = None,
        strategy: Optional[MutationStrategy] = None
    ) -> List[MutationResult]:
        """
        Generate mutations for multiple prompts.
        
        Args:
            prompts: List of prompts to mutate
            evaluation_results: Optional evaluation results for each prompt
            strategy: Specific strategy to use (or random for each)
            
        Returns:
            List of MutationResults
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            eval_result = evaluation_results[i] if evaluation_results and i < len(evaluation_results) else None
            
            # Extract linguistic feedback if available
            linguistic_feedback = None
            if eval_result and eval_result.common_weaknesses:
                linguistic_feedback = "Common issues: " + ", ".join(eval_result.common_weaknesses)
            
            mutation = self.mutate(
                prompt,
                strategy=strategy,
                linguistic_feedback=linguistic_feedback,
                evaluation_result=eval_result
            )
            results.append(mutation)
        
        return results
    
    def create_crossover(
        self,
        prompt1: str,
        prompt2: str
    ) -> MutationResult:
        """
        Create a crossover between two prompts (genetic algorithm style).
        
        Args:
            prompt1: First parent prompt
            prompt2: Second parent prompt
            
        Returns:
            MutationResult with crossed-over prompt
        """
        crossover_prompt = f"""You are an expert prompt engineer. Create a new prompt by combining the best elements of these two prompts:

**Prompt 1:**
{prompt1}

**Prompt 2:**
{prompt2}

**Task:**
Create a new prompt that takes the best ideas from both prompts. The result should be:
- More effective than either parent
- Coherent and well-structured
- Incorporating strengths from both

**Output Format (JSON):**
{{
  "new_prompt": "The combined prompt text here...",
  "explanation": "How this combines the best of both prompts...",
  "diff_summary": "Elements taken from each prompt...",
  "expected_improvement": "Why this should be better..."
}}
"""
        
        try:
            response = self.llm_client.complete(crossover_prompt, temperature=0.7)
            mutation_data = self._parse_mutation_output(response.content)
            
            if not mutation_data:
                # Fallback: simple concatenation
                mutation_data = {
                    "new_prompt": f"{prompt1}\n\nAdditionally:\n{prompt2}",
                    "explanation": "Simple combination of both prompts",
                    "diff_summary": "Concatenated both prompts",
                    "expected_improvement": "Combined instructions"
                }
            
            return MutationResult(
                new_prompt=mutation_data["new_prompt"],
                strategy=MutationStrategy.STRUCTURAL,  # Closest match
                description=mutation_data.get("explanation", "Crossover applied"),
                diff_summary=mutation_data.get("diff_summary", "Combined prompts"),
                estimated_tokens=count_tokens(mutation_data["new_prompt"], self.llm_client.model),
                expected_improvement=mutation_data.get("expected_improvement", "Combined strengths"),
                parent_prompt=f"{prompt1[:50]}... x {prompt2[:50]}..."
            )
        
        except Exception as e:
            logger.error(f"Crossover generation failed: {e}")
            # Fallback to first prompt
            return MutationResult(
                new_prompt=prompt1,
                strategy=MutationStrategy.STRUCTURAL,
                description=f"Crossover failed: {e}",
                diff_summary="No crossover (error)",
                estimated_tokens=count_tokens(prompt1, self.llm_client.model),
                expected_improvement="None",
                parent_prompt=prompt1
            )
