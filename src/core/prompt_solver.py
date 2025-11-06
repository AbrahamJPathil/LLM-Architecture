"""
PromptSolver: Batch execution engine for running prompts against test cases.
Supports both Tier-1 (production) and Tier-3 (iteration) LLMs.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.llm_client import LLMClient, LLMResponse, create_llm_client
from src.utils.cost_tracker import CostTracker, TokenUsage
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TestCase:
    """A single test case for prompt evaluation."""
    
    id: str
    input: str
    context: Optional[str] = None
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt(self, base_prompt: str) -> str:
        """
        Combine base prompt with test case input and context.
        
        Args:
            base_prompt: The prompt template/instruction
            
        Returns:
            Complete prompt ready for LLM
        """
        parts = [base_prompt]
        
        if self.context:
            parts.append(f"\n\nContext:\n{self.context}")
        
        parts.append(f"\n\nInput:\n{self.input}")
        
        return "\n".join(parts)


@dataclass
class ExecutionResult:
    """Result of executing a prompt on a single test case."""
    
    test_case_id: str
    output: str
    
    # Performance metrics
    latency_ms: float
    input_tokens: int
    output_tokens: int
    
    # Metadata
    model: str
    provider: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Whether execution completed successfully."""
        return self.error is None
    
    @property
    def token_usage(self) -> TokenUsage:
        """Get TokenUsage for cost tracking."""
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model=self.model
        )


@dataclass
class PromptResult:
    """Results of executing a prompt across multiple test cases."""
    
    prompt_text: str
    prompt_hash: str
    
    # Results per test case
    results: List[ExecutionResult]
    
    # Aggregate metrics
    total_latency_ms: float
    avg_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    
    # Success metrics
    successful_count: int
    failed_count: int
    
    # Metadata
    model: str
    tier: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Percentage of successful executions."""
        total = self.successful_count + self.failed_count
        return self.successful_count / total if total > 0 else 0.0


class PromptSolver:
    """
    Executes prompts against test cases with performance tracking.
    Supports both synchronous and asynchronous batch execution.
    """
    
    def __init__(
        self,
        tier: str = "tier3",
        cost_tracker: Optional[CostTracker] = None,
        max_concurrent: int = 10
    ):
        """
        Initialize PromptSolver.
        
        Args:
            tier: LLM tier to use ('tier1', 'tier3', 'judge')
            cost_tracker: Optional cost tracker for monitoring
            max_concurrent: Max parallel executions
        """
        self.tier = tier
        self.llm_client = create_llm_client(tier)
        self.cost_tracker = cost_tracker
        self.max_concurrent = max_concurrent
        
        logger.info(
            "PromptSolver initialized",
            tier=tier,
            model=self.llm_client.model,
            provider=self.llm_client.provider
        )
    
    def _compute_prompt_hash(self, prompt: str) -> str:
        """Generate hash for prompt deduplication."""
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def execute_single(
        self,
        prompt: str,
        test_case: TestCase,
        **llm_kwargs
    ) -> ExecutionResult:
        """
        Execute prompt on a single test case (synchronous).
        
        Args:
            prompt: The prompt template/instruction
            test_case: Test case to evaluate
            **llm_kwargs: Additional LLM parameters
            
        Returns:
            ExecutionResult
        """
        test_prompt = test_case.to_prompt(prompt)
        
        try:
            response = self.llm_client.complete(test_prompt, **llm_kwargs)
            
            # Track cost if tracker provided
            if self.cost_tracker:
                self.cost_tracker.add_usage(response.token_usage)
            
            return ExecutionResult(
                test_case_id=test_case.id,
                output=response.content,
                latency_ms=response.latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                model=response.model,
                provider=response.provider
            )
        
        except Exception as e:
            logger.error(
                f"Execution failed for test case {test_case.id}",
                error=str(e),
                test_case_id=test_case.id
            )
            
            return ExecutionResult(
                test_case_id=test_case.id,
                output="",
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                model=self.llm_client.model,
                provider=self.llm_client.provider,
                error=str(e)
            )
    
    async def execute_single_async(
        self,
        prompt: str,
        test_case: TestCase,
        **llm_kwargs
    ) -> ExecutionResult:
        """
        Execute prompt on a single test case (asynchronous).
        
        Args:
            prompt: The prompt template/instruction
            test_case: Test case to evaluate
            **llm_kwargs: Additional LLM parameters
            
        Returns:
            ExecutionResult
        """
        test_prompt = test_case.to_prompt(prompt)
        
        try:
            response = await self.llm_client.complete_async(test_prompt, **llm_kwargs)
            
            # Track cost if tracker provided
            if self.cost_tracker:
                self.cost_tracker.add_usage(response.token_usage)
            
            return ExecutionResult(
                test_case_id=test_case.id,
                output=response.content,
                latency_ms=response.latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                model=response.model,
                provider=response.provider
            )
        
        except Exception as e:
            logger.error(
                f"Execution failed for test case {test_case.id}",
                error=str(e),
                test_case_id=test_case.id
            )
            
            return ExecutionResult(
                test_case_id=test_case.id,
                output="",
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                model=self.llm_client.model,
                provider=self.llm_client.provider,
                error=str(e)
            )
    
    def execute_batch(
        self,
        prompt: str,
        test_cases: List[TestCase],
        **llm_kwargs
    ) -> PromptResult:
        """
        Execute prompt against multiple test cases synchronously.
        
        Args:
            prompt: The prompt template/instruction
            test_cases: List of test cases to evaluate
            **llm_kwargs: Additional LLM parameters
            
        Returns:
            PromptResult with all execution results
        """
        logger.info(
            f"Executing batch of {len(test_cases)} test cases",
            tier=self.tier,
            test_count=len(test_cases)
        )
        
        results = []
        for test_case in test_cases:
            result = self.execute_single(prompt, test_case, **llm_kwargs)
            results.append(result)
        
        return self._create_prompt_result(prompt, results)
    
    async def execute_batch_async(
        self,
        prompt: str,
        test_cases: List[TestCase],
        batch_mode: bool = True,  # NEW: Combine all test cases into one API call
        **llm_kwargs
    ) -> PromptResult:
        """
        Execute prompt against multiple test cases asynchronously.
        
        Args:
            prompt: The prompt template/instruction
            test_cases: List of test cases to evaluate
            batch_mode: If True, combine all test cases into one API call (faster)
            **llm_kwargs: Additional LLM parameters
            
        Returns:
            PromptResult with all execution results
        """
        # FAST MODE: Combine all test cases into one API call
        if batch_mode and len(test_cases) > 1:
            return await self._execute_batch_combined(prompt, test_cases, **llm_kwargs)
        
        # ORIGINAL MODE: Individual API calls per test case
        logger.info(
            f"Executing batch of {len(test_cases)} test cases (async)",
            tier=self.tier,
            test_count=len(test_cases),
            max_concurrent=self.max_concurrent
        )
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def execute_with_semaphore(test_case: TestCase) -> ExecutionResult:
            async with semaphore:
                return await self.execute_single_async(prompt, test_case, **llm_kwargs)
        
        # Execute all in parallel (limited by semaphore)
        tasks = [execute_with_semaphore(tc) for tc in test_cases]
        results = await asyncio.gather(*tasks)
        
        return self._create_prompt_result(prompt, list(results))
    
    async def _execute_batch_combined(
        self,
        prompt: str,
        test_cases: List[TestCase],
        **llm_kwargs
    ) -> PromptResult:
        """
        FAST: Execute all test cases in a single API call.
        Combines all inputs into one prompt.
        """
        import time
        
        logger.info(
            f"Executing {len(test_cases)} test cases in SINGLE API call (fast mode)",
            tier=self.tier
        )
        
        # Combine all test cases into one prompt
        combined_prompt = f"{prompt}\n\n"
        combined_prompt += "Process each of the following inputs separately:\n\n"
        
        for i, tc in enumerate(test_cases, 1):
            combined_prompt += f"=== Input {i} (ID: {tc.id}) ===\n"
            if tc.context:
                combined_prompt += f"Context: {tc.context}\n"
            combined_prompt += f"{tc.input}\n\n"
        
        combined_prompt += "\nProvide output for each input in the same order, clearly labeled."
        
        # Make single API call
        start_time = time.time()
        response = await self.llm_client.complete_async(combined_prompt, **llm_kwargs)
        latency_ms = (time.time() - start_time) * 1000
        
        # Split response back into individual results
        # Simple split by "=== Input N ===" markers
        output_parts = response.content.split("=== Input")
        
        results = []
        for i, tc in enumerate(test_cases):
            # Try to extract output for this test case
            if i + 1 < len(output_parts):
                output = output_parts[i + 1].split("===")[1] if "===" in output_parts[i + 1] else output_parts[i + 1]
                output = output.strip()
            else:
                output = response.content  # Fallback: use entire response
            
            # Create result for this test case
            results.append(ExecutionResult(
                test_case_id=tc.id,
                output=output,
                latency_ms=latency_ms / len(test_cases),  # Divide latency
                input_tokens=response.input_tokens // len(test_cases),  # Approximate
                output_tokens=response.output_tokens // len(test_cases),
                model=response.model,
                provider=response.provider
            ))
        
        return self._create_prompt_result(prompt, results)
    
    def _create_prompt_result(
        self,
        prompt: str,
        results: List[ExecutionResult]
    ) -> PromptResult:
        """Create PromptResult from execution results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_latency = sum(r.latency_ms for r in results)
        avg_latency = total_latency / len(results) if results else 0.0
        
        total_tokens = sum(r.input_tokens + r.output_tokens for r in results)
        total_cost = sum(r.token_usage.cost_usd for r in results)
        
        prompt_hash = self._compute_prompt_hash(prompt)
        
        return PromptResult(
            prompt_text=prompt,
            prompt_hash=prompt_hash,
            results=results,
            total_latency_ms=total_latency,
            avg_latency_ms=avg_latency,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            successful_count=len(successful),
            failed_count=len(failed),
            model=self.llm_client.model,
            tier=self.tier
        )
    
    def switch_tier(self, new_tier: str):
        """
        Switch to a different LLM tier.
        
        Args:
            new_tier: New tier ('tier1', 'tier3', 'judge')
        """
        logger.info(f"Switching from {self.tier} to {new_tier}")
        self.tier = new_tier
        self.llm_client = create_llm_client(new_tier)


def parse_structured_output(
    output: str,
    expected_format: str = "json"
) -> Optional[Dict[str, Any]]:
    """
    Parse structured output from LLM response.
    
    Args:
        output: Raw LLM output
        expected_format: Expected format ('json', 'xml', etc.)
        
    Returns:
        Parsed data or None if parsing fails
    """
    if expected_format == "json":
        try:
            # Try to extract JSON from markdown code blocks
            if "```json" in output:
                start = output.find("```json") + 7
                end = output.find("```", start)
                json_str = output[start:end].strip()
            elif "```" in output:
                start = output.find("```") + 3
                end = output.find("```", start)
                json_str = output[start:end].strip()
            else:
                json_str = output.strip()
            
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON output: {e}")
            return None
    
    return None
