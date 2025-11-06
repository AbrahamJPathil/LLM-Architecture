"""
PromptVerifier: Two-stage evaluation system.
Stage 1: Deterministic checks (schema, regex, format)
Stage 2: Semantic evaluation using Judge LLM
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from src.core.llm_client import create_llm_client
from src.core.prompt_solver import ExecutionResult, PromptResult, TestCase
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeterministicCheck:
    """A deterministic validation check."""
    
    name: str
    description: str
    check_fn: Callable[[str, TestCase], bool]
    weight: float = 1.0
    critical: bool = False  # If True, failure means score = 0


@dataclass
class EvaluationFeedback:
    """Structured feedback from evaluation."""
    
    overall_quality: float  # 0.0 to 1.0
    
    # Specific aspects
    accuracy: float
    completeness: float
    clarity: float
    relevance: float
    
    # Textual feedback
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    
    # Deterministic check results
    deterministic_checks: Dict[str, bool] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single output."""
    
    test_case_id: str
    
    # Scores
    composite_score: float  # Final weighted score
    deterministic_score: float
    semantic_score: Optional[float]
    
    # Detailed feedback
    feedback: EvaluationFeedback
    
    # Pass/fail
    passed: bool
    
    # Raw judge output (for debugging)
    judge_raw_output: Optional[str] = None


@dataclass
class BatchEvaluationResult:
    """Results of evaluating a batch of outputs."""
    
    prompt_result: PromptResult
    evaluation_results: List[EvaluationResult]
    
    # Aggregate metrics
    avg_composite_score: float
    avg_deterministic_score: float
    avg_semantic_score: Optional[float]
    
    pass_rate: float
    
    # Consistency metrics
    consistency_score: float  # How similar are outputs for similar inputs
    
    # Aggregated feedback
    common_weaknesses: List[str]
    common_strengths: List[str]


class PromptVerifier:
    """
    Two-stage evaluation system:
    1. Deterministic checks (fast, rule-based)
    2. Semantic judge (LLM-based, nuanced)
    """
    
    def __init__(
        self,
        domain: str = "general",
        use_judge: bool = True,
        deterministic_checks: Optional[List[DeterministicCheck]] = None,
        evaluation_weights: Optional[Dict[str, float]] = None,
        max_concurrent_judge: int = 3,
        batch_judge: bool = True  # NEW: Use single batched judge call for speed
    ):
        """
        Initialize PromptVerifier.
        
        Args:
            domain: Domain for domain-specific checks
            use_judge: Whether to use Judge LLM
            deterministic_checks: Custom deterministic checks
            evaluation_weights: Weights for composite scoring
            max_concurrent_judge: Max concurrent per-case judge calls (when batch_judge=False)
            batch_judge: If True, use single batched judge call for all test cases (faster, cheaper)
        """
        self.domain = domain
        self.use_judge = use_judge
        self.max_concurrent_judge = max(1, max_concurrent_judge)
        self.batch_judge = batch_judge
        
        # Initialize Judge LLM if enabled
        if use_judge:
            self.judge_client = create_llm_client("judge")
        
        # Setup deterministic checks
        self.deterministic_checks = deterministic_checks or self._get_default_checks()
        
        # Evaluation weights
        self.weights = evaluation_weights or {
            "deterministic": 0.3,
            "semantic": 0.7
        }
        
        logger.info(
            "PromptVerifier initialized",
            domain=domain,
            use_judge=use_judge,
            batch_judge=batch_judge,
            num_checks=len(self.deterministic_checks)
        )
    
    def _get_default_checks(self) -> List[DeterministicCheck]:
        """Get default deterministic checks."""
        checks = [
            DeterministicCheck(
                name="non_empty",
                description="Output must not be empty",
                check_fn=lambda output, tc: len(output.strip()) > 0,
                weight=1.0,
                critical=True
            ),
            DeterministicCheck(
                name="length_reasonable",
                description="Output length should be reasonable (10-5000 chars)",
                check_fn=lambda output, tc: 10 <= len(output) <= 5000,
                weight=0.5,
                critical=False
            ),
            DeterministicCheck(
                name="no_error_markers",
                description="Output should not contain error markers",
                check_fn=lambda output, tc: not any(
                    marker in output.lower()
                    for marker in ["error:", "failed:", "cannot", "unable to"]
                ),
                weight=0.8,
                critical=False
            ),
        ]
        
        # Add domain-specific checks
        if self.domain == "legal":
            checks.extend(self._get_legal_checks())
        
        return checks
    
    def _get_legal_checks(self) -> List[DeterministicCheck]:
        """Get legal domain-specific checks."""
        return [
            DeterministicCheck(
                name="has_quotation",
                description="Legal analysis should quote relevant text",
                check_fn=lambda output, tc: '"' in output or "'" in output,
                weight=0.6,
                critical=False
            ),
            DeterministicCheck(
                name="structured_format",
                description="Output should have structure (bullets, numbers, sections)",
                check_fn=lambda output, tc: any(
                    marker in output
                    for marker in ["\n-", "\n*", "\n1.", "\n2.", "##", "Section"]
                ),
                weight=0.5,
                critical=False
            ),
        ]
    
    def evaluate_single(
        self,
        output: str,
        test_case: TestCase,
        rubric: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a single output against a test case.
        
        Args:
            output: LLM output to evaluate
            test_case: Test case with expected output
            rubric: Optional evaluation rubric
            
        Returns:
            EvaluationResult
        """
        # Stage 1: Deterministic checks
        deterministic_results = {}
        deterministic_score = 0.0
        total_weight = 0.0
        
        for check in self.deterministic_checks:
            try:
                passed = check.check_fn(output, test_case)
                deterministic_results[check.name] = passed
                
                if passed:
                    deterministic_score += check.weight
                elif check.critical:
                    # Critical check failed, score is 0
                    logger.warning(
                        f"Critical check failed: {check.name}",
                        test_case_id=test_case.id
                    )
                    return self._create_failed_result(
                        test_case.id,
                        deterministic_results,
                        f"Critical check failed: {check.name}"
                    )
                
                total_weight += check.weight
            
            except Exception as e:
                logger.error(f"Check {check.name} failed: {e}")
                deterministic_results[check.name] = False
        
        # Normalize deterministic score
        deterministic_score = deterministic_score / total_weight if total_weight > 0 else 0.0
        
        # Stage 2: Semantic evaluation with Judge LLM
        semantic_score = None
        feedback = None
        judge_raw = None
        
        if self.use_judge:
            feedback, judge_raw = self._judge_evaluate(
                output,
                test_case,
                rubric
            )
            semantic_score = feedback.overall_quality if feedback else 0.0
        else:
            # Create basic feedback without Judge
            feedback = EvaluationFeedback(
                overall_quality=deterministic_score,
                accuracy=deterministic_score,
                completeness=deterministic_score,
                clarity=deterministic_score,
                relevance=deterministic_score,
                strengths=["Passed deterministic checks"],
                weaknesses=[],
                suggestions=[],
                deterministic_checks=deterministic_results
            )
        
        # Compute composite score
        if semantic_score is not None:
            composite_score = (
                self.weights["deterministic"] * deterministic_score +
                self.weights["semantic"] * semantic_score
            )
        else:
            composite_score = deterministic_score
        
        return EvaluationResult(
            test_case_id=test_case.id,
            composite_score=composite_score,
            deterministic_score=deterministic_score,
            semantic_score=semantic_score,
            feedback=feedback,
            passed=composite_score >= 0.5,  # Threshold for passing
            judge_raw_output=judge_raw
        )
    
    def _judge_evaluate(
        self,
        output: str,
        test_case: TestCase,
        rubric: Optional[Dict[str, Any]]
    ) -> tuple[Optional[EvaluationFeedback], Optional[str]]:
        """
        Use Judge LLM to evaluate output semantically.
        
        Returns:
            (EvaluationFeedback, raw_output) tuple
        """
        judge_prompt = self._create_judge_prompt(output, test_case, rubric)
        
        try:
            response = self.judge_client.complete(
                judge_prompt,
                temperature=0.0  # Deterministic evaluation
            )
            
            # Parse structured feedback from Judge
            feedback = self._parse_judge_output(response.content)
            return feedback, response.content
        
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}", test_case_id=test_case.id)
            return None, None
    
    def _create_judge_prompt(
        self,
        output: str,
        test_case: TestCase,
        rubric: Optional[Dict[str, Any]]
    ) -> str:
        """Create prompt for Judge LLM."""
        prompt_parts = [
            "You are an expert evaluator. Evaluate the following output based on the task and expected criteria.",
            "",
            f"**Task Input:**\n{test_case.input}",
        ]
        
        if test_case.context:
            prompt_parts.append(f"\n**Context:**\n{test_case.context}")
        
        if test_case.expected_output:
            prompt_parts.append(f"\n**Expected Output (Reference):**\n{test_case.expected_output}")
        
        prompt_parts.append(f"\n**Actual Output to Evaluate:**\n{output}")
        
        if rubric:
            prompt_parts.append(f"\n**Evaluation Rubric:**\n{json.dumps(rubric, indent=2)}")
        
        prompt_parts.append("""
\n**Please evaluate the output and provide feedback in JSON format:**

{
  "overall_quality": 0.85,
  "accuracy": 0.9,
  "completeness": 0.8,
  "clarity": 0.85,
  "relevance": 0.9,
  "strengths": ["Clear structure", "Comprehensive coverage"],
  "weaknesses": ["Missing some edge cases"],
  "suggestions": ["Add examples", "Clarify ambiguous terms"]
}

Scores should be between 0.0 and 1.0.
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_judge_output(self, judge_output: str) -> Optional[EvaluationFeedback]:
        """Parse structured feedback from Judge LLM output."""
        try:
            # Extract JSON from output
            if "```json" in judge_output:
                start = judge_output.find("```json") + 7
                end = judge_output.find("```", start)
                json_str = judge_output[start:end].strip()
            elif "```" in judge_output:
                start = judge_output.find("```") + 3
                end = judge_output.find("```", start)
                json_str = judge_output[start:end].strip()
            else:
                # Try to find JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', judge_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = judge_output.strip()
            
            data = json.loads(json_str)
            
            return EvaluationFeedback(
                overall_quality=data.get("overall_quality", 0.5),
                accuracy=data.get("accuracy", 0.5),
                completeness=data.get("completeness", 0.5),
                clarity=data.get("clarity", 0.5),
                relevance=data.get("relevance", 0.5),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                suggestions=data.get("suggestions", [])
            )
        
        except Exception as e:
            logger.error(f"Failed to parse judge output: {e}")
            return None
    
    async def _judge_evaluate_async(
        self,
        output: str,
        test_case: TestCase,
        rubric: Optional[Dict[str, Any]]
    ) -> tuple[Optional[EvaluationFeedback], Optional[str]]:
        """
        Use Judge LLM to evaluate output semantically (ASYNC).
        
        Returns:
            (EvaluationFeedback, raw_output) tuple
        """
        judge_prompt = self._create_judge_prompt(output, test_case, rubric)
        
        try:
            response = await self.judge_client.complete_async(
                prompt=judge_prompt,
                temperature=0.0  # Deterministic evaluation
            )
            
            # Parse structured feedback from Judge
            feedback = self._parse_judge_output(response.content)
            return feedback, response.content
        
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}", test_case_id=test_case.id)
            return None, None
    
    async def evaluate_single_async(
        self,
        output: str,
        test_case: TestCase,
        rubric: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a single output against a test case (ASYNC).
        
        Args:
            output: LLM output to evaluate
            test_case: Test case with expected output
            rubric: Optional evaluation rubric
            
        Returns:
            EvaluationResult
        """
        # Stage 1: Deterministic checks (still synchronous - fast)
        deterministic_results = {}
        deterministic_score = 0.0
        total_weight = 0.0
        
        for check in self.deterministic_checks:
            try:
                passed = check.check_fn(output, test_case)
                deterministic_results[check.name] = passed
                
                if passed:
                    deterministic_score += check.weight
                elif check.critical:
                    # Critical check failed, score is 0
                    logger.warning(
                        f"Critical check failed: {check.name}",
                        test_case_id=test_case.id
                    )
                    return self._create_failed_result(
                        test_case.id,
                        deterministic_results,
                        f"Critical check failed: {check.name}"
                    )
                
                total_weight += check.weight
            
            except Exception as e:
                logger.error(f"Check {check.name} failed: {e}")
                deterministic_results[check.name] = False
        
        # Normalize deterministic score
        deterministic_score = deterministic_score / total_weight if total_weight > 0 else 0.0
        
        # Stage 2: Semantic evaluation with Judge LLM (ASYNC)
        semantic_score = None
        feedback = None
        judge_raw = None
        
        if self.use_judge:
            feedback, judge_raw = await self._judge_evaluate_async(
                output,
                test_case,
                rubric
            )
            semantic_score = feedback.overall_quality if feedback else 0.0
        else:
            # Create basic feedback without Judge
            feedback = EvaluationFeedback(
                overall_quality=deterministic_score,
                accuracy=deterministic_score,
                completeness=deterministic_score,
                clarity=deterministic_score,
                relevance=deterministic_score,
                strengths=["Passed deterministic checks"],
                weaknesses=[],
                suggestions=[],
                deterministic_checks=deterministic_results
            )
        
        # Compute composite score
        if semantic_score is not None:
            composite_score = (
                self.weights["deterministic"] * deterministic_score +
                self.weights["semantic"] * semantic_score
            )
        else:
            composite_score = deterministic_score
        
        return EvaluationResult(
            test_case_id=test_case.id,
            composite_score=composite_score,
            deterministic_score=deterministic_score,
            semantic_score=semantic_score,
            feedback=feedback,
            passed=composite_score >= 0.5,  # Threshold for passing
            judge_raw_output=judge_raw
        )
    
    async def evaluate_batch_async(
        self,
        prompt_result: PromptResult,
        test_cases: List[TestCase],
        rubric: Optional[Dict[str, Any]] = None
    ) -> BatchEvaluationResult:
        """
        Evaluate all outputs in a PromptResult (ASYNC with parallel or batched Judge calls).
        
        Args:
            prompt_result: Results from PromptSolver
            test_cases: Original test cases
            rubric: Optional evaluation rubric
            
        Returns:
            BatchEvaluationResult
        """
        # Create test case lookup
        test_case_map = {tc.id: tc for tc in test_cases}
        
        # Evaluate each result - first do deterministic checks for all (fast, can be serial)
        deterministic_results = {}
        for exec_result in prompt_result.results:
            if not exec_result.success:
                # Execution failed, create failed evaluation
                deterministic_results[exec_result.test_case_id] = self._create_failed_result(
                    exec_result.test_case_id,
                    {},
                    f"Execution error: {exec_result.error}"
                )
            else:
                test_case = test_case_map.get(exec_result.test_case_id)
                if not test_case:
                    logger.warning(f"Test case {exec_result.test_case_id} not found")
                    continue

                # Do deterministic checks (fast)
                det_score = 0.0
                total_weight = 0.0
                det_check_results = {}
                
                # Fast synchronous deterministic checks
                for check in self.deterministic_checks:
                    try:
                        passed = check.check_fn(exec_result.output, test_case)
                        det_check_results[check.name] = passed
                        
                        if passed:
                            det_score += check.weight
                        elif check.critical:
                            deterministic_results[exec_result.test_case_id] = self._create_failed_result(
                                exec_result.test_case_id,
                                det_check_results,
                                f"Critical check failed: {check.name}"
                            )
                            break
                        
                        total_weight += check.weight
                    
                    except Exception as e:
                        logger.error(f"Check {check.name} failed: {e}")
                        det_check_results[check.name] = False

                # Store for async Judge eval
                if exec_result.test_case_id not in deterministic_results:
                    deterministic_results[exec_result.test_case_id] = (
                        exec_result,
                        test_case,
                        det_score / total_weight if total_weight > 0 else 0.0,
                        det_check_results
                    )

        # BATCH JUDGE MODE: Single call for all test cases
        if self.use_judge and self.batch_judge:
            evaluation_results = await self._evaluate_batch_judge(
                deterministic_results,
                rubric
            )
        # PARALLEL JUDGE MODE: Per-case calls with concurrency limit
        elif self.use_judge:
            evaluation_results = await self._evaluate_parallel_judge(
                deterministic_results,
                rubric
            )
        # NO JUDGE: Use deterministic scores only
        else:
            evaluation_results = self._evaluate_deterministic_only(deterministic_results)
        
        # Filter out None results
        evaluation_results = [r for r in evaluation_results if r is not None]
        
        # Compute aggregate metrics
        return self._create_batch_result(prompt_result, evaluation_results)
    
    async def _evaluate_batch_judge(
        self,
        deterministic_results: Dict[str, Any],
        rubric: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """
        Use a SINGLE batched Judge call to evaluate all test cases.
        Much faster and cheaper than per-case calls.
        """
        # Separate failed vs need-judge
        failed = [res for res in deterministic_results.values() if isinstance(res, EvaluationResult)]
        to_judge = [(tc_id, res) for tc_id, res in deterministic_results.items() if not isinstance(res, EvaluationResult)]
        
        if not to_judge:
            return failed
        
        # Build combined prompt for Judge
        combined_prompt = self._build_batch_judge_prompt(to_judge, rubric)
        
        # Single Judge call
        try:
            response = await self.judge_client.complete_async(
                combined_prompt,
                temperature=0.0
            )
            
            # Parse batched response
            judge_scores = self._parse_batch_judge_response(response.content, to_judge)
            
        except Exception as e:
            logger.error(f"Batch judge call failed: {e}")
            # Fallback: use deterministic scores only
            judge_scores = {}
        
        # Build EvaluationResults
        results = list(failed)
        for tc_id, (exec_result, test_case, det_score, det_checks) in to_judge:
            judge_data = judge_scores.get(tc_id, {})
            
            feedback = EvaluationFeedback(
                overall_quality=judge_data.get("overall_quality", det_score),
                accuracy=judge_data.get("accuracy", det_score),
                completeness=judge_data.get("completeness", det_score),
                clarity=judge_data.get("clarity", det_score),
                relevance=judge_data.get("relevance", det_score),
                strengths=judge_data.get("strengths", ["Passed deterministic checks"]),
                weaknesses=judge_data.get("weaknesses", []),
                suggestions=judge_data.get("suggestions", []),
                deterministic_checks=det_checks
            )
            
            semantic_score = feedback.overall_quality
            composite_score = (
                self.weights["deterministic"] * det_score +
                self.weights["semantic"] * semantic_score
            )
            
            results.append(EvaluationResult(
                test_case_id=tc_id,
                composite_score=composite_score,
                deterministic_score=det_score,
                semantic_score=semantic_score,
                feedback=feedback,
                passed=composite_score >= 0.5,
                judge_raw_output=judge_data.get("raw", "")
            ))
        
        return results
    
    async def _evaluate_parallel_judge(
        self,
        deterministic_results: Dict[str, Any],
        rubric: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """
        Use PARALLEL per-case Judge calls with concurrency limit.
        Original behavior - more expensive but granular.
        """
        async def evaluate_with_judge(tc_id, det_result, sem):
            if isinstance(det_result, EvaluationResult):
                return det_result
            
            exec_result, test_case, det_score, det_checks = det_result
            
            async with sem:
                feedback, judge_raw = await self._judge_evaluate_async(
                    exec_result.output,
                    test_case,
                    rubric
                )
            
            semantic_score = feedback.overall_quality if feedback else 0.0
            composite_score = (
                self.weights["deterministic"] * det_score +
                self.weights["semantic"] * semantic_score
            ) if feedback else det_score
            
            return EvaluationResult(
                test_case_id=tc_id,
                composite_score=composite_score,
                deterministic_score=det_score,
                semantic_score=semantic_score,
                feedback=feedback or EvaluationFeedback(
                    overall_quality=det_score,
                    accuracy=det_score,
                    completeness=det_score,
                    clarity=det_score,
                    relevance=det_score,
                    strengths=["Passed deterministic checks"],
                    weaknesses=[],
                    suggestions=[],
                    deterministic_checks=det_checks
                ),
                passed=composite_score >= 0.5,
                judge_raw_output=judge_raw
            )
        
        sem = asyncio.Semaphore(self.max_concurrent_judge)
        return await asyncio.gather(*[
            evaluate_with_judge(tc_id, det_result, sem)
            for tc_id, det_result in deterministic_results.items()
        ])
    
    def _evaluate_deterministic_only(
        self,
        deterministic_results: Dict[str, Any]
    ) -> List[EvaluationResult]:
        """Use deterministic scores only (no Judge)."""
        results = []
        for tc_id, det_result in deterministic_results.items():
            if isinstance(det_result, EvaluationResult):
                results.append(det_result)
            else:
                exec_result, test_case, det_score, det_checks = det_result
                feedback = EvaluationFeedback(
                    overall_quality=det_score,
                    accuracy=det_score,
                    completeness=det_score,
                    clarity=det_score,
                    relevance=det_score,
                    strengths=["Passed deterministic checks"],
                    weaknesses=[],
                    suggestions=[],
                    deterministic_checks=det_checks
                )
                results.append(EvaluationResult(
                    test_case_id=tc_id,
                    composite_score=det_score,
                    deterministic_score=det_score,
                    semantic_score=None,
                    feedback=feedback,
                    passed=det_score >= 0.5
                ))
        return results
    
    def _build_batch_judge_prompt(
        self,
        to_judge: List[tuple],
        rubric: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build a combined prompt for batch judge evaluation."""
        prompt_parts = [
            "You are an expert evaluator. Evaluate the quality of each LLM output below.",
            "",
            "For EACH test case, provide scores (0.0 to 1.0) and feedback.",
            "",
            "Return a JSON object with test_case_id as keys:",
            '{"test_case_id": {"overall_quality": 0.8, "accuracy": 0.9, "completeness": 0.7, "clarity": 0.8, "relevance": 0.9, "strengths": ["..."], "weaknesses": ["..."], "suggestions": ["..."]}}',
            "",
            "Test Cases:"
        ]
        
        for tc_id, (exec_result, test_case, det_score, det_checks) in to_judge:
            prompt_parts.append(f"\n=== Test Case: {tc_id} ===")
            if test_case.expected_output:
                prompt_parts.append(f"Expected: {test_case.expected_output[:200]}...")
            prompt_parts.append(f"Actual Output: {exec_result.output[:500]}...")
            prompt_parts.append("")
        
        prompt_parts.append("\nReturn ONLY valid JSON with scores for each test case ID.")
        return "\n".join(prompt_parts)
    
    def _parse_batch_judge_response(
        self,
        response: str,
        to_judge: List[tuple]
    ) -> Dict[str, Dict]:
        """Parse the batched Judge response into per-case scores."""
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            scores = json.loads(response.strip())
            
            # Ensure all test cases have entries
            for tc_id, _ in to_judge:
                if tc_id not in scores:
                    logger.warning(f"Judge did not return score for {tc_id}")
                    scores[tc_id] = {
                        "overall_quality": 0.5,
                        "accuracy": 0.5,
                        "completeness": 0.5,
                        "clarity": 0.5,
                        "relevance": 0.5,
                        "strengths": [],
                        "weaknesses": ["Judge failed to evaluate"],
                        "suggestions": []
                    }
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to parse batch judge response: {e}")
            # Return fallback scores for all
            return {
                tc_id: {
                    "overall_quality": 0.5,
                    "accuracy": 0.5,
                    "completeness": 0.5,
                    "clarity": 0.5,
                    "relevance": 0.5,
                    "strengths": [],
                    "weaknesses": ["Parse error"],
                    "suggestions": []
                }
                for tc_id, _ in to_judge
            }
    
    def _create_failed_result(
        self,
        test_case_id: str,
        deterministic_results: Dict[str, bool],
        reason: str
    ) -> EvaluationResult:
        """Create a failed evaluation result."""
        feedback = EvaluationFeedback(
            overall_quality=0.0,
            accuracy=0.0,
            completeness=0.0,
            clarity=0.0,
            relevance=0.0,
            strengths=[],
            weaknesses=[reason],
            suggestions=["Fix critical issues"],
            deterministic_checks=deterministic_results
        )
        
        return EvaluationResult(
            test_case_id=test_case_id,
            composite_score=0.0,
            deterministic_score=0.0,
            semantic_score=0.0,
            feedback=feedback,
            passed=False
        )
    
    def evaluate_batch(
        self,
        prompt_result: PromptResult,
        test_cases: List[TestCase],
        rubric: Optional[Dict[str, Any]] = None
    ) -> BatchEvaluationResult:
        """
        Evaluate all outputs in a PromptResult.
        
        Args:
            prompt_result: Results from PromptSolver
            test_cases: Original test cases
            rubric: Optional evaluation rubric
            
        Returns:
            BatchEvaluationResult
        """
        # Create test case lookup
        test_case_map = {tc.id: tc for tc in test_cases}
        
        # Evaluate each result
        evaluation_results = []
        for exec_result in prompt_result.results:
            if not exec_result.success:
                # Execution failed, create failed evaluation
                eval_result = self._create_failed_result(
                    exec_result.test_case_id,
                    {},
                    f"Execution error: {exec_result.error}"
                )
            else:
                test_case = test_case_map.get(exec_result.test_case_id)
                if not test_case:
                    logger.warning(f"Test case {exec_result.test_case_id} not found")
                    continue
                
                eval_result = self.evaluate_single(
                    exec_result.output,
                    test_case,
                    rubric
                )
            
            evaluation_results.append(eval_result)
        
        # Compute aggregate metrics
        return self._create_batch_result(prompt_result, evaluation_results)
    
    def _create_batch_result(
        self,
        prompt_result: PromptResult,
        evaluation_results: List[EvaluationResult]
    ) -> BatchEvaluationResult:
        """Create BatchEvaluationResult from individual evaluations."""
        if not evaluation_results:
            # No results to evaluate
            return BatchEvaluationResult(
                prompt_result=prompt_result,
                evaluation_results=[],
                avg_composite_score=0.0,
                avg_deterministic_score=0.0,
                avg_semantic_score=None,
                pass_rate=0.0,
                consistency_score=0.0,
                common_weaknesses=[],
                common_strengths=[]
            )
        
        # Calculate averages
        avg_composite = sum(r.composite_score for r in evaluation_results) / len(evaluation_results)
        avg_deterministic = sum(r.deterministic_score for r in evaluation_results) / len(evaluation_results)
        
        semantic_scores = [r.semantic_score for r in evaluation_results if r.semantic_score is not None]
        avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else None
        
        pass_rate = sum(1 for r in evaluation_results if r.passed) / len(evaluation_results)
        
        # Calculate consistency (variance of scores)
        scores = [r.composite_score for r in evaluation_results]
        variance = sum((s - avg_composite) ** 2 for s in scores) / len(scores)
        consistency_score = max(0.0, 1.0 - variance)  # Higher is more consistent
        
        # Aggregate feedback
        all_weaknesses = []
        all_strengths = []
        for r in evaluation_results:
            if r.feedback:  # Check if feedback exists
                all_weaknesses.extend(r.feedback.weaknesses)
                all_strengths.extend(r.feedback.strengths)
        
        # Find common themes (simple frequency count)
        weakness_freq = {}
        for w in all_weaknesses:
            weakness_freq[w] = weakness_freq.get(w, 0) + 1
        
        strength_freq = {}
        for s in all_strengths:
            strength_freq[s] = strength_freq.get(s, 0) + 1
        
        common_weaknesses = sorted(weakness_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        common_strengths = sorted(strength_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return BatchEvaluationResult(
            prompt_result=prompt_result,
            evaluation_results=evaluation_results,
            avg_composite_score=avg_composite,
            avg_deterministic_score=avg_deterministic,
            avg_semantic_score=avg_semantic,
            pass_rate=pass_rate,
            consistency_score=consistency_score,
            common_weaknesses=[w[0] for w in common_weaknesses],
            common_strengths=[s[0] for s in common_strengths]
        )
