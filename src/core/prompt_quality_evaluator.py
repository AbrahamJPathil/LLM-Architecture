"""
PromptQualityEvaluator: Direct prompt quality assessment without test cases.
Evaluates prompts based on intrinsic quality criteria.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import List, Optional

from src.core.llm_client import create_llm_client
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptQualityScore:
    """Quality assessment of a prompt."""
    
    overall_score: float  # 0.0 to 1.0
    
    # Quality dimensions
    clarity: float
    specificity: float
    structure: float
    completeness: float
    effectiveness: float
    
    # Feedback
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    
    # Raw output
    raw_feedback: Optional[str] = None


class PromptQualityEvaluator:
    """
    Evaluates prompt quality directly without test cases.
    Uses Judge LLM to assess intrinsic prompt characteristics.
    """
    
    def __init__(
        self,
        domain: str = "general",
        tier: str = "tier1",
        task_description: Optional[str] = None,
        user_context: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            domain: Domain name (legal, medical, code, etc.)
            tier: LLM tier to use for evaluation
            task_description: Description of what the prompt should accomplish
            user_context: Additional context about the use case
        """
        self.domain = domain
        self.task_description = task_description
        self.user_context = user_context
        self.judge_client = create_llm_client(tier=tier)
    
    async def evaluate_prompt_async(
        self,
        prompt: str,
        reference_prompt: Optional[str] = None
    ) -> PromptQualityScore:
        """
        Evaluate a single prompt's quality.
        
        Args:
            prompt: The prompt to evaluate
            reference_prompt: Optional reference prompt for comparison
            
        Returns:
            PromptQualityScore with detailed assessment
        """
        logger.debug(f"🔍 Evaluating prompt quality via Judge LLM (context: {self.user_context})")
        
        judge_prompt = self._build_judge_prompt(prompt, reference_prompt)
        
        logger.debug(f"Judge prompt length: {len(judge_prompt)} chars")
        
        response = await self.judge_client.complete_async(
            prompt=judge_prompt,
            temperature=0.3,
            max_tokens=800
        )
        
        logger.debug(f"✅ Judge response received: {len(response.content)} chars")
        
        # Parse response
        try:
            score = self._parse_judge_response(response.content)
            return score
        except Exception as e:
            logger.error(f"Failed to parse judge response: {e}")
            # Return neutral score on parse failure
            return PromptQualityScore(
                overall_score=0.5,
                clarity=0.5,
                specificity=0.5,
                structure=0.5,
                completeness=0.5,
                effectiveness=0.5,
                strengths=["Parse error - using fallback score"],
                weaknesses=[],
                suggestions=[],
                raw_feedback=response.content
            )
    
    def evaluate_prompt(
        self,
        prompt: str,
        reference_prompt: Optional[str] = None
    ) -> PromptQualityScore:
        """Synchronous wrapper for evaluate_prompt_async."""
        return asyncio.run(self.evaluate_prompt_async(prompt, reference_prompt))
    
    async def evaluate_batch_async(
        self,
        prompts: List[str],
        reference_prompt: Optional[str] = None
    ) -> List[PromptQualityScore]:
        """
        Evaluate multiple prompts in parallel.
        
        Args:
            prompts: List of prompts to evaluate
            reference_prompt: Optional reference for comparison
            
        Returns:
            List of PromptQualityScore objects
        """
        tasks = [
            self.evaluate_prompt_async(prompt, reference_prompt)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
    
    def _build_judge_prompt(
        self,
        prompt: str,
        reference_prompt: Optional[str] = None
    ) -> str:
        """Build the judge prompt for quality evaluation."""
        
        parts = [
            "You are an expert prompt engineer. Evaluate the quality of the following prompt.",
            ""
        ]
        
        if self.task_description:
            parts.extend([
                f"Task Context: {self.task_description}",
                ""
            ])
        
        if self.user_context:
            parts.extend([
                f"Use Case Context: {self.user_context}",
                ""
            ])
        
        if self.domain and self.domain != "general":
            parts.extend([
                f"Domain: {self.domain}",
                ""
            ])
        
        parts.extend([
            "Evaluate the prompt on these dimensions (0.0 to 1.0):",
            "1. CLARITY: Is it clear and unambiguous?",
            "2. SPECIFICITY: Does it provide specific instructions?",
            "3. STRUCTURE: Is it well-organized?",
            "4. COMPLETENESS: Does it cover all necessary aspects?",
            "5. EFFECTIVENESS: Will it likely produce good results?",
            ""
        ])
        
        if reference_prompt:
            parts.extend([
                "REFERENCE PROMPT (for comparison):",
                reference_prompt,
                "",
                "PROMPT TO EVALUATE:",
                prompt,
                ""
            ])
        else:
            parts.extend([
                "PROMPT TO EVALUATE:",
                prompt,
                ""
            ])
        
        parts.extend([
            "Respond ONLY with valid JSON in this exact format:",
            "{",
            '  "clarity": 0.85,',
            '  "specificity": 0.90,',
            '  "structure": 0.80,',
            '  "completeness": 0.75,',
            '  "effectiveness": 0.88,',
            '  "overall_score": 0.84,',
            '  "strengths": ["Clear instructions", "Good structure"],',
            '  "weaknesses": ["Could be more specific"],',
            '  "suggestions": ["Add examples", "Define output format"]',
            "}",
            "",
            "JSON Response:"
        ])
        
        return "\n".join(parts)
    
    def _parse_judge_response(self, response: str) -> PromptQualityScore:
        """Parse judge response into PromptQualityScore."""
        
        # Try to extract JSON
        response = response.strip()
        
        # Find JSON block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
        else:
            json_str = response
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}, response: {response[:200]}")
            raise
        
        # Extract scores
        return PromptQualityScore(
            overall_score=float(data.get("overall_score", 0.5)),
            clarity=float(data.get("clarity", 0.5)),
            specificity=float(data.get("specificity", 0.5)),
            structure=float(data.get("structure", 0.5)),
            completeness=float(data.get("completeness", 0.5)),
            effectiveness=float(data.get("effectiveness", 0.5)),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            suggestions=data.get("suggestions", []),
            raw_feedback=response
        )
