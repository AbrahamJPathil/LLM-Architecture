"""
Simple Task Analyzer - Extracts essential context from user prompts.

This module provides a lightweight task analysis system that extracts:
- Domain and context
- Core objectives
- Task complexity
- Constraints and requirements

Used as the first step in the prompt evolution pipeline.
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class TaskAnalysis:
    """Results from analyzing a user prompt."""
    domain: str
    objectives: list[str]
    complexity: str  # "simple", "moderate", "complex"
    constraints: list[str]
    context: str
    required_skills: list[str]
    raw_prompt: str


class SimpleTaskAnalyzer:
    """
    Analyzes user prompts to extract essential information for prompt evolution.
    
    This is a simplified version focused on extracting only what's needed for
    scenario generation and prompt evolution, without unnecessary features like
    user profiles, chat history, or batch processing.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini"):
        """
        Initialize the task analyzer.
        
        Args:
            api_key: OpenAI API key (uses environment variable if not provided)
            model: Model to use for analysis (default: gpt-5-mini)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized SimpleTaskAnalyzer with model: {model}")
    
    def analyze_prompt(self, user_prompt: str) -> TaskAnalysis:
        """
        Analyze a user prompt to extract essential task information.
        
        Args:
            user_prompt: The raw prompt from the user
            
        Returns:
            TaskAnalysis object with extracted information
        """
        logger.info("Analyzing user prompt...")
        
        # Create analysis prompt
        analysis_prompt = self._create_analysis_prompt(user_prompt)
        
        try:
            # Call LLM for analysis
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing tasks and extracting structured information. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Create TaskAnalysis object
            analysis = TaskAnalysis(
                domain=result.get("domain", "general"),
                objectives=result.get("objectives", []),
                complexity=result.get("complexity", "moderate"),
                constraints=result.get("constraints", []),
                context=result.get("context", ""),
                required_skills=result.get("required_skills", []),
                raw_prompt=user_prompt
            )
            
            logger.info(f"Analysis complete - Domain: {analysis.domain}, Complexity: {analysis.complexity}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error during prompt analysis: {e}")
            # Return basic analysis on error
            return TaskAnalysis(
                domain="general",
                objectives=["Process user request"],
                complexity="moderate",
                constraints=[],
                context=user_prompt,
                required_skills=["general knowledge"],
                raw_prompt=user_prompt
            )
    
    def _create_analysis_prompt(self, user_prompt: str) -> str:
        """Create the prompt for LLM analysis."""
        return f"""Analyze the following user prompt and extract structured information.

User Prompt:
{user_prompt}

Extract the following information and return as JSON:

1. **domain**: The primary domain or field (e.g., "legal", "medical", "customer_support", "technical", "creative", "general")

2. **objectives**: A list of specific goals or tasks the user wants to accomplish (be concrete and actionable)

3. **complexity**: Rate as "simple", "moderate", or "complex" based on:
   - Simple: Single straightforward task, minimal context needed
   - Moderate: Multiple steps or some specialized knowledge required
   - Complex: Multi-faceted, requires deep expertise or careful handling

4. **constraints**: Any specific requirements, limitations, or rules mentioned (e.g., "must be under 100 words", "formal tone", "use legal terminology")

5. **context**: A brief summary of the background or situation (2-3 sentences)

6. **required_skills**: Skills or knowledge areas needed to complete this task well

Return ONLY valid JSON in this exact format:
{{
  "domain": "string",
  "objectives": ["string", "string"],
  "complexity": "simple|moderate|complex",
  "constraints": ["string", "string"],
  "context": "string",
  "required_skills": ["string", "string"]
}}"""
    
    def get_analysis_summary(self, analysis: TaskAnalysis) -> str:
        """
        Generate a human-readable summary of the task analysis.
        
        Args:
            analysis: TaskAnalysis object
            
        Returns:
            Formatted summary string
        """
        summary = f"""
Task Analysis Summary
{'=' * 50}

Domain: {analysis.domain}
Complexity: {analysis.complexity}

Objectives:
"""
        for i, obj in enumerate(analysis.objectives, 1):
            summary += f"  {i}. {obj}\n"
        
        if analysis.constraints:
            summary += "\nConstraints:\n"
            for constraint in analysis.constraints:
                summary += f"  - {constraint}\n"
        
        if analysis.required_skills:
            summary += "\nRequired Skills:\n"
            for skill in analysis.required_skills:
                summary += f"  - {skill}\n"
        
        summary += f"\nContext:\n{analysis.context}\n"
        summary += "=" * 50
        
        return summary
