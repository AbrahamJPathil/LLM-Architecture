"""
Prompt Enhancer - Converts vague user input into complete, structured system prompts.

This module takes analyzed task information and generates a well-structured,
complete initial prompt ready for evolution.
"""

import json
import logging
from typing import Optional
from openai import OpenAI

from .task_analyzer import TaskAnalysis

logger = logging.getLogger(__name__)


class PromptEnhancer:
    """
    Enhances vague user input into complete, structured system prompts.
    
    Takes TaskAnalysis results and generates a production-ready initial prompt
    by filling gaps, adding domain knowledge, and structuring properly.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini"):
        """
        Initialize the prompt enhancer.
        
        Args:
            api_key: OpenAI API key (uses environment variable if not provided)
            model: Model to use for enhancement (default: gpt-5-mini)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized PromptEnhancer with model: {model}")
    
    def enhance_prompt(self, task_analysis: TaskAnalysis) -> str:
        """
        Generate an enhanced, complete system prompt from task analysis.
        
        Args:
            task_analysis: TaskAnalysis object with extracted information
            
        Returns:
            Complete, structured system prompt ready for evolution
        """
        logger.info("Enhancing prompt based on task analysis...")
        
        # Create enhancement prompt
        enhancement_prompt = self._create_enhancement_prompt(task_analysis)
        
        try:
            # Call LLM for enhancement
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert prompt engineer who creates comprehensive, production-ready system prompts for AI assistants."
                    },
                    {
                        "role": "user",
                        "content": enhancement_prompt
                    }
                ]
            )
            
            enhanced_prompt = response.choices[0].message.content.strip()
            logger.info(f"Enhancement complete - generated {len(enhanced_prompt)} characters")
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error during prompt enhancement: {e}")
            # Return basic enhanced version on error
            return self._create_fallback_prompt(task_analysis)
    
    def _create_enhancement_prompt(self, task_analysis: TaskAnalysis) -> str:
        """Create the prompt for LLM enhancement."""
        objectives_str = "\n".join(f"  {i}. {obj}" for i, obj in enumerate(task_analysis.objectives, 1))
        constraints_str = "\n".join(f"  - {const}" for const in task_analysis.constraints) if task_analysis.constraints else "  None specified"
        skills_str = ", ".join(task_analysis.required_skills)
        
        return f"""You are creating a comprehensive system prompt for an AI assistant based on a user's vague request.

ORIGINAL USER INPUT:
"{task_analysis.raw_prompt}"

ANALYSIS RESULTS:
- Domain: {task_analysis.domain}
- Complexity: {task_analysis.complexity}
- Context: {task_analysis.context}

OBJECTIVES:
{objectives_str}

CONSTRAINTS:
{constraints_str}

REQUIRED SKILLS:
{skills_str}

YOUR TASK:
Create a complete, production-ready system prompt that:

1. **Defines the Role Clearly**
   - Give the AI a specific professional identity
   - Explain its core purpose and responsibilities

2. **Lists Specific Objectives**
   - Turn the objectives into actionable instructions
   - Be concrete and measurable
   - Use "you will" or "your role is to" language

3. **Defines Success Criteria**
   - How will performance be measured?
   - What constitutes a good vs bad response?

4. **Specifies Constraints & Guidelines**
   - What must the AI never do?
   - What tone/style should it use?
   - Any compliance or safety requirements?

5. **Provides Response Structure** (if applicable)
   - Step-by-step approach
   - Format guidelines
   - Examples of good responses

6. **Handles Edge Cases**
   - What to do when uncertain
   - When to ask for clarification
   - When to escalate or refuse

DOMAIN-SPECIFIC ENHANCEMENTS:
Based on the {task_analysis.domain} domain, add relevant:
- Industry best practices
- Common scenarios the AI will face
- Domain-specific terminology to use/avoid
- Typical user expectations

OUTPUT FORMAT:
Write a clear, structured system prompt (200-400 words) that could be deployed immediately.
Use professional but accessible language.
Structure with clear sections and bullet points.
Make it actionable and unambiguous.

DO NOT:
- Include meta-commentary ("Here's the prompt...")
- Use placeholders like [Company Name] - be specific based on context
- Add assumptions not grounded in the analysis
- Make it overly verbose - be concise but complete

Generate the enhanced system prompt now:"""
    
    def _create_fallback_prompt(self, task_analysis: TaskAnalysis) -> str:
        """Create a basic enhanced prompt as fallback."""
        objectives_list = "\n".join(f"- {obj}" for obj in task_analysis.objectives)
        constraints_list = "\n".join(f"- {const}" for const in task_analysis.constraints) if task_analysis.constraints else "- No specific constraints"
        
        return f"""You are a professional {task_analysis.domain} assistant.

Your primary objectives are:
{objectives_list}

Constraints and guidelines:
{constraints_list}

When responding:
- Maintain a professional and helpful tone
- Provide clear, accurate information
- Ask for clarification when needed
- Follow industry best practices

Your goal is to assist users effectively while adhering to the guidelines above."""
    
    def get_enhancement_summary(self, original: str, enhanced: str) -> str:
        """
        Generate a summary comparing original to enhanced prompt.
        
        Args:
            original: Original user input
            enhanced: Enhanced system prompt
            
        Returns:
            Formatted comparison summary
        """
        summary = f"""
Prompt Enhancement Summary
{'=' * 60}

ORIGINAL INPUT ({len(original)} chars):
{original}

ENHANCED PROMPT ({len(enhanced)} chars):
{enhanced}

IMPROVEMENTS:
- Length increased by {len(enhanced) - len(original)} characters
- Added structure and clarity
- Included specific objectives and constraints
- Ready for production use
{'=' * 60}
"""
        return summary
