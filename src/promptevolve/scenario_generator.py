"""
Scenario Generator - Creates test scenarios from task analysis.

This module converts task objectives into TestScenario objects that can be
used by the PromptEvolve system for prompt optimization.
"""

import json
import logging
from typing import List, Optional
from dataclasses import dataclass
from openai import OpenAI

from .task_analyzer import TaskAnalysis

logger = logging.getLogger(__name__)


@dataclass
class TestScenario:
    """A test scenario for prompt evolution."""
    description: str
    desired_output: str
    bad_output: str
    input_data: dict


class ScenarioGenerator:
    """
    Generates test scenarios from task analysis results.
    
    Creates specific test cases with desired and bad outputs to guide
    the prompt evolution process.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini"):
        """
        Initialize the scenario generator.
        
        Args:
            api_key: OpenAI API key (uses environment variable if not provided)
            model: Model to use for generation (default: gpt-5-mini)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized ScenarioGenerator with model: {model}")
    
    def generate_from_task(
        self, 
        task_analysis: TaskAnalysis, 
        num_scenarios: int = 3
    ) -> List[TestScenario]:
        """
        Generate test scenarios from task analysis.
        
        Args:
            task_analysis: TaskAnalysis object with extracted information
            num_scenarios: Number of scenarios to generate (default: 3)
            
        Returns:
            List of TestScenario objects
        """
        logger.info(f"Generating {num_scenarios} test scenarios...")
        
        # Create generation prompt
        generation_prompt = self._create_generation_prompt(task_analysis, num_scenarios)
        
        try:
            # Call LLM for scenario generation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating comprehensive test scenarios. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": generation_prompt
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            scenarios_data = result.get("scenarios", [])
            
            # Create TestScenario objects
            scenarios = []
            for scenario_data in scenarios_data:
                scenario = TestScenario(
                    description=scenario_data.get("description", ""),
                    desired_output=scenario_data.get("desired_output", ""),
                    bad_output=scenario_data.get("bad_output", ""),
                    input_data=scenario_data.get("input_data", {})
                )
                scenarios.append(scenario)
            
            logger.info(f"Generated {len(scenarios)} test scenarios successfully")
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating scenarios: {e}")
            # Return basic scenario on error
            return [
                TestScenario(
                    description="Default scenario",
                    desired_output="Clear, helpful response addressing user needs",
                    bad_output="Vague, unhelpful, or incorrect response",
                    input_data={"prompt": task_analysis.raw_prompt}
                )
            ]
    
    def _create_generation_prompt(self, task_analysis: TaskAnalysis, num_scenarios: int) -> str:
        """Create the prompt for scenario generation."""
        objectives_str = "\n".join(f"- {obj}" for obj in task_analysis.objectives)
        constraints_str = "\n".join(f"- {const}" for const in task_analysis.constraints) if task_analysis.constraints else "None specified"
        
        return f"""Create {num_scenarios} diverse test scenarios for prompt optimization.

Task Information:
- Domain: {task_analysis.domain}
- Complexity: {task_analysis.complexity}
- Context: {task_analysis.context}

Objectives:
{objectives_str}

Constraints:
{constraints_str}

Required Skills:
{', '.join(task_analysis.required_skills)}

For each scenario, create:
1. **description**: What aspect of the task this scenario tests
2. **desired_output**: What a good/ideal response should look like (be specific with examples)
3. **bad_output**: What a poor/incorrect response would look like (be specific)
4. **input_data**: Any specific input parameters for this scenario (include "query" or relevant fields)

Create scenarios that:
- Cover different aspects of the objectives
- Test edge cases and common situations
- Vary in difficulty within the task complexity
- Include specific, realistic examples
- Consider the constraints mentioned

Return ONLY valid JSON in this exact format:
{{
  "scenarios": [
    {{
      "description": "string describing what this tests",
      "desired_output": "detailed example of good output",
      "bad_output": "detailed example of bad output",
      "input_data": {{
        "query": "example user query",
        "additional_field": "value"
      }}
    }}
  ]
}}

Make scenarios realistic and actionable for the {task_analysis.domain} domain."""
    
    def scenarios_to_dict(self, scenarios: List[TestScenario]) -> List[dict]:
        """
        Convert TestScenario objects to dictionaries for JSON serialization.
        
        Args:
            scenarios: List of TestScenario objects
            
        Returns:
            List of dictionaries
        """
        return [
            {
                "description": s.description,
                "desired_output": s.desired_output,
                "bad_output": s.bad_output,
                "input_data": s.input_data
            }
            for s in scenarios
        ]
    
    def get_scenarios_summary(self, scenarios: List[TestScenario]) -> str:
        """
        Generate a human-readable summary of scenarios.
        
        Args:
            scenarios: List of TestScenario objects
            
        Returns:
            Formatted summary string
        """
        summary = f"\nGenerated Test Scenarios ({len(scenarios)})\n"
        summary += "=" * 50 + "\n"
        
        for i, scenario in enumerate(scenarios, 1):
            summary += f"\nScenario {i}: {scenario.description}\n"
            summary += f"Input: {scenario.input_data.get('query', 'N/A')}\n"
            summary += f"✓ Desired: {scenario.desired_output[:100]}...\n"
            summary += f"✗ Bad: {scenario.bad_output[:100]}...\n"
        
        summary += "\n" + "=" * 50
        return summary
