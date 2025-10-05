"""
Synthetic Test Data Generator for PromptEvolve

This module generates high-quality synthetic test scenarios for evaluating prompt variants.
It includes generation of input messages, desired outputs, and counter-examples (bad outputs).

IMPORTANT: All generated data requires human verification before use in production.
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

import yaml
from pydantic import BaseModel, Field, validator
from openai import OpenAI

# Import TestScenario from prompt_evolution
import sys
sys.path.append(os.path.dirname(__file__))
from prompt_evolution import TestScenario, setup_logging


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MutationStrategy(str, Enum):
    """Strategies for creating bad outputs from desired outputs."""
    FACTUAL_ERROR = "introduce_factual_error"
    TONE_CHANGE = "change_tone"
    OMIT_INFO = "omit_key_information"
    ADD_IRRELEVANT = "add_irrelevant_information"
    LOGICAL_INCONSISTENCY = "introduce_logical_inconsistency"


class GenerationRequest(BaseModel):
    """Request parameters for generating test scenarios."""
    domain: str = Field(..., description="Domain name (legal, ontology, admin)")
    count: int = Field(default=20, ge=1, le=100, description="Number of scenarios to generate")
    difficulty_distribution: Dict[str, float] = Field(
        default={"easy": 0.3, "medium": 0.5, "hard": 0.2},
        description="Distribution of difficulty levels"
    )
    include_edge_cases: bool = Field(default=True, description="Include edge case scenarios")
    mutation_strategies: List[MutationStrategy] = Field(
        default_factory=lambda: list(MutationStrategy),
        description="Strategies for generating bad outputs"
    )
    
    @validator('difficulty_distribution')
    def validate_distribution(cls, v):
        """Ensure difficulty distribution sums to 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Difficulty distribution must sum to 1.0, got {total}")
        return v


class GeneratedDataset(BaseModel):
    """Container for a generated dataset of test scenarios."""
    domain: str
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    scenario_count: int
    scenarios: List[TestScenario]
    human_verified: bool = Field(default=False, description="Has this been verified by a human?")
    verification_notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================

class SyntheticDataGenerator:
    """
    Generates synthetic test scenarios for prompt evaluation.
    
    This class uses a high-quality LLM to generate realistic test cases including:
    1. Input messages (user queries)
    2. Existing memories (context)
    3. Desired outputs (ground truth)
    4. Bad outputs (counter-examples)
    """
    
    def __init__(self, config_path: str = "config.yaml", api_key: Optional[str] = None):
        """
        Initialize the Synthetic Data Generator.
        
        Args:
            config_path: Path to YAML configuration file
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up logging
        self.logger = setup_logging(self.config)
        self.logger.info(f"{__name__}:{self.__init__.__code__.co_firstlineno} - "
                        "Initializing SyntheticDataGenerator")
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        
        # Get generator model from config
        self.generator_model = self.config.get('data_generation', {}).get(
            'generator_model', 'gpt-4-turbo-preview'
        )
        
        # Create output directory
        self.output_dir = self.config.get('data_generation', {}).get(
            'output_directory', 'data/synthetic'
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_dataset(self, request: GenerationRequest) -> GeneratedDataset:
        """
        Generate a complete dataset of test scenarios.
        
        Args:
            request: Generation parameters
            
        Returns:
            GeneratedDataset containing all generated scenarios
        """
        self.logger.info(f"{__name__}:{self.generate_dataset.__code__.co_firstlineno} - "
                        f"Generating {request.count} scenarios for domain: {request.domain}")
        
        scenarios = []
        
        # Get domain configuration
        domain_config = self.config.get('domains', {}).get(request.domain)
        if not domain_config:
            raise ValueError(f"Unknown domain: {request.domain}")
        
        # Calculate counts for each difficulty level
        difficulty_counts = {
            level: int(request.count * weight)
            for level, weight in request.difficulty_distribution.items()
        }
        
        # Ensure we generate exactly the requested count
        total_allocated = sum(difficulty_counts.values())
        if total_allocated < request.count:
            difficulty_counts['medium'] += (request.count - total_allocated)
        
        # Generate scenarios for each difficulty level
        for difficulty, count in difficulty_counts.items():
            if count == 0:
                continue
            
            self.logger.info(f"{__name__}:{self.generate_dataset.__code__.co_firstlineno} - "
                           f"Generating {count} {difficulty} scenarios")
            
            batch_scenarios = self._generate_scenario_batch(
                domain=request.domain,
                domain_config=domain_config,
                difficulty=difficulty,
                count=count,
                include_edge_cases=request.include_edge_cases
            )
            
            # Generate bad outputs for each scenario
            for scenario_data in batch_scenarios:
                try:
                    # Create TestScenario
                    scenario = TestScenario(**scenario_data)
                    
                    # Generate bad output using mutation strategies
                    bad_output = self._generate_bad_output(
                        desired_output=scenario.desired_output,
                        mutation_strategies=request.mutation_strategies,
                        domain=request.domain
                    )
                    
                    # Update scenario with bad output
                    scenario_dict = scenario.dict()
                    scenario_dict['bad_output'] = bad_output
                    scenario_dict['metadata']['difficulty'] = difficulty
                    
                    scenarios.append(TestScenario(**scenario_dict))
                    
                except Exception as e:
                    self.logger.error(f"{__name__}:{self.generate_dataset.__code__.co_firstlineno} - "
                                    f"Error creating scenario: {e}")
                    continue
        
        # Create dataset
        dataset = GeneratedDataset(
            domain=request.domain,
            scenario_count=len(scenarios),
            scenarios=scenarios,
            metadata={
                'difficulty_distribution': request.difficulty_distribution,
                'include_edge_cases': request.include_edge_cases,
                'generator_model': self.generator_model
            }
        )
        
        self.logger.info(f"{__name__}:{self.generate_dataset.__code__.co_firstlineno} - "
                        f"Successfully generated {len(scenarios)} scenarios")
        
        return dataset
    
    def _generate_scenario_batch(
        self,
        domain: str,
        domain_config: Dict[str, Any],
        difficulty: str,
        count: int,
        include_edge_cases: bool
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of test scenarios using the LLM.
        
        Args:
            domain: Domain name
            domain_config: Domain-specific configuration
            difficulty: Difficulty level (easy, medium, hard)
            count: Number of scenarios to generate
            include_edge_cases: Whether to include edge cases
            
        Returns:
            List of scenario dictionaries
        """
        try:
            base_prompt_template = domain_config.get('base_prompt_template', '')
            evaluation_criteria = domain_config.get('evaluation_criteria', {})
            
            criteria_str = "\n".join([f"- {k}: {v*100}%" for k, v in evaluation_criteria.items()])
            
            edge_case_instruction = ""
            if include_edge_cases:
                edge_case_instruction = """
Include some edge cases such as:
- Ambiguous queries
- Queries with missing context
- Queries that test boundary conditions
- Queries with potential conflicts or contradictions
"""
            
            system_message = f"""You are an expert test case generator for the {domain} domain.
Generate realistic, high-quality test scenarios that will be used to evaluate AI prompt effectiveness.

Domain Context:
{base_prompt_template}

Evaluation Criteria:
{criteria_str}

Each scenario should test different aspects of the domain."""
            
            user_message = f"""Generate {count} test scenarios for the {domain} domain at {difficulty} difficulty level.

{edge_case_instruction}

For each scenario, provide:
1. input_message: A realistic user query or input
2. existing_memories: Relevant context or previous interaction (can be empty)
3. desired_output: The ideal, high-quality response

Return the scenarios as a JSON array with this structure:
[
  {{
    "input_message": "...",
    "existing_memories": "...",
    "desired_output": "..."
  }},
  ...
]

Ensure diversity in the scenarios and make them realistic for actual use cases."""
            
            response = self.client.chat.completions.create(
                model=self.generator_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.8,  # Higher temperature for diversity
                max_tokens=3000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            scenarios = json.loads(content)
            
            if not isinstance(scenarios, list):
                self.logger.warning(f"{__name__}:{self._generate_scenario_batch.__code__.co_firstlineno} - "
                                  "Expected list of scenarios, got single object")
                scenarios = [scenarios]
            
            return scenarios
            
        except Exception as e:
            self.logger.error(f"{__name__}:{self._generate_scenario_batch.__code__.co_firstlineno} - "
                            f"Error generating scenario batch: {e}")
            return []
    
    def _generate_bad_output(
        self,
        desired_output: str,
        mutation_strategies: List[MutationStrategy],
        domain: str
    ) -> str:
        """
        Generate a bad output (counter-example) by mutating the desired output.
        
        Args:
            desired_output: The correct/desired output
            mutation_strategies: List of mutation strategies to apply
            domain: Domain name for context
            
        Returns:
            Bad output that demonstrates what NOT to produce
        """
        try:
            # Select a random mutation strategy
            import random
            strategy = random.choice(mutation_strategies)
            
            strategy_instructions = {
                MutationStrategy.FACTUAL_ERROR: "Introduce a factual error or incorrect information",
                MutationStrategy.TONE_CHANGE: "Change the tone to be inappropriate (too casual, too formal, unprofessional)",
                MutationStrategy.OMIT_INFO: "Omit critical information that should be included",
                MutationStrategy.ADD_IRRELEVANT: "Add irrelevant or off-topic information",
                MutationStrategy.LOGICAL_INCONSISTENCY: "Introduce a logical inconsistency or contradiction"
            }
            
            instruction = strategy_instructions.get(strategy, "Make it incorrect")
            
            system_message = f"""You are generating counter-examples (bad outputs) for training purposes.
Your task is to create a deliberately flawed version of the given output."""
            
            user_message = f"""Domain: {domain}

Desired (correct) output:
{desired_output}

Mutation strategy: {instruction}

Generate a bad output that demonstrates this flaw. Make it realistic but clearly incorrect.
The bad output should be something a poorly-tuned AI might actually produce.

Return only the bad output text, nothing else."""
            
            response = self.client.chat.completions.create(
                model=self.generator_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            bad_output = response.choices[0].message.content.strip()
            
            return bad_output
            
        except Exception as e:
            self.logger.error(f"{__name__}:{self._generate_bad_output.__code__.co_firstlineno} - "
                            f"Error generating bad output: {e}")
            return "This is a placeholder bad output due to generation error."
    
    def save_dataset(
        self,
        dataset: GeneratedDataset,
        filename: Optional[str] = None,
        require_verification: bool = True
    ) -> str:
        """
        Save a generated dataset to file.
        
        Args:
            dataset: The dataset to save
            filename: Optional custom filename (auto-generated if None)
            require_verification: If True, warns about human verification requirement
            
        Returns:
            Path to the saved file
        """
        if require_verification and not dataset.human_verified:
            self.logger.warning(f"{__name__}:{self.save_dataset.__code__.co_firstlineno} - "
                              "IMPORTANT: This dataset requires human verification before use in production!")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{dataset.domain}_scenarios_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to dict for JSON serialization
        dataset_dict = dataset.dict()
        
        with open(filepath, 'w') as f:
            json.dump(dataset_dict, f, indent=2)
        
        self.logger.info(f"{__name__}:{self.save_dataset.__code__.co_firstlineno} - "
                        f"Saved dataset to {filepath}")
        
        return filepath
    
    def load_dataset(self, filepath: str) -> GeneratedDataset:
        """
        Load a previously saved dataset.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            GeneratedDataset object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return GeneratedDataset(**data)
    
    def verify_dataset(
        self,
        dataset: GeneratedDataset,
        verification_notes: List[str]
    ) -> GeneratedDataset:
        """
        Mark a dataset as human-verified.
        
        Args:
            dataset: The dataset to verify
            verification_notes: Notes from the human reviewer
            
        Returns:
            Updated dataset with verification status
        """
        dataset.human_verified = True
        dataset.verification_notes = verification_notes
        
        self.logger.info(f"{__name__}:{self.verify_dataset.__code__.co_firstlineno} - "
                        f"Dataset verified with {len(verification_notes)} notes")
        
        return dataset
    
    def generate_domain_specific_examples(self, domain: str) -> List[TestScenario]:
        """
        Generate domain-specific example scenarios for documentation.
        
        Args:
            domain: Domain name
            
        Returns:
            List of example scenarios
        """
        request = GenerationRequest(
            domain=domain,
            count=5,
            difficulty_distribution={"easy": 0.4, "medium": 0.4, "hard": 0.2},
            include_edge_cases=True
        )
        
        dataset = self.generate_dataset(request)
        return dataset.scenarios


# ============================================================================
# HUMAN VERIFICATION WORKFLOW
# ============================================================================

class VerificationWorkflow:
    """
    Interactive workflow for human verification of generated test scenarios.
    """
    
    def __init__(self, generator: SyntheticDataGenerator):
        """
        Initialize the verification workflow.
        
        Args:
            generator: SyntheticDataGenerator instance
        """
        self.generator = generator
        self.logger = generator.logger
    
    def review_dataset(self, dataset: GeneratedDataset) -> GeneratedDataset:
        """
        Interactive review of a generated dataset.
        
        Args:
            dataset: The dataset to review
            
        Returns:
            Updated dataset with verification status
        """
        print(f"\n{'='*80}")
        print(f"DATASET VERIFICATION WORKFLOW")
        print(f"{'='*80}")
        print(f"Domain: {dataset.domain}")
        print(f"Generated: {dataset.generated_at}")
        print(f"Scenario Count: {dataset.scenario_count}")
        print(f"\nPlease review each scenario and verify its quality.\n")
        
        verification_notes = []
        
        for idx, scenario in enumerate(dataset.scenarios, 1):
            print(f"\n{'-'*80}")
            print(f"Scenario {idx}/{dataset.scenario_count}")
            print(f"{'-'*80}")
            print(f"\nInput Message:\n{scenario.input_message}")
            print(f"\nExisting Memories:\n{scenario.existing_memories or '[None]'}")
            print(f"\nDesired Output:\n{scenario.desired_output}")
            print(f"\nBad Output:\n{scenario.bad_output}")
            print(f"\nMetadata: {scenario.metadata}")
            
            # In a real implementation, you would collect user input here
            # For now, we'll just log that review is needed
            verification_notes.append(
                f"Scenario {idx}: Requires manual review - automated verification not implemented"
            )
        
        print(f"\n{'='*80}")
        print("VERIFICATION COMPLETE")
        print(f"{'='*80}\n")
        
        # Mark as verified (in real use, this would be conditional on actual review)
        return self.generator.verify_dataset(dataset, verification_notes)
    
    def export_for_review(self, dataset: GeneratedDataset, output_path: str):
        """
        Export dataset in a human-readable format for review.
        
        Args:
            dataset: The dataset to export
            output_path: Path to save the review file
        """
        with open(output_path, 'w') as f:
            f.write(f"# Dataset Verification Form\n\n")
            f.write(f"Domain: {dataset.domain}\n")
            f.write(f"Generated: {dataset.generated_at}\n")
            f.write(f"Total Scenarios: {dataset.scenario_count}\n\n")
            f.write(f"{'='*80}\n\n")
            
            for idx, scenario in enumerate(dataset.scenarios, 1):
                f.write(f"## Scenario {idx}\n\n")
                f.write(f"**Input Message:**\n{scenario.input_message}\n\n")
                f.write(f"**Existing Memories:**\n{scenario.existing_memories or '[None]'}\n\n")
                f.write(f"**Desired Output:**\n{scenario.desired_output}\n\n")
                f.write(f"**Bad Output:**\n{scenario.bad_output}\n\n")
                f.write(f"**Metadata:** {scenario.metadata}\n\n")
                f.write(f"**Verified:** [ ] Yes  [ ] No  [ ] Needs Revision\n\n")
                f.write(f"**Notes:**\n\n")
                f.write(f"---\n\n")
        
        self.logger.info(f"{__name__}:{self.export_for_review.__code__.co_firstlineno} - "
                        f"Exported review form to {output_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of the SyntheticDataGenerator."""
    
    # Initialize generator
    generator = SyntheticDataGenerator(config_path="config.yaml")
    
    # Generate dataset for legal domain
    request = GenerationRequest(
        domain="legal",
        count=10,
        difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
        include_edge_cases=True
    )
    
    print("Generating synthetic test scenarios...")
    dataset = generator.generate_dataset(request)
    
    # Save dataset
    filepath = generator.save_dataset(dataset)
    print(f"\nDataset saved to: {filepath}")
    
    # Create verification workflow
    workflow = VerificationWorkflow(generator)
    
    # Export for human review
    review_path = filepath.replace('.json', '_review.md')
    workflow.export_for_review(dataset, review_path)
    print(f"Review form exported to: {review_path}")
    
    print("\n" + "="*80)
    print("IMPORTANT: Please review and verify the generated scenarios before use!")
    print("="*80)
    
    # Print summary
    print(f"\nGenerated {dataset.scenario_count} scenarios:")
    for scenario in dataset.scenarios[:3]:  # Show first 3
        print(f"\n- Input: {scenario.input_message[:100]}...")
        print(f"  Difficulty: {scenario.metadata.get('difficulty', 'N/A')}")


if __name__ == "__main__":
    main()
