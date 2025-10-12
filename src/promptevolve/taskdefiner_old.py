import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from openai import OpenAI
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OutputType(Enum):
    """Enumeration of possible output types"""
    TEXT = "text"
    CODE = "code"
    DATA = "data"
    ANALYSIS = "analysis"
    DESIGN = "design"
    PLAN = "plan"
    DOCUMENTATION = "documentation"
    API = "api"
    UI_COMPONENT = "ui_component"
    OTHER = "other"


class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Complexity(Enum):
    """Task complexity levels"""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    HIGHLY_COMPLEX = 4


@dataclass
class UserProfile:
    """User profile containing preferences and history metadata"""
    user_id: str
    name: Optional[str]
    domain_expertise: List[str]
    preferred_output_formats: List[str]
    typical_complexity: str
    past_projects: List[str]
    constraints_history: List[str]
    communication_style: str
    technical_level: str

    def to_context_string(self) -> str:
        """Convert profile to context string for LLM"""
        return f"""
        User Profile:
        - Domain Expertise: {', '.join(self.domain_expertise)}
        - Preferred Formats: {', '.join(self.preferred_output_formats)}
        - Technical Level: {self.technical_level}
        - Typical Complexity: {self.typical_complexity}
        - Recent Projects: {', '.join(self.past_projects[-5:])}
        - Common Constraints: {', '.join(self.constraints_history[-3:])}
        """


class LLMTaskDefinitionSystem:
    """Main system for preparing task definitions using LLMs"""

    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo-preview"):
        """
        Initialize the system with OpenAI API

        Args:
            api_key: OpenAI API key (defaults to env variable)
            model: Model to use for processing
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.task_definitions = []
        self.chat_history_store = {}  # In production, use a proper database
        self.user_profiles = {}  # In production, use a proper database

    def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from storage"""
        # In production, this would load from a database
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        # Default profile for new users
        return UserProfile(
            user_id=user_id,
            name=None,
            domain_expertise=["general"],
            preferred_output_formats=["json", "markdown"],
            typical_complexity="moderate",
            past_projects=[],
            constraints_history=[],
            communication_style="professional",
            technical_level="intermediate"
        )

    def get_chat_history(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieve recent chat history for context"""
        if user_id not in self.chat_history_store:
            return []

        history = self.chat_history_store[user_id]
        return history[-limit:] if len(history) > limit else history

    def save_chat_message(self, user_id: str, role: str, content: str):
        """Save a chat message to history"""
        if user_id not in self.chat_history_store:
            self.chat_history_store[user_id] = []

        self.chat_history_store[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def extract_context_with_priority(self, raw_prompt: str, chat_history: List[Dict],
                                     user_profile: UserProfile) -> Dict[str, Any]:
        """
        Use LLM to extract and prioritize context from all available information
        """
        context_prompt = f"""
        Analyze the following information and extract relevant context for task definition.
        Prioritize information based on relevance to the current prompt.

        CURRENT PROMPT:
        {raw_prompt}

        USER PROFILE:
        {user_profile.to_context_string()}

        RECENT CHAT HISTORY:
        {json.dumps(chat_history[-5:], indent=2) if chat_history else "No previous history"}

        Extract and return a JSON object with:
        {{
            "primary_goal": "main objective extracted from prompt",
            "domain": "identified domain/field",
            "inferred_constraints": ["list of constraints from context"],
            "missing_information": ["critical missing pieces"],
            "assumptions": ["reasonable assumptions based on profile and history"],
            "priority_rationale": "why certain information was prioritized",
            "context_confidence": 0.0-1.0
        }}

        Focus on extracting actionable context without asking questions.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a task analysis expert. Extract context and prioritize information for task definition."},
                    {"role": "user", "content": context_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return {
                "primary_goal": "Process the given request",
                "domain": "general",
                "inferred_constraints": [],
                "missing_information": ["Unable to extract full context"],
                "assumptions": ["Standard processing required"],
                "priority_rationale": "Fallback due to extraction error",
                "context_confidence": 0.3
            }

    def generate_problem_statement(self, raw_prompt: str, context: Dict) -> Dict[str, Any]:
        """
        Use LLM to generate clear problem statement and objectives
        """
        problem_prompt = f"""
        Based on the following prompt and context, generate a clear problem statement and objectives.

        ORIGINAL PROMPT:
        {raw_prompt}

        EXTRACTED CONTEXT:
        {json.dumps(context, indent=2)}

        Generate a JSON object with:
        {{
            "problem_statement": "one clear, concise sentence defining the problem",
            "objectives": ["specific, measurable objectives"],
            "success_criteria": ["how success will be measured"],
            "scope": "clear boundaries of what's included/excluded"
        }}

        Make it actionable and unambiguous.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at defining clear, actionable problems from requirements."},
                    {"role": "user", "content": problem_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error generating problem statement: {e}")
            return {
                "problem_statement": f"Process request: {raw_prompt[:100]}",
                "objectives": ["Complete the requested task"],
                "success_criteria": ["Task completed successfully"],
                "scope": "As specified in the prompt"
            }

    def classify_task(self, raw_prompt: str, problem_def: Dict, context: Dict) -> Dict[str, Any]:
        """
        Use LLM to classify task metadata
        """
        classify_prompt = f"""
        Classify the following task based on the prompt and problem definition.

        PROMPT:
        {raw_prompt}

        PROBLEM DEFINITION:
        {json.dumps(problem_def, indent=2)}

        CONTEXT:
        {json.dumps(context, indent=2)}

        Return a JSON object with:
        {{
            "output_type": "one of: text, code, data, analysis, design, plan, documentation, api, ui_component, other",
            "priority": "one of: low, medium, high, critical",
            "complexity": "one of: simple, moderate, complex, highly_complex",
            "estimated_effort": "estimated time/effort required",
            "required_skills": ["list of skills needed"],
            "dependencies": ["list of dependencies"],
            "risk_level": "low, medium, high"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at task classification and project management."},
                    {"role": "user", "content": classify_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error classifying task: {e}")
            return {
                "output_type": "other",
                "priority": "medium",
                "complexity": "moderate",
                "estimated_effort": "unknown",
                "required_skills": [],
                "dependencies": [],
                "risk_level": "medium"
            }

    def validate_and_enhance(self, task_definition: Dict) -> Dict[str, Any]:
        """
        Use LLM to validate and enhance the task definition
        """
        validate_prompt = f"""
        Review and enhance the following task definition for completeness and clarity.

        TASK DEFINITION:
        {json.dumps(task_definition, indent=2)}

        Return a JSON object with:
        {{
            "validation_status": "valid, needs_revision, or invalid",
            "completeness_score": 0.0-1.0,
            "clarity_score": 0.0-1.0,
            "actionability_score": 0.0-1.0,
            "issues_found": ["list of issues if any"],
            "enhancements": ["suggested improvements"],
            "additional_considerations": ["important points to consider"],
            "ready_for_execution": true/false
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a quality assurance expert for task definitions."},
                    {"role": "user", "content": validate_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error validating task: {e}")
            return {
                "validation_status": "valid",
                "completeness_score": 0.7,
                "clarity_score": 0.7,
                "actionability_score": 0.7,
                "issues_found": [],
                "enhancements": [],
                "additional_considerations": [],
                "ready_for_execution": True
            }

    def prepare_task_definition(self, raw_prompt: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Main method to prepare a complete task definition

        Args:
            raw_prompt: The user's input prompt
            user_id: User identifier for context retrieval

        Returns:
            Complete task definition in JSON format
        """
        logger.info(f"Preparing task definition for user: {user_id}")

        # Save the prompt to chat history
        self.save_chat_message(user_id, "user", raw_prompt)

        # Step 1: Load user profile and chat history
        user_profile = self.load_user_profile(user_id)
        chat_history = self.get_chat_history(user_id)

        # Step 2: Extract and prioritize context
        logger.info("Extracting context with priority...")
        context = self.extract_context_with_priority(raw_prompt, chat_history, user_profile)

        # Step 3: Generate problem statement and objectives
        logger.info("Generating problem statement...")
        problem_def = self.generate_problem_statement(raw_prompt, context)

        # Step 4: Classify the task
        logger.info("Classifying task...")
        classification = self.classify_task(raw_prompt, problem_def, context)

        # Step 5: Compile full task definition
        task_id = f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(raw_prompt.encode()).hexdigest()[:8]}"

        task_definition = {
            "id": task_id,
            "created_at": datetime.now().isoformat(),
            "user_id": user_id,
            "status": "draft",
            "raw_prompt": raw_prompt,
            "problem_statement": problem_def["problem_statement"],
            "objectives": problem_def["objectives"],
            "success_criteria": problem_def["success_criteria"],
            "scope": problem_def["scope"],
            "context": {
                "primary_goal": context["primary_goal"],
                "domain": context["domain"],
                "constraints": context["inferred_constraints"],
                "missing_information": context["missing_information"],
                "assumptions": context["assumptions"],
                "priority_rationale": context["priority_rationale"],
                "confidence_score": context["context_confidence"]
            },
            "classification": {
                "output_type": classification["output_type"],
                "priority": classification["priority"],
                "complexity": classification["complexity"],
                "estimated_effort": classification["estimated_effort"],
                "required_skills": classification["required_skills"],
                "dependencies": classification["dependencies"],
                "risk_level": classification["risk_level"]
            },
            "metadata": {
                "version": "1.0",
                "model_used": self.model,
                "profile_based": True,
                "history_context_used": len(chat_history) > 0
            }
        }

        # Step 6: Validate and enhance
        logger.info("Validating and enhancing definition...")
        validation = self.validate_and_enhance(task_definition)

        task_definition["validation"] = validation
        task_definition["status"] = "finalized" if validation["ready_for_execution"] else "needs_review"

        # Store the definition
        self.task_definitions.append(task_definition)

        # Save to chat history
        self.save_chat_message(user_id, "assistant", f"Task definition created: {task_id}")

        logger.info(f"Task definition completed: {task_id}")

        return task_definition

    def save_to_file(self, task_definition: Dict, filename: str = None) -> str:
        """Save task definition to JSON file"""
        if not filename:
            filename = f"{task_definition['id']}.json"

        with open(filename, 'w') as f:
            json.dump(task_definition, f, indent=2)

        logger.info(f"Task definition saved to {filename}")
        return filename

    def batch_process(self, prompts: List[str], user_id: str = "default") -> List[Dict]:
        """Process multiple prompts in batch"""
        results = []
        for prompt in prompts:
            try:
                result = self.prepare_task_definition(prompt, user_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing prompt: {e}")
                results.append({
                    "error": str(e),
                    "raw_prompt": prompt,
                    "status": "failed"
                })
        return results


# Utility class for managing user profiles
class ProfileManager:
    """Manage and update user profiles based on interactions"""

    def __init__(self, llm_system: LLMTaskDefinitionSystem):
        self.llm_system = llm_system

    def update_profile_from_task(self, user_id: str, task_definition: Dict):
        """Update user profile based on completed task definitions"""
        profile = self.llm_system.load_user_profile(user_id)

        # Update domain expertise
        if task_definition["context"]["domain"] not in profile.domain_expertise:
            profile.domain_expertise.append(task_definition["context"]["domain"])

        # Update project history
        profile.past_projects.append(task_definition["problem_statement"][:100])

        # Update constraints history
        for constraint in task_definition["context"]["constraints"]:
            if constraint not in profile.constraints_history:
                profile.constraints_history.append(constraint)

        # Store updated profile
        self.llm_system.user_profiles[user_id] = profile

        logger.info(f"Profile updated for user: {user_id}")


# Example usage
def main():
    """Example usage of the LLM Task Definition System"""

    # Initialize the system
    system = LLMTaskDefinitionSystem(
        api_key="OPENAI_API_KEY",  # Set your API key as environment variable
        model="gpt-4-turbo-preview"  # or "gpt-3.5-turbo" for faster/cheaper processing
    )

    # Create a sample user profile
    sample_profile = UserProfile(
        user_id="user123",
        name="John Developer",
        domain_expertise=["software development", "web applications", "data analysis"],
        preferred_output_formats=["json", "python", "react"],
        typical_complexity="complex",
        past_projects=[
            "Build REST API for user management",
            "Create data visualization dashboard",
            "Optimize database queries"
        ],
        constraints_history=["must be scalable", "follow SOLID principles", "include unit tests"],
        communication_style="technical",
        technical_level="expert"
    )

    # Store the profile
    system.user_profiles["user123"] = sample_profile

    # Add some sample chat history
    system.chat_history_store["user123"] = [
        {"role": "user", "content": "I need to build a microservice for processing payments", "timestamp": "2024-01-01T10:00:00"},
        {"role": "assistant", "content": "I'll help you design a payment processing microservice", "timestamp": "2024-01-01T10:01:00"},
        {"role": "user", "content": "It should handle multiple payment providers and be PCI compliant", "timestamp": "2024-01-01T10:02:00"}
    ]

    # Process a new prompt
    prompt = """
    Create a real-time analytics dashboard that processes streaming data from IoT sensors,
    displays key metrics, and sends alerts when thresholds are exceeded.
    """

    # Generate task definition
    task_def = system.prepare_task_definition(prompt, user_id="user123")

    # Save to file
    filename = system.save_to_file(task_def)

    # Print the result
    print(json.dumps(task_def, indent=2))
    print(f"\nTask definition saved to: {filename}")

    # Example of batch processing
    batch_prompts = [
        "Implement user authentication with OAuth2",
        "Create automated testing pipeline for CI/CD",
        "Build recommendation engine using collaborative filtering"
    ]

    print("\nBatch processing multiple prompts...")
    batch_results = system.batch_process(batch_prompts, user_id="user123")

    for i, result in enumerate(batch_results, 1):
        print(f"\nTask {i}: {result.get('id', 'ERROR')}")
        print(f"Status: {result.get('status', 'failed')}")
        if 'problem_statement' in result:
            print(f"Problem: {result['problem_statement']}")


if __name__ == "__main__":
    main()

