# LLM-Architecture

An AI Engineering System designed to automate and optimize LLM interactions through intelligent prompt engineering and task definition.

---

# Table of Contents

1. [PromptEvolve: Self-Improving Prompt Engineering Agent](#promptevolve)
2. [Task Definer System](#task-definer-system)
3. [Integration Guide](#integration-guide)

---

<a name="promptevolve"></a>
# PromptEvolve: Self-Improving Prompt Engineering Agent

## 🎯 Overview

**PromptEvolve** is a robust, Python-based closed-loop system that automates the prompt engineering process using **R-Zero co-evolutionary principles**. It continuously tests, evaluates, reflects upon, and iteratively improves prompts to achieve reliable, high-performing results before deployment.

### Key Features

- ✅ **Automated Prompt Optimization** - CI/CD pipeline for prompts
- 🧬 **Evolutionary Algorithm** - Genetic algorithm-style selection and combination
- 🧠 **System 2 Thinking** - Self-debate mechanism (Proposer/Skeptic/Judge)
- 📊 **Comprehensive Metrics** - Success rate, quality, consistency, and efficiency scoring
- 🎯 **Domain Adaptability** - Built-in support for Legal, Ontology, and Admin domains
- 🧪 **Synthetic Data Generation** - Automated test case creation with human verification
- 🔒 **Production-Ready** - Full logging, error handling, and validation

---

## 🏗️ Architecture

The system implements the **R-Zero Framework** with four core components:

### 1. **Prompt Challenger** (Generation)
Generates prompt variations using multiple techniques:
- Chain-of-Thought reasoning
- Few-Shot examples (with counter-examples)
- Role specifications
- Step-by-step instructions
- Self-debate loop for quality assurance

### 2. **Prompt Solver** (Execution)
Executes prompt variants against standardized test scenarios using the production model.

### 3. **Prompt Verifier** (Evaluation)
Evaluates effectiveness with comprehensive metrics:
- Success Rate
- Quality Score
- Consistency Score
- Efficiency Score
- Domain-specific metrics

### 4. **Evolution Engine** (Selection/Iteration)
Manages the iterative improvement process:
- Genetic algorithm-based selection
- Prompt combination (crossover)
- Termination logic (thresholds, plateaus, iterations)

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- OpenAI API key

### Setup

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates .venv automatically)
uv sync

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Create necessary directories
mkdir -p logs data/synthetic data/history data/test_scenarios prompts results
```

### Running the Code

```bash
# Use uv run to execute scripts with dependencies
uv run python prompt_evolution.py

# Or generate data
uv run python data_generator.py

# Or run setup
uv run python setup_promptevolve.py
```

---

## 🚀 Quick Start

### Basic Usage

```python
from prompt_evolution import PromptEvolution, TestScenario

# Initialize the system
system = PromptEvolution(config_path="config.yaml")

# Create test scenarios
test_scenarios = [
    TestScenario(
        input_message="What is the capital of France?",
        existing_memories="",
        desired_output="The capital of France is Paris.",
        bad_output="The capital is London.",
        metadata={"difficulty": "easy"}
    ),
    # Add more scenarios...
]

# Base prompt to optimize
base_prompt = "You are a helpful assistant. Answer questions accurately and concisely."

# Run evolution
final_state = system.evolve_prompt(
    base_prompt=base_prompt,
    test_scenarios=test_scenarios
)

# View results
print(f"Evolution completed after {final_state.generation} generations")
print(f"Final prompt:\n{final_state.current_prompt}")
print(f"Success rate: {final_state.results.success_rate:.1%}")
```

### Domain-Specific Optimization

```python
from prompt_evolution import DomainPromptOptimizer

# Initialize domain optimizer
optimizer = DomainPromptOptimizer(config_path="config.yaml")

# Optimize for legal domain
final_state = optimizer.optimize_for_domain(
    domain_name="legal",
    test_scenarios=legal_test_scenarios
)
```

### Synthetic Data Generation

```python
from data_generator import SyntheticDataGenerator, GenerationRequest

# Initialize generator
generator = SyntheticDataGenerator(config_path="config.yaml")

# Generate test scenarios
request = GenerationRequest(
    domain="legal",
    count=20,
    difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
    include_edge_cases=True
)

dataset = generator.generate_dataset(request)

# Save for human verification
filepath = generator.save_dataset(dataset)
print(f"Dataset saved to: {filepath}")
```

---

## ⚙️ Configuration

All settings are managed in `config.yaml`:

### Model Configuration
```yaml
models:
  prompt_writer:
    name: "gpt-4-turbo-preview"
    temperature: 0.7
  
  prompt_solver:
    name: "gpt-3.5-turbo"  # Your production model
    temperature: 0.3
  
  prompt_verifier:
    name: "gpt-4-turbo-preview"
    temperature: 0.2
```

### Evolution Settings
```yaml
evolution:
  max_iterations: 10
  selection_top_percentage: 0.30
  enable_crossover: true
  population_size: 5
```

### Performance Thresholds
```yaml
thresholds:
  success_rate: 0.85
  quality_score: 0.80
  consistency_score: 0.75
```

---

## 📊 Metrics & Evaluation

### Composite Scoring Formula

```
Composite Score = (Success Rate × 0.4) + 
                  (Quality Score × 0.3) + 
                  (Consistency Score × 0.2) + 
                  (Efficiency Score × 0.1)
```

---

## 📁 Project Structure

```
LLM-Architecture/
├── config.yaml                  # Main configuration file
├── prompt_evolution.py          # Core evolution system (1,100+ lines)
├── data_generator.py            # Synthetic data generation (600+ lines)
├── test_evolution_engine.py     # Unit tests (600+ lines)
├── pyproject.toml              # Python package configuration (uv)
├── README.md                    # This file
├── taskdefiner.py              # Task definition system
├── logs/                        # Log files
├── data/
│   ├── synthetic/              # Generated test scenarios
│   ├── history/                # Evolution history
│   └── test_scenarios/         # Test scenario datasets
├── prompts/                     # Generated prompts
└── results/                     # Evolution results
```

---

<a name="task-definer-system"></a>
# Task Definer System

The **Task Definer System** is a Python-based framework that leverages Large Language Models (LLMs) to analyze user prompts, extract context, define problems, classify tasks, validate definitions, and generate actionable task specifications.

## 🏷️ Core Components

### Enumerations
- **`OutputType`**: TEXT, CODE, DATA, ANALYSIS, DESIGN, PLAN, DOCUMENTATION, API, UI_COMPONENT, OTHER
- **`Priority`**: LOW, MEDIUM, HIGH, CRITICAL
- **`Complexity`**: SIMPLE, MODERATE, COMPLEX, HIGHLY_COMPLEX

### UserProfile (Dataclass)
Represents user preferences and history:
- `user_id`, `name`, `domain_expertise`
- `preferred_output_formats`, `typical_complexity`
- `past_projects`, `constraints_history`
- `communication_style`, `technical_level`

### LLMTaskDefinitionSystem
Main system for task definition with methods:
- `extract_context_with_priority()` - Analyzes prompt, chat history, and profile
- `generate_problem_statement()` - Creates problem statement, objectives, success criteria
- `classify_task()` - Classifies task metadata (type, priority, complexity, skills)
- `validate_and_enhance()` - Validates completeness and execution readiness
- `prepare_task_definition()` - End-to-end task processing pipeline
- `batch_process()` - Processes multiple prompts at once

## Example Usage

```python
from taskdefiner import LLMTaskDefinitionSystem, UserProfile

# Initialize system
system = LLMTaskDefinitionSystem(api_key="YOUR_API_KEY")

# Create user profile
profile = UserProfile(
    user_id="user123",
    name="John Developer",
    domain_expertise=["software development", "web apps"],
    preferred_output_formats=["json", "python"],
    technical_level="expert"
)
system.user_profiles["user123"] = profile

# Process a task
prompt = "Create a real-time analytics dashboard for IoT sensors"
task_def = system.prepare_task_definition(prompt, "user123")

# Save to file
filename = system.save_to_file(task_def)
```

---

<a name="integration-guide"></a>
# Integration Guide: PromptEvolve + Task Definer

## Sequential Pipeline Pattern

```python
from taskdefiner import LLMTaskDefinitionSystem
from prompt_evolution import PromptEvolution, TestScenario

# Step 1: Define the task
task_system = LLMTaskDefinitionSystem(api_key="YOUR_API_KEY")
task_def = task_system.prepare_task_definition(
    raw_prompt="Create analytics dashboard for IoT sensors",
    user_id="user123"
)

# Step 2: Extract base prompt from task definition
base_prompt = f"""You are an expert {task_def['classification']['task_type']} specialist.

Problem: {task_def['problem_statement']}

Objectives:
{chr(10).join([f"- {obj}" for obj in task_def['objectives']])}

Success Criteria:
{chr(10).join([f"- {sc}" for sc in task_def['success_criteria']])}

Provide a comprehensive solution that meets all requirements."""

# Step 3: Create test scenarios
test_scenarios = [
    TestScenario(
        input_message=f"How would you {obj}?",
        existing_memories="",
        desired_output=f"To {obj}, I would...",
        bad_output="I cannot help with that.",
        metadata={"objective": obj}
    )
    for obj in task_def['objectives'][:5]
]

# Step 4: Optimize the prompt
evolution_system = PromptEvolution(config_path="config.yaml")
final_state = evolution_system.evolve_prompt(
    base_prompt=base_prompt,
    test_scenarios=test_scenarios
)

print(f"Success rate: {final_state.results.success_rate:.1%}")
```

## Domain-Aware Integration

```python
from taskdefiner import LLMTaskDefinitionSystem
from prompt_evolution import DomainPromptOptimizer

# Map task types to domains
TASK_TYPE_TO_DOMAIN = {
    "legal_analysis": "legal",
    "knowledge_extraction": "ontology",
    "administrative_task": "admin",
}

def optimize_task_prompt(raw_prompt: str, user_id: str = "default"):
    """Complete pipeline: task definition → prompt optimization."""
    # Define the task
    task_system = LLMTaskDefinitionSystem()
    task_def = task_system.prepare_task_definition(raw_prompt, user_id)
    
    # Determine domain
    task_type = task_def['classification']['task_type']
    domain = TASK_TYPE_TO_DOMAIN.get(task_type, "admin")
    
    # Generate test scenarios from task
    test_scenarios = generate_test_scenarios_from_task(task_def)
    
    # Optimize for domain
    optimizer = DomainPromptOptimizer(config_path="config.yaml")
    final_state = optimizer.optimize_for_domain(
        domain_name=domain,
        test_scenarios=test_scenarios
    )
    
    return final_state
```

---

## 🧪 Testing

```bash
# Run all tests
python test_evolution_engine.py

# Or use pytest
pytest test_evolution_engine.py -v

# With coverage
pytest test_evolution_engine.py --cov=prompt_evolution --cov-report=html
```

---

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Run `uv sync`
2. **API Rate Limits**: Adjust `safety.rate_limit` in config.yaml
3. **Memory Issues**: Reduce `evolution.population_size`
4. **Low Success Rates**: Review test scenarios, adjust thresholds, increase iterations

---

## 📄 License

This project is part of the LLM-Architecture repository.

---

## 🙏 Acknowledgments

- Built on **R-Zero co-evolutionary principles**
- Inspired by genetic algorithms and System 2 thinking

---

**Ready to evolve your prompts? Start optimizing! 🚀**
