# PromptEvolve Project Structure

```
LLM-Architecture/
├── config/                     # Configuration files
│   ├── config.yaml            # Main configuration
│   └── .env.example           # API keys template
│
├── src/                       # Source code
│   └── promptevolve/          # Main Python package
│       ├── __init__.py        # Package initialization
│       ├── prompt_evolution.py # Core evolution engine
│       ├── taskdefiner.py     # Task definition system
│       ├── data_generator.py  # Test scenario generator
│       ├── llm_client.py      # LLM client wrapper
│       └── setup_promptevolve.py # Setup utilities
│
├── docs/                      # Documentation
│   ├── README.md             # Main documentation
│   ├── API_KEY_TEST_RESULTS.md # API testing results
│   └── SCENARIO_USAGE_GUIDE.md # Test scenario guide
│
├── examples/                  # Example scripts
│   ├── test_evolution_engine.py
│   └── test_simple.py
│
├── tests/                     # Unit tests (future)
│
├── data/                      # Data storage
│   ├── history/              # Evolution history
│   └── test_scenarios/       # Test scenario files
│
├── logs/                      # Log files
│   └── prompt_evolution.log
│
├── results/                   # Evolution results
│   └── evolution_result_*.json
│
├── prompts/                   # Prompt templates
│
├── run_evolution.py          # Main entry point
├── pyproject.toml            # Package configuration
└── uv.lock                   # Dependency lock file
```

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Configure API Keys
```bash
export OPENAI_API_KEY='your-key-here'
# or
export GEMINI_API_KEY='your-key-here'
```

### 3. Run Evolution
```bash
# From project root
python run_evolution.py

# Or using uv
uv run python run_evolution.py
```

### 4. Use as Python Package
```python
from promptevolve import PromptEvolution, TestScenario

system = PromptEvolution()
scenarios = [TestScenario(...)]
result = system.evolve_prompt(base_prompt="...", test_scenarios=scenarios)
```

## Directory Details

### `/config`
- **config.yaml**: Main configuration file containing API settings, model configs, evolution parameters, and domain configurations
- **.env.example**: Template for environment variables

### `/src/promptevolve`
- **prompt_evolution.py**: Core R-Zero evolution engine with all evolution logic
- **taskdefiner.py**: Task definition and requirements extraction
- **data_generator.py**: Synthetic test scenario generation
- **llm_client.py**: Unified LLM client wrapper

### `/docs`
All documentation files including guides, API references, and test results

### `/examples`
Example scripts showing how to use the system

### `/data`
- **history/**: Stores evolution history for learning
- **test_scenarios/**: Domain-specific test scenario JSON files

### `/logs`
System logs for debugging and monitoring

### `/results`
Evolution results with final prompts, metrics, and learnings

## Configuration

Edit `config/config.yaml` to customize:
- API provider (OpenAI or Gemini)
- Model selection
- Evolution parameters
- Domain-specific settings
- Logging configuration

## API Keys

**Recommended**: Use environment variables
```bash
export OPENAI_API_KEY='sk-...'
export GEMINI_API_KEY='AI...'
```

**Alternative**: Edit `config/config.yaml`
```yaml
api_keys:
  openai: "sk-..."
  gemini: "AI..."
```
