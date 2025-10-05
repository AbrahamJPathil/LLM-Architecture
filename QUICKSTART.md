# Quick Start Guide 🚀

## Installation & Setup

### 1. Navigate to Project
```bash
cd /home/sinan/Documents/proj/LLM-Architecture
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Set API Key
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

Or for Gemini:
```bash
export GEMINI_API_KEY='your-gemini-api-key-here'
```

### 4. Configure Provider (Optional)
Edit `config/config.yaml`:
```yaml
api_provider: "openai"  # or "gemini"
```

### 5. Run Evolution
```bash
uv run python run_evolution.py
```

## Usage Examples

### Basic Usage
```python
import sys
sys.path.insert(0, 'src')

from promptevolve import PromptEvolution, TestScenario

# Initialize
system = PromptEvolution()

# Define test scenarios
scenarios = [
    TestScenario(
        input_message="What is the capital of France?",
        desired_output="The capital of France is Paris.",
        bad_output="The capital is London.",
        existing_memories="",
        metadata={"difficulty": "easy"}
    )
]

# Run evolution
result = system.evolve_prompt(
    base_prompt="You are a helpful assistant.",
    test_scenarios=scenarios
)

print(f"Final prompt: {result.current_prompt}")
print(f"Success rate: {result.results.success_rate:.1%}")
```

### Using Custom Config
```python
from promptevolve import PromptEvolution

system = PromptEvolution(config_path="config/config.yaml")
```

### Domain-Specific Optimization
```python
from promptevolve.prompt_evolution import DomainPromptOptimizer

optimizer = DomainPromptOptimizer()
result = optimizer.optimize_for_domain(
    domain_name="legal",
    test_scenarios=legal_scenarios
)
```

## Project Structure

```
LLM-Architecture/
├── src/promptevolve/      # Source code (import from here)
├── config/                # Configuration files
├── docs/                  # Documentation
├── examples/              # Example scripts
├── data/                  # Test scenarios & history
├── logs/                  # System logs
├── results/               # Evolution results
└── run_evolution.py       # Main entry point
```

## Common Tasks

### View Results
```bash
cat results/evolution_result_*.json | jq
```

### Check Logs
```bash
tail -f logs/prompt_evolution.log
```

### Switch API Provider
```bash
# Edit config/config.yaml
api_provider: "gemini"  # Change to "openai" or "gemini"
```

## Troubleshooting

### Import Error
```bash
# Make sure you're using uv run
uv run python run_evolution.py

# Or add src to path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### API Key Issues
```bash
# Verify key is set
echo $OPENAI_API_KEY

# Or put in config/config.yaml
api_keys:
  openai: "sk-..."
```

### Path Errors
All paths are now relative to project root:
- Config: `config/config.yaml`
- Logs: `logs/prompt_evolution.log`
- Results: `results/`
- Data: `data/`

## Files & Directories

| Path | Purpose |
|------|---------|
| `run_evolution.py` | Main script to run evolution |
| `src/promptevolve/` | Python package with all source code |
| `config/config.yaml` | Main configuration |
| `docs/` | All documentation |
| `examples/` | Example scripts |
| `results/` | Evolution output (JSON files) |
| `logs/` | System logs |

## Next Steps

1. ✅ Basic setup complete
2. Review `docs/README.md` for detailed documentation
3. Check `docs/SCENARIO_USAGE_GUIDE.md` to understand test scenarios
4. See `examples/` for usage patterns
5. Customize `config/config.yaml` for your use case

## Support

- **Documentation**: `/README.md`
- **API Test Results**: `docs/API_KEY_TEST_RESULTS.md`
- **Project Structure**: `docs/PROJECT_STRUCTURE.md`
- **Scenario Guide**: `docs/SCENARIO_USAGE_GUIDE.md`
