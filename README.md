# WarmStart - AI Prompt Optimizer 🚀# WarmStart: Self-Evolving Prompt Optimization System



**Evolve your prompts to perfection in seconds using genetic algorithms and AI judges.**A production-ready genetic algorithm-based prompt evolution system with RAG-based pattern learning and warm-start capabilities.



No test cases needed. Just provide your prompt and watch it evolve.## Overview



---WarmStart uses evolutionary algorithms to automatically optimize prompts across domains, extracting successful patterns into a reusable knowledge base. New domains benefit from patterns learned in previous optimizations (warm-start), dramatically reducing time-to-production.



## ⚡ Quick Start### Key Features



```bash- **Multi-provider LLM support**: OpenAI, Anthropic, Gemini, Groq

python simple_cli.py "Your prompt here"- **Genetic Algorithm optimization**: Tournament selection, elitism, mutation strategies

```- **RAG Pattern Library**: Vector DB-backed pattern reuse across domains

- **Cost optimization**: Cheap iterations + periodic production validation

**That's it!** The system will optimize your prompt automatically.- **HITL governance**: Human-in-the-loop review and approval

- **Production monitoring**: Drift detection, automated retraining triggers

---

## Architecture

## 🎯 Example

```

```bash┌─────────────────────────────────────────────────────────────┐

python simple_cli.py "Summarize text" \│                     Evolution Engine                         │

  --context "Medical discharge summaries for patient handoff" \│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐     │

  --domain medical│  │ Population │→ │ Tournament   │→ │ Mutation        │     │

```│  │ Management │  │ Selection    │  │ (Challenger)    │     │

│  └────────────┘  └──────────────┘  └─────────────────┘     │

**Result in 22 seconds:**└─────────────────────────────────────────────────────────────┘

         ↓                    ↑                    ↓

| Metric | Score |┌──────────────┐     ┌────────────────┐    ┌──────────────┐

|--------|-------|│ PromptSolver │     │ PromptVerifier │    │  RAG Pattern │

| **Original** | "Summarize text" |│  (Executor)  │────→│   (Judge)      │    │   Library    │

| **Optimized** | "Condense the patient's medical history, treatment plan, and follow-up instructions into a concise summary for seamless handoff to the next healthcare provider" |└──────────────┘     └────────────────┘    └──────────────┘

| **Quality Score** | **0.88/1.0** ⭐ |         ↓                    ↓                    ↑

| **Clarity** | 0.90 |┌─────────────────────────────────────────────────────────────┐

| **Specificity** | 0.95 |│               Multi-Provider LLM Gateway                     │

| **Effectiveness** | 0.90 |│        OpenAI │ Anthropic │ Gemini │ Groq                   │

└─────────────────────────────────────────────────────────────┘

---```



## 📦 Installation## Quick Start



```bash### Installation

# 1. Install dependencies

pip install -r requirements.txt1. **Run setup script:**

```powershell

# 2. Set up API keys.\setup.ps1

# Create .env file with:```

OPENAI_API_KEY=your_key_here

XAI_API_KEY=your_xai_key_here  # Optional fallbackThis will create a virtual environment, install dependencies, and initialize the database.

```

2. **Your API keys are already configured** in `.env`

---

### Test the System

## 💡 Usage
### Vector DB with uv (Windows)

If you're on Windows and want to enable the vector DB (Chroma) for semantic warm-start:

1) Install uv (fast Python env manager)
- PowerShell (one-time):
  - Visit https://docs.astral.sh/uv/getting-started/ and install uv (or `pipx install uv`).

2) Create and activate a Python 3.11 virtualenv
- PowerShell:
  - uv venv --python 3.11 venv
  - .\venv\Scripts\Activate.ps1

3) Install dependencies (includes chromadb + sentence-transformers)
- PowerShell:
  - uv pip install --upgrade pip setuptools wheel
  - uv pip install -r .\requirements.txt

4) Run the optimizer (no-emoji prevents Windows console issues)
- PowerShell:
  - python .\simple_cli.py "Your prompt" --context "task context" --domain general --no-emoji

Notes:
- If `PatternLibrary initialized (ChromaDB)` appears in logs, RAG uses the vector index. Otherwise, it will fall back to SQL/JSONL.
- If you previously ran without Chroma, you can backfill existing artifacts into the vector index:
  - python tools\reindex_vector.py --limit 1000

Flags of interest:
- `--rag-off` to disable warm-start
- `--rag-top-k N` to control retrieved artifacts
- `--no-emoji` to strip non-ASCII on Windows consoles


**Test core components** (~2 minutes, $0.10):

### Basic```powershell

```bashpython test_core_components.py

python simple_cli.py "Extract key points"```

```

**Run a full evolution** (~5 minutes, $0.50-1.00):

### With Context (Recommended)```powershell

```bashpython test_evolution.py

python simple_cli.py "Extract key points" \```

  --context "Legal contracts focusing on payment terms"

```### What You'll See



### Full OptionsThe evolution will:

```bash1. Start with 3 seed prompts

python simple_cli.py "Your prompt" \2. Generate mutations using LLM

  --domain legal \3. Evaluate each prompt on test cases

  --context "Specific use case details" \4. Select best performers

  --population 8 \5. Evolve for 5 generations

  --generations 5 \6. Output champion prompt with fitness score

  --output results.json

```Expected output:

```

--->>> Generation 1/5

Evaluating 10 members...

## 🎨 DomainsGeneration 1 complete: best_fitness=0.82, avg_fitness=0.68



Choose a domain for better optimization:🏆 Champion Prompt (Fitness: 0.89):

[Your optimized prompt here]

| Domain | Use For | Optimizes For |

|--------|---------|---------------|Total cost: $0.75

| `legal` | Contracts, legal docs | Precision, completeness, formal language |```

| `medical` | Clinical notes, summaries | Accuracy, clarity, medical terminology |

| `code` | Code review, analysis | Technical detail, structure, specificity |### Next Steps

| `general` | Everything else | Balanced quality across all dimensions |

See **`QUICKSTART.md`** for detailed instructions and **`SESSION_COMPLETE.md`** for complete status!

---

## Project Structure

## 📊 How It Works

```

```WarmStart/

1. 🧬 MUTATION PHASE├── src/

   ├─ Generates 3-8 prompt variations│   ├── core/              # Core components

   ├─ Uses your context for domain-specific improvements│   │   ├── llm_client.py      # Multi-provider LLM abstraction

   └─ Parallel generation (fast!)│   │   ├── prompt_solver.py   # Batch execution engine

│   │   ├── prompt_verifier.py # Evaluation & Judge LLM

2. ⚖️ JUDGE EVALUATION PHASE│   │   └── prompt_challenger.py # Mutation generator

   ├─ GPT-4 evaluates each prompt│   ├── evolution/         # Genetic algorithm

   ├─ Scores: Clarity, Specificity, Structure, Completeness, Effectiveness│   │   ├── engine.py          # Main GA loop

   └─ Parallel evaluation (fast!)│   │   ├── population.py      # Population management

│   │   └── selection.py       # Selection strategies

3. 📈 EVOLUTION│   ├── patterns/          # Pattern extraction & RAG

   ├─ Selects best prompts│   │   ├── extractor.py       # Pattern mining

   ├─ Creates next generation│   │   ├── vector_db.py       # Vector DB client

   └─ Stops when quality target reached (0.90)│   │   └── retriever.py       # RAG retrieval

│   ├── data/              # Dataset management

✅ Total Time: 20-60 seconds│   │   ├── dataset.py         # Golden set & synthetic

💰 Cost: ~$0.02-0.05 per run│   │   ├── hitl.py            # Human-in-the-loop

```│   │   └── validators.py     # Deterministic checks

│   ├── monitoring/        # Production monitoring

---│   │   ├── tracker.py         # Metrics tracking

│   │   ├── drift.py           # Drift detection

## 🎯 Parameters│   │   └── governance.py      # Safety & constitution

│   ├── models/            # Database models

| Parameter | Default | Description |│   │   └── schema.py

|-----------|---------|-------------|│   ├── utils/             # Utilities

| `prompt` | *Required* | Your initial prompt |│   │   ├── config.py

| `--domain` | `general` | Domain: `legal`, `medical`, `code`, `general` |│   │   ├── logging.py

| `--context` | `None` | Use case context (highly recommended!) |│   │   └── cost_tracker.py

| `--population` | `8` | Candidates per generation (4-12) |│   └── cli.py             # Command-line interface

| `--generations` | `5` | Max generations (3-10) |├── tests/

| `--output` | `None` | Save to JSON file |├── config/

│   ├── default.yaml

---│   └── domains/           # Domain-specific configs

│       └── legal.yaml

## 📈 Quality Metrics├── data/

│   ├── golden/            # Gold standard datasets

Every prompt is scored on 5 dimensions (0.0 to 1.0):│   ├── synthetic/         # Generated test cases

│   └── artifacts/         # Extracted patterns

| Dimension | What It Measures |├── experiments/           # Run logs and results

|-----------|------------------|├── migrations/            # Database migrations

| **Clarity** | Clear, unambiguous instructions |├── notebooks/             # Analysis notebooks

| **Specificity** | Specific requirements and constraints |├── requirements.txt

| **Structure** | Well-organized format |├── .env.example

| **Completeness** | Covers all necessary aspects |└── README.md

| **Effectiveness** | Likely to produce good results |```



**Overall Score** = Weighted average## Phases



Target: **0.90** (system stops early if reached)### Phase 0: Prep & Design ✓

- Scope, metrics, infrastructure decisions

---

### Phase 1: Eval Dataset & HITL

## 🌟 Real Examples- Golden dataset creation

- Synthetic test case generation

### Legal Domain- Human review pipeline

```bash

python simple_cli.py "Extract key terms" \### Phase 2: Core Components (MVP)

  --domain legal \- PromptSolver, PromptVerifier, PromptChallenger

  --context "Commercial contracts"- EvolutionEngine, PatternExtractor

```

### Phase 3: RAG Pattern Library

**Before:** "Extract key terms"  - Vector DB integration

**After:** "Extract key legal terms from contracts and categorize them into clauses, obligations, and definitions. Create a detailed report including a summary of each category and any relevant case law."  - Pattern retrieval and warm-start

**Score:** 0.90

### Phase 4: Cost Control

### Code Review- Two-phase evaluation

```bash- Context compaction

python simple_cli.py "Review this code" \

  --domain code \### Phase 5: Validation & Pattern Ingestion

  --context "Python security vulnerabilities"- Champion selection

```- Pattern extraction and HITL approval



**Before:** "Review this code"  ### Phase 6: Monitoring & Governance

**After:** "Perform a comprehensive security review of this Python code, identifying potential vulnerabilities including SQL injection, XSS, authentication bypasses, and insecure dependencies. Provide severity ratings and remediation steps for each finding."  - Production monitoring

**Score:** 0.92- Automated retraining

- Safety checks

---

## Metrics Tracked

## 💡 Pro Tips

- `training_avg_score`: Average performance during evolution

### 1. Always Provide Context- `final_validation_score`: Champion performance on holdout set

```bash- `consistency_score`: Output stability across runs

# ❌ Vague- `efficiency_score`: Token usage and latency

python simple_cli.py "Analyze data"- `tokens_used`: Total token consumption

- `production_mismatch_rate`: Tier-3 vs Tier-1 alignment

# ✅ Specific- `pattern_effectiveness`: Success rate of extracted patterns

python simple_cli.py "Analyze data" \

  --context "Customer churn analysis for SaaS product, focus on behavioral patterns"## Default Parameters

```

```yaml

The more specific your context, the better the optimization!population_size: 50

tournament_size: 2

### 2. Choose the Right Domainelite_percentage: 5

- Business docs → `general`mutation_rate: 0.8

- Legal/contracts → `legal`generations: 50

- Medical/clinical → `medical`rag_retrieval_k: 5

- Technical/code → `code`prod_check_cadence: 10

prod_check_top_n: 5

### 3. Adjust Speed vs Qualitybatch_eval_size: 50

```bash```

# Fast (20s) - Good for iteration

--population 4 --generations 3## License



# Balanced (40s) - RecommendedMIT

--population 8 --generations 5

## Contributing

# Thorough (90s) - Maximum quality

--population 12 --generations 8See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```

### 4. Save Your Results
```bash
python simple_cli.py "prompt" --output my_prompt.json
```

Output includes:
- Original & optimized prompts
- All quality scores
- Strengths & weaknesses
- Evolution statistics

---

## 🔧 Troubleshooting

### "Rate limit exceeded"
**Solution:**
1. Add `XAI_API_KEY` to `.env` for automatic fallback
2. Reduce `--population` to 4-6
3. Wait 60 seconds between runs

### Low scores / not improving
**Solution:**
1. Add more specific `--context`
2. Increase `--population` (more diversity)
3. Try a different `--domain`
4. Run multiple times and compare

### Parse errors during mutations
**Normal!** System automatically uses fallback. 1-2 failures per run is expected and doesn't affect quality.

---

## 📂 Output Example

```
================================================================================
🧬 MUTATION PHASE: Generating 5 initial mutations in parallel
================================================================================

================================================================================
📊 GENERATION 1/5
================================================================================

================================================================================
⚖️  JUDGE EVALUATION PHASE: Evaluating 6 prompts
================================================================================

Best fitness: 0.840

================================================================================
📈 Generation 0 Summary: avg=0.667, best=0.840, diversity=1.000, cost=$0.0045
================================================================================

================================================================================
🧬 MUTATION PHASE: Generating next generation
================================================================================

================================================================================
📊 GENERATION 2/5
================================================================================

================================================================================
⚖️  JUDGE EVALUATION PHASE: Evaluating 5 prompts
================================================================================

🎯 Target score 0.9 achieved! Fitness: 0.910

✅ OPTIMIZATION COMPLETE!

================================================================================
🔴 YOUR ORIGINAL PROMPT:
================================================================================
Extract key terms

================================================================================
🟢 OPTIMIZED PROMPT (Quality Score: 0.910):
================================================================================
Extract key legal terms from contracts and categorize them into clauses,
obligations, and definitions. Create a detailed report including a summary
of each category and any relevant case law.

📈 Quality Breakdown:
  Clarity:       0.90
  Specificity:   0.95
  Structure:     0.85
  Completeness:  0.92
  Effectiveness: 0.93

✅ Strengths:
  • High specificity
  • Clear categorization structure
  • Comprehensive reporting format

⚠️  Areas for Improvement:
  • Could add examples of each category
```

---

## 🏗️ Architecture

```
simple_cli.py
├─ SimpleEvolutionEngine
│  ├─ PromptQualityEvaluator (GPT-4 judge)
│  ├─ PromptChallenger (GPT-3.5 mutations)
│  └─ Population (candidate management)
└─ SQLite database (results storage)
```

**Fast & Efficient:**
- Mutations: GPT-3.5 (cheap, fast)
- Evaluation: GPT-4 (accurate, reliable)
- Parallel execution: Multiple prompts at once
- Early stopping: Stops when target reached

---

## 📄 Files

- `simple_cli.py` - Main entry point
- `src/evolution/simple_engine.py` - Evolution logic
- `src/core/prompt_quality_evaluator.py` - Judge evaluation
- `src/core/prompt_challenger.py` - Mutation generation
- `config/default.yaml` - Configuration
- `warmstart.db` - Results database

---

## 🎓 Understanding the Output

### Quality Scores Explained

**0.90+** = Excellent! Production-ready  
**0.80-0.89** = Very good, minor improvements possible  
**0.70-0.79** = Good, some improvements needed  
**< 0.70** = Needs work, add more context

### Generation Stats

- **avg**: Average fitness of all candidates
- **best**: Best candidate fitness
- **diversity**: How different the candidates are (1.0 = very diverse)
- **cost**: Total API cost so far

---

## 🚀 Advanced Usage

### Programmatic API

```python
import asyncio
from src.evolution.simple_engine import SimpleEvolutionEngine
from src.evolution.results import EvolutionConfig
from src.utils.config import get_config

async def optimize():
    config = get_config(domain="legal")
    evo_config = EvolutionConfig.from_dict(config.to_dict())
    
    engine = SimpleEvolutionEngine(
        domain="legal",
        task_description="Contract analysis",
        initial_prompt="Extract terms",
        user_context="Commercial agreements",
        config=evo_config
    )
    
    result = await engine.evolve()
    return result.champion_prompt, result.champion_fitness

prompt, score = asyncio.run(optimize())
print(f"Optimized: {prompt}")
print(f"Score: {score:.3f}")
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Speed** | 20-60 seconds |
| **Cost** | $0.02-0.05 per run |
| **Success Rate** | 90%+ reach 0.90 target |
| **Typical Generations** | 1-3 (early stopping) |
| **API Calls** | 10-25 per run |

---

## 🔐 Requirements

- Python 3.10+
- OpenAI API key (required)
- XAI API key (optional, for fallback)

---

## 📝 License

MIT - Free to use and modify

---

## 🙏 Credits

Built with:
- OpenAI GPT-4 (evaluation)
- OpenAI GPT-3.5 (mutations)
- XAI Grok (fallback provider)
- Genetic algorithms
- Lots of coffee ☕

---

## 🎯 Quick Reference

```bash
# Minimal
python simple_cli.py "Your prompt"

# Recommended
python simple_cli.py "Your prompt" \
  --context "Your use case" \
  --domain <legal|medical|code|general>

# Full control
python simple_cli.py "Your prompt" \
  --context "Your use case" \
  --domain legal \
  --population 8 \
  --generations 5 \
  --output result.json
```

**Need help?** Check `SIMPLE_MODE.md` for more examples and tips!

---

**Happy prompt optimization! 🎉**
