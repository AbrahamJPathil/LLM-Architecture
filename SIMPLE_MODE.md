# Simple WarmStart - Prompt Optimization Without Test Cases

## Overview

The **Simple WarmStart** system optimizes prompts **without requiring test cases**. You only provide:
1. Your prompt
2. Optional context about your use case
3. Optional domain

The system evaluates prompt quality directly using LLM judges.

## Quick Start

### Basic Usage

```bash
python simple_cli.py "Your prompt here"
```

### With Domain

```bash
python simple_cli.py "Extract key contract terms" --domain legal
```

### With Context

```bash
python simple_cli.py "Analyze code for bugs" --context "Python security reviews"
```

### Full Example

```bash
python simple_cli.py "Summarize medical reports" \
  --domain medical \
  --context "Focus on diagnosis and treatment plans" \
  --population 10 \
  --generations 5 \
  --output results.json
```

## How It Works

### 1. **Initialization** (5-10 seconds)
- Creates initial population through parallel mutation generation
- Generates 5-15 variations of your prompt

### 2. **Evolution** (20-60 seconds depending on settings)
- Evaluates prompt quality on multiple dimensions:
  - **Clarity**: Is it clear and unambiguous?
  - **Specificity**: Does it provide specific instructions?
  - **Structure**: Is it well-organized?
  - **Completeness**: Does it cover necessary aspects?
  - **Effectiveness**: Will it produce good results?

- Each generation:
  1. Evaluates all prompts in parallel using GPT-4 judge
  2. Selects best performers
  3. Generates mutations for next generation
  4. Stops early if target quality score reached (0.90)

### 3. **Results**
- Shows original vs optimized prompt
- Quality breakdown by dimension
- Strengths and improvement areas
- Evolution statistics

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--domain` | general | Domain: legal, medical, code, general |
| `--context` | None | Additional context about your use case |
| `--population` | 8 | Population size (more = better but slower) |
| `--generations` | 5 | Max generations to run |
| `--output` | None | Save results to JSON file |
| `--mock` | False | Use mock LLM for testing (no API costs) |

## Example Run

```bash
$ python simple_cli.py "Extract key legal terms from contracts" --domain legal

🚀 STARTING OPTIMIZATION
📝 YOUR INITIAL PROMPT:
   "Extract key legal terms from contracts"

Generating 5 initial mutations in parallel...
>>> Generation 1/3
Evaluating 6 members...
Best fitness: 0.880

>>> Generation 2/3
Evaluating 5 members...
🎯 Target score 0.9 achieved! Fitness: 0.900

✅ OPTIMIZATION COMPLETE!
Duration: 32.0s
Stopping reason: target_score_achieved (0.900)

================================================================================
🔴 YOUR ORIGINAL PROMPT:
================================================================================
Extract key legal terms from contracts

================================================================================
🟢 OPTIMIZED PROMPT (Quality Score: 0.900):
================================================================================
Extract key legal terms from contracts and categorize them into clauses, 
obligations, and definitions. Create a detailed report including a summary 
of each category and any relevant case law.

📈 Quality Breakdown:
  Clarity:       0.90
  Specificity:   0.95
  Structure:     0.85
  Completeness:  0.90
  Effectiveness: 0.92

✅ Strengths:
  • High specificity
  • Clear categorization tasks
  • Inclusion of case law enhances completeness

⚠️  Areas for Improvement:
  • Might lack guidance on how to identify relevant case law
```

## Performance

- **Initial mutations**: 5-10 seconds (parallel)
- **Per generation**: 5-15 seconds
- **Total runtime**: 20-60 seconds
- **API cost**: $0.01-0.05 per run (using GPT-4 for evaluation, GPT-3.5 for mutations)

## Advantages vs Test Case Approach

### ✅ Simple Mode (No Test Cases)
- **Fast**: 30-60 seconds total
- **Easy**: Just provide prompt and context
- **General**: Works for any task
- **Low cost**: ~$0.02 per run

### 🔬 Test Case Mode (Original)
- **More precise**: Evaluates actual outputs
- **Domain-specific**: Tests on real examples
- **Validated**: Measures performance on specific cases
- **Higher cost**: ~$0.10-0.50 per run
- **Slower**: 2-5 minutes

## When to Use Which?

### Use Simple Mode When:
- You want quick prompt improvements
- You don't have test cases ready
- You're exploring different phrasings
- Cost/speed is a priority

### Use Test Case Mode When:
- You have specific test cases
- You need precise performance metrics
- You're optimizing for a specific task
- Quality is more important than speed

## Tips

1. **Be specific in context**: More context = better optimization
   ```bash
   # Good
   python simple_cli.py "Summarize text" \
     --context "Medical discharge summaries for patient handoff"
   
   # Less effective
   python simple_cli.py "Summarize text"
   ```

2. **Choose right domain**: Helps tailor evaluation criteria
   - `legal`: Precision, completeness, formal language
   - `medical`: Accuracy, clarity, clinical terminology
   - `code`: Structure, specificity, technical detail
   - `general`: All-around quality

3. **Adjust population/generations** based on needs:
   - **Fast explore**: `--population 6 --generations 3` (20s)
   - **Balanced**: `--population 8 --generations 5` (40s)
   - **Thorough**: `--population 12 --generations 8` (90s)

4. **Save results** for later reference:
   ```bash
   python simple_cli.py "Your prompt" --output optimized_prompt.json
   ```

## Output File Format

When using `--output`, saves JSON with:

```json
{
  "original_prompt": "Extract key legal terms...",
  "optimized_prompt": "Extract key legal terms and categorize...",
  "quality_score": 0.900,
  "context": "Your context",
  "domain": "legal",
  "generations": 1,
  "cost_usd": 0.0234,
  "duration_seconds": 32.0,
  "quality_breakdown": {
    "clarity": 0.90,
    "specificity": 0.95,
    "structure": 0.85,
    "completeness": 0.90,
    "effectiveness": 0.92,
    "strengths": ["High specificity", "Clear categorization"],
    "weaknesses": ["Might lack guidance on case law"],
    "suggestions": ["Add examples", "Define output format"]
  }
}
```

## Troubleshooting

### "Rate limit exceeded"
- Wait 60 seconds and retry
- Reduce `--population` size
- System will automatically retry with XAI fallback

### Optimization converges slowly
- Increase `--population` for more diversity
- Add more specific `--context`
- Try different `--domain`

### Results not satisfactory
- Run multiple times and compare
- Provide more detailed context
- Use test case mode for specific requirements
