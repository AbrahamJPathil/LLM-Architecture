# WarmStart - Comprehensive Documentation

**WarmStart** is an AI-powered prompt optimization system that uses evolutionary algorithms to automatically improve prompts without requiring test cases. It learns from past optimizations using a **Retrieval-Augmented Generation (RAG)** approach with ChromaDB vector storage.

---

## Table of Contents

0. [Getting Started from Scratch](#getting-started-from-scratch)
1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Cold Start vs Warm Start](#cold-start-vs-warm-start)
4. [Evolution Process](#evolution-process)
5. [Command Line Interface](#command-line-interface)
6. [Inspection & Debugging Tools](#inspection--debugging-tools)
7. [Project Structure](#project-structure)
8. [Understanding the Output](#understanding-the-output)
9. [Database & Artifacts](#database--artifacts)
10. [Advanced Topics](#advanced-topics)

---

## Getting Started from Scratch

**Complete setup guide for someone with zero setup - follow these steps in order:**

### Prerequisites

Before you start, ensure you have:

1. **Python 3.11 or higher** installed
   - Check version: `python --version`
   - Download from: https://www.python.org/downloads/
   - ⚠️ **Important**: During installation, check "Add Python to PATH"

2. **Git** (optional, for cloning the repository)
   - Download from: https://git-scm.com/downloads

3. **OpenAI API Key**
   - Sign up at: https://platform.openai.com/
   - Go to API Keys section and create a new key
   - You'll need billing set up (system costs ~$0.01-0.05 per optimization)

4. **Windows PowerShell** or **Command Prompt**
   - Already included with Windows

### Step-by-Step Setup

#### Step 1: Get the Project

**Option A: If you have the folder already**
```powershell
cd C:\Users\YourName\OneDrive\Desktop\WarmStart
```

**Option B: Clone from repository (if hosted on GitHub)**
```powershell
git clone https://github.com/your-repo/WarmStart.git
cd WarmStart
```

**Option C: Download ZIP**
- Download and extract to your desired location
- Open PowerShell and navigate to that folder

#### Step 2: Run the Setup Script

```powershell
# Run the automated setup (creates virtual environment and installs dependencies)
.\setup.ps1
```

**What this does:**
- Creates a Python virtual environment in `venv/`
- Installs all required packages from `requirements.txt`
- Sets up the project structure

**If you see an error about execution policy:**
```powershell
# Run this first, then try setup again
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Step 3: Configure Your API Key

```powershell
# Copy the example environment file
copy .env.example .env

# Open .env in Notepad
notepad .env
```

**Edit the `.env` file:**
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Save and close the file.

#### Step 4: Activate Virtual Environment

```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# You should see (venv) at the start of your prompt
```

**If activation fails:**
```powershell
# Alternative activation method
.\venv\Scripts\python.exe -m pip list
# Or just use: .\venv\Scripts\python instead of python in commands
```

#### Step 5: Verify Installation

```powershell
# Check that dependencies are installed
python -c "import chromadb; import openai; print('✓ All dependencies installed!')"
```

**Expected output:** `✓ All dependencies installed!`

**If you see errors:**
```powershell
# Reinstall dependencies
.\venv\Scripts\python -m pip install -r requirements.txt
```

#### Step 6: Run Your First Optimization! 🚀

```powershell
# Simple test run
python simple_cli.py "Summarize customer feedback" --no-emoji
```

**Expected behavior:**
- Takes 10-30 seconds
- Shows evolution progress
- Displays before/after prompts
- Costs approximately $0.01-0.03

**Your first run is a "cold start"** (no warm-start patterns yet)

#### Step 7: Check the Database

```powershell
# View what was stored
python tools/inspect_library.py
```

You should see patterns extracted from your first run!

#### Step 8: Run Again (Now with Warm Start!)

```powershell
# Same or similar task - will use warm-start
python simple_cli.py "Analyze customer reviews" --no-emoji
```

**You should see:**
```
Retrieved 5 warm-start artifacts:
  • CLEAR_TASK_DEFINITION (score: 0.904)
  • OUTPUT_FORMATTING (score: 0.880)
  ...
```

**Congratulations! You're now using warm-start optimization! 🎉**

---

### Quick Reference Card

**After setup, here are your main commands:**

```powershell
# Activate environment (do this each time you open a new terminal)
.\venv\Scripts\Activate.ps1

# Optimize a prompt
python simple_cli.py "Your prompt here" --domain general

# View pattern library
python tools/inspect_library.py

# View complete database
python tools/show_all_db.py

# Deactivate environment (when done)
deactivate
```

---

### Troubleshooting First-Time Setup

#### "Python not found"
```powershell
# Use full path to Python
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe simple_cli.py "test"

# Or add Python to PATH:
# 1. Search "Environment Variables" in Windows Start
# 2. Edit "Path" variable
# 3. Add Python installation directory
```

#### "pip is not recognized"
```powershell
# Use python -m pip instead
python -m pip install -r requirements.txt
```

#### "OpenAI API key not found"
```powershell
# Verify .env file exists and has correct format
Get-Content .env

# Should show: OPENAI_API_KEY=sk-...
# No spaces around the = sign
# No quotes around the key
```

#### "ChromaDB installation failed"
```powershell
# Install Visual C++ Build Tools (required for some packages)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then reinstall:
pip install --upgrade chromadb sentence-transformers
```

#### "Rate limit exceeded" from OpenAI
```powershell
# Use smaller runs while testing:
python simple_cli.py "test" --population 4 --generations 2

# Or use mock mode (no API calls):
python simple_cli.py "test" --mock
```

---

## Quick Start

### Installation

**Windows:**
```powershell
# Clone or navigate to the project
cd WarmStart

# Run setup script (creates venv, installs dependencies)
.\setup.ps1

# Create .env file with your OpenAI API key
copy .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```

**Linux/macOS:**
```bash
# Clone or navigate to the project
cd WarmStart

# Run setup script (creates venv, installs dependencies)
chmod +x setup.sh
./setup.sh

# Create .env file with your OpenAI API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```

### Basic Usage

**Windows:**
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Optimize a prompt (simplest form)
python simple_cli.py "Summarize customer feedback"

# With domain and context
python simple_cli.py "Extract contract terms" --domain legal --context "Real estate contracts"

# Larger optimization run
python simple_cli.py "Your prompt" --population 15 --generations 8
```

**Linux/macOS:**
```bash
# Activate virtual environment
source venv/bin/activate

# Optimize a prompt (simplest form)
python simple_cli.py "Summarize customer feedback"

# With domain and context
python simple_cli.py "Extract contract terms" --domain legal --context "Real estate contracts"

# Larger optimization run
python simple_cli.py "Your prompt" --population 15 --generations 8
```

---

## Core Concepts

### What is WarmStart?

WarmStart automatically improves prompts through **evolutionary optimization**:
- **No test cases required** - Uses LLM judges to evaluate quality
- **Learns from history** - Stores successful patterns and reuses them
- **Domain-aware** - Optimizes for specific domains (legal, medical, code, general)
- **Fast & cost-effective** - Uses tiered LLM strategy (fast mutations, quality evaluation)

### Key Components

#### 1. **Evolutionary Algorithm**
- Creates a **population** of prompt variations
- Evaluates each using quality metrics (clarity, specificity, structure, etc.)
- Selects the best prompts (**elitism**)
- **Mutates** winners to create new generations
- Repeats until quality target is reached or max generations

#### 2. **Pattern Library (RAG Store)**
- Stores successful prompt patterns as **Prompt DNA**
- Uses **ChromaDB** (vector database) for semantic retrieval
- Enables **warm-start**: new optimizations learn from past successes
- Falls back to SQL and JSONL if vector DB unavailable

#### 3. **LLM Judge System**
- **Tier 1** (GPT-4): High-quality evaluation and pattern extraction
- **Tier 2** (GPT-4o-mini): Balanced cost/performance for judgments
- **Tier 3** (GPT-4o-mini): Fast, cheap mutations
- Can use **mock mode** for testing without API costs

---

## Cold Start vs Warm Start

### Cold Start 🥶
**First time running or when library is empty**

- Starts with your **initial prompt only**
- No prior knowledge to draw from
- Explores randomly through mutations
- Slower to find optimal patterns
- Every run "reinvents the wheel"

**Example:**
```powershell
# Force cold start (disable RAG)
python simple_cli.py "Your prompt" --rag-off
```

**Log Output:**
```
[info] PatternLibrary initialized (ChromaDB)
[info] Warm-start seeding: no artifacts found
[info] Cold start: Population will start with initial prompt only
```

### Warm Start 🔥
**When library has artifacts from previous runs**

- **Queries vector database** for similar past patterns
- Seeds population with proven techniques
- Starts optimization at a higher baseline
- Faster convergence to high-quality prompts
- Compounds knowledge over time

**Example:**
```powershell
# Normal run (warm-start enabled by default)
python simple_cli.py "Analyze customer reviews" --domain general
```

**Log Output:**
```
[info] PatternLibrary initialized (ChromaDB)
[info] Retrieved 5 warm-start artifacts:
[info]   • DOMAIN_SPECIFICITY (score: 0.904)
[info]   • CLEAR_TASK_DEFINITION (score: 0.904)
[info]   • OUTPUT_FORMATTING (score: 0.880)
[info]   • CATEGORIZATION_BY_SENTIMENT (score: 0.904)
[info]   • STRUCTURED_OUTPUT_SECTIONS (score: 0.740)
[info] Seeding population with 5 high-quality patterns...
```

### How Warm-Start Works

1. **Query Phase**: When you start a new optimization, WarmStart searches the vector database for artifacts similar to your task
2. **Seeding Phase**: Top-K artifacts (default: 5) are retrieved and used to seed the initial population
3. **Evolution Phase**: Evolution proceeds normally but starts from a much better baseline
4. **Storage Phase**: After completion, new patterns are extracted and stored back to the library

**RAG Retrieval Flow:**
```
Your Prompt → Embed with sentence-transformers
              ↓
ChromaDB vector search (cosine similarity)
              ↓
Top-K artifacts retrieved → Seed population
              ↓
Evolution starts with proven patterns
```

---

## Evolution Process

### Population & Generations

**Population**: A group of prompt variations competing against each other
- Default size: **8 prompts**
- Each prompt is evaluated independently
- Best performers survive to next generation

**Generation**: One complete cycle of evaluate → select → mutate
- Default: **5 generations**
- Each generation typically improves quality
- Early stopping if target score reached

### Selection Strategy: Tournament + Elitism

#### Tournament Selection
- Randomly pick small groups of prompts
- Best prompt from each group advances
- Introduces randomness while favoring quality
- Default tournament size: **3**

#### Elitism
- **Top performers always survive** to next generation
- Prevents losing good solutions
- Default elite percentage: **30%**
- Example: In population of 10, top 3 are guaranteed to survive

### Mutations

**Mutations** create variations of successful prompts. Types include:

1. **CLARIFICATION_IMPROVEMENT** - Add specific details and examples
2. **STRUCTURE_ENHANCEMENT** - Improve formatting and organization
3. **CONTEXT_EXPANSION** - Add relevant domain knowledge
4. **CONSTRAINT_ADDITION** - Add requirements or limitations
5. **SIMPLIFICATION** - Remove unnecessary complexity
6. **STYLE_ADJUSTMENT** - Change tone or formality
7. **SPECIFICITY_BOOST** - Make instructions more precise
8. **FORMAT_CHANGE** - Alter output structure requirements

**Example Mutation:**
```
Original: "Summarize the document"
↓ CLARIFICATION_IMPROVEMENT
Result: "Summarize the document in 3-5 bullet points, focusing on 
        key findings, recommendations, and action items."
```

### Quality Evaluation (LLM Judge)

Each prompt is scored on **five dimensions** (0-1 scale):

1. **Clarity** (0.30 weight): Is the prompt easy to understand?
2. **Specificity** (0.25 weight): Does it provide clear requirements?
3. **Structure** (0.20 weight): Is it well-organized?
4. **Completeness** (0.15 weight): Does it cover all necessary aspects?
5. **Effectiveness** (0.10 weight): Will it produce desired results?

**Final Fitness Score** = Weighted average of dimensions

**Target Score**: 0.90 (default) - High-quality prompt
**Early Stopping**: Stops if target reached before max generations

---

## Command Line Interface

### Main Program: `simple_cli.py`

**Basic Syntax:**
```powershell
python simple_cli.py "<prompt>" [options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `prompt` | The prompt you want to optimize | `"Summarize customer feedback"` |

### Optional Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--domain` | string | `general` | Domain context: `legal`, `medical`, `code`, `general` |
| `--context` | string | `None` | Additional context about your use case |
| `--population` | int | `8` | Number of prompts per generation |
| `--generations` | int | `5` | Maximum generations to evolve |
| `--output` | string | `None` | Save results to JSON file |
| `--mock` | flag | `False` | Use mock LLM (no API calls, testing only) |
| `--no-emoji` | flag | `False` | Strip emojis (Windows PowerShell compatibility) |
| `--rag-off` | flag | `False` | Disable warm-start (force cold start) |
| `--rag-top-k` | int | `5` | Number of artifacts to retrieve for warm-start |

### Examples

#### 1. Basic Optimization
```powershell
python simple_cli.py "Write a product description"
```

#### 2. Legal Domain with Context
```powershell
python simple_cli.py "Extract key contract terms" --domain legal --context "Commercial lease agreements"
```

#### 3. Large Optimization Run
```powershell
python simple_cli.py "Debug this code" --domain code --population 15 --generations 10
```

#### 4. Save Results to File
```powershell
python simple_cli.py "Your prompt" --output results.json
```

#### 5. Windows-Safe Output (No Emojis)
```powershell
python simple_cli.py "Your prompt" --no-emoji
```

#### 6. Force Cold Start (Disable RAG)
```powershell
python simple_cli.py "Your prompt" --rag-off
```

#### 7. Retrieve More Warm-Start Patterns
```powershell
python simple_cli.py "Your prompt" --rag-top-k 10
```

---

## Inspection & Debugging Tools

### 1. Inspect Pattern Library

**File**: `tools/inspect_library.py`

Shows the **latest artifacts** stored in the pattern library with their quality scores and provenance.

```powershell
python tools/inspect_library.py
```

**Output Example:**
```
PROMPT PATTERN LIBRARY - Total artifacts: 30
Latest artifacts:

1. EXAMPLE_PROVIDING (pattern) - Score: 0.880
   Domains: general
   Description: Includes an example of the expected output...
   From Run: run_general_f1586955 | Domain: general | Started: 2025-11-05 04:21:00

2. CONTENT_REQUIREMENT (pattern) - Score: 0.880
   Domains: general
   Description: Demands highlighting pros and cons...
   From Run: run_general_f1586955 | Domain: general | Started: 2025-11-05 04:21:00
```

**Use Cases:**
- Quick check of what patterns are available
- Verify new artifacts were stored after a run
- See which runs contributed patterns

### 2. Show Complete Database

**File**: `tools/show_all_db.py`

Comprehensive report showing **all data** in the database: artifacts, evolution runs, candidates, and statistics.

```powershell
python tools/show_all_db.py
```

**Sections Included:**
1. **Artifacts** - All patterns, mutations, and champions with full details and provenance
2. **Evolution Runs** - Complete history of optimization sessions
3. **Prompt Candidates** - Individual variations tested (last 50)
4. **Summary Statistics** - Aggregate metrics across the database

**Output Example:**
```
====================================================================
WARMSTART DATABASE - COMPLETE CONTENTS
====================================================================

ARTIFACTS (Prompt DNA Patterns): 30 total
--------------------------------------------------------------------

 1. ID: 30 | Type: PATTERN  | Score: 0.880
    Name: EXAMPLE_PROVIDING
    Content: EXAMPLE_PROVIDING
    From Run: run_general_f1586955 | Domain: general | Started: 2025-11-05
    Description: Includes an example of the expected output...
    Domains: general
    Usage: 0 times | Success: 0
    Created: 2025-11-05 04:21:21

EVOLUTION RUNS (Optimization Sessions): 12 total
--------------------------------------------------------------------

 1. Run ID: run_general_f1586955
    Domain: general | Task: Optimize prompt for general domain...
    Best Training Score: 0.880 | Generation: 0 | Candidates: 4
    Status: completed
    Warm Start: Yes
    Started: 2025-11-05 04:21:00

DATABASE SUMMARY STATISTICS
====================================================================

Artifacts:
  • Total: 30
  • Patterns: 19
  • Mutations: 0
  • Champions: 11
  • Average Effectiveness Score: 0.852

Runs:
  • Total: 12
  • Completed: 12
```

**Use Cases:**
- Audit complete system history
- Debug pattern storage issues
- Analyze optimization performance trends
- Understand provenance (which run created which artifacts)

### 3. Reindex Vector Store

**File**: `tools/reindex_vector.py`

Rebuilds the ChromaDB vector index from SQL database. Useful if vector store becomes out of sync.

```powershell
python tools/reindex_vector.py
```

---

## Project Structure

### Root Directory

```
WarmStart/
├── simple_cli.py           # Main CLI entry point
├── setup.ps1              # Environment setup script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── warmstart.db          # SQLite database (created on first run)
├── chroma_db/            # ChromaDB vector store (created on first run)
├── src/                  # Source code
├── tools/                # Inspection and utility scripts
└── config/               # Configuration files
```

### Source Code: `src/`

```
src/
├── core/                 # Core functionality
│   ├── prompt_challenger.py      # Mutation generator
│   ├── prompt_quality_evaluator.py  # LLM judge evaluator
│   └── llm_client.py            # OpenAI API client
│
├── evolution/            # Evolution engine
│   ├── simple_engine.py         # Main evolution orchestrator
│   ├── population.py            # Population management
│   └── results.py               # Result tracking
│
├── patterns/             # Pattern library (RAG)
│   ├── pattern_library.py       # Vector + SQL storage
│   ├── extractor.py            # Pattern extraction from prompts
│   └── prompt_dna.py           # Pattern definition structures
│
├── models/               # Database models
│   ├── database.py             # Database connection
│   └── schema.py               # SQLAlchemy ORM models
│
└── utils/                # Utilities
    ├── logging.py              # Structured logging (structlog)
    ├── cost_tracker.py         # API cost tracking
    ├── text.py                 # Text utilities (emoji handling)
    └── config.py               # Configuration management
```

### Key Files Explained

#### `simple_cli.py`
- **Purpose**: Main entry point for prompt optimization
- **What it does**:
  - Parses command-line arguments
  - Initializes database and evolution engine
  - Runs optimization loop
  - Displays before/after results with quality breakdown
  - Optionally saves results to JSON

#### `src/evolution/simple_engine.py`
- **Purpose**: Core evolution orchestrator
- **What it does**:
  - Manages warm-start retrieval from pattern library
  - Creates and evaluates population members
  - Performs tournament selection
  - Generates mutations using PromptChallenger
  - Tracks evolution metrics and costs
  - Extracts and persists Prompt DNA after completion
  - Links artifacts to their originating run (provenance)

#### `src/patterns/pattern_library.py`
- **Purpose**: RAG storage and retrieval system
- **What it does**:
  - Stores artifacts in ChromaDB (vector) + SQLite (structured)
  - Semantic search using sentence-transformers embeddings
  - Falls back to SQL LIKE and JSONL substring search if needed
  - Serializes complex metadata for Chroma compatibility
  - Tracks artifact provenance (source_run_id)

#### `src/patterns/extractor.py`
- **Purpose**: Extract reusable patterns from prompts
- **What it does**:
  - Uses LLM to analyze successful prompts
  - Identifies structural patterns (e.g., "CLEAR_TASK_DEFINITION")
  - Identifies mutation operators used
  - Returns structured "Prompt DNA" records
  - Used after evolution to populate the library

#### `src/core/prompt_quality_evaluator.py`
- **Purpose**: LLM judge for prompt quality
- **What it does**:
  - Evaluates prompts on 5 dimensions
  - Returns weighted fitness score (0-1)
  - Provides strengths and weaknesses feedback
  - Used to rank population members

#### `src/core/prompt_challenger.py`
- **Purpose**: Generate prompt mutations
- **What it does**:
  - Takes a prompt and creates variations
  - Uses 8 mutation types (CLARIFICATION, STRUCTURE, etc.)
  - Can incorporate warm-start patterns
  - Used to create new generations

#### `src/models/schema.py`
- **Purpose**: Database schema definitions
- **Defines**:
  - `Artifact`: Patterns, mutations, champions
  - `EvolutionRun`: Metadata about optimization sessions
  - `PromptCandidate`: Individual prompts tested
  - `Generation`: Stats per generation
  - Relationships between tables

---

## Understanding the Output

### Console Output Sections

#### 1. Configuration Summary
```
WarmStart - Quick Prompt Optimization (No Test Cases Required)
Domain: general
====================================================================

Configuration:
  Population: 8
  Generations: 5
  Target Score: 0.90
```

#### 2. Warm-Start Seeding (if enabled)
```
PatternLibrary initialized (ChromaDB)
Retrieved 5 warm-start artifacts:
  • DOMAIN_SPECIFICITY (score: 0.904)
  • CLEAR_TASK_DEFINITION (score: 0.904)
  • OUTPUT_FORMATTING (score: 0.880)
  • CATEGORIZATION_BY_SENTIMENT (score: 0.904)
  • STRUCTURED_OUTPUT_SECTIONS (score: 0.740)
Seeding population with 5 high-quality patterns...
```

**Interpretation:**
- System found 5 relevant past patterns
- Each pattern has proven quality score
- These will be used to seed initial population

#### 3. Evolution Progress
```
====================================================================
🚀 STARTING OPTIMIZATION
====================================================================

📝 YOUR INITIAL PROMPT:
   "Summarize customer feedback"

====================================================================

Generation 0 Evaluation:
  Evaluating 8 candidates...
  ✓ Candidate 1/8: fitness=0.820
  ✓ Candidate 2/8: fitness=0.850
  ...
  Champion: fitness=0.880

Generation 1 Mutation:
  Creating 8 new variations...
  ...
```

**Interpretation:**
- Each generation evaluates all population members
- Best fitness score shown per generation
- System creates mutations and re-evaluates

#### 4. Completion Summary
```
====================================================================
✅ OPTIMIZATION COMPLETE!
====================================================================

📊 Results:
  Generations: 1
  Candidates tested: 8
  Duration: 12.3s
  Total cost: $0.0245
  Stopping reason: Target score reached
```

#### 5. Before vs After Comparison
```
====================================================================
🔴 YOUR ORIGINAL PROMPT:
====================================================================

Summarize customer feedback

====================================================================
🟢 OPTIMIZED PROMPT (Quality Score: 0.904):
====================================================================

Analyze customer feedback and provide a structured summary with the 
following:

1. Key Themes: Identify the top 3-5 recurring topics or concerns
2. Sentiment Breakdown: Categorize feedback as positive, negative, 
   or neutral
3. Actionable Insights: List specific recommendations based on the 
   feedback
4. Priority Items: Highlight urgent issues requiring immediate 
   attention

Output Format: JSON with sections for themes, sentiment_distribution, 
insights, and priority_items.

====================================================================
```

#### 6. Quality Breakdown
```
📈 Quality Breakdown:
  Clarity:       0.95
  Specificity:   0.92
  Structure:     0.90
  Completeness:  0.88
  Effectiveness: 0.87

✅ Strengths:
  • Clear task definition with explicit requirements
  • Well-structured with numbered sections
  • Specifies output format (JSON)
  • Includes categorization by sentiment

⚠️  Areas for Improvement:
  • Could provide example of expected output
  • May benefit from edge case handling guidance
```

**Interpretation:**
- Each dimension scored 0-1
- Strengths highlight what works well
- Weaknesses suggest further improvements

#### 7. Evolution Progress Summary
```
📈 Evolution Progress:
  Gen 0: best=0.820, avg=0.745
  Gen 1: best=0.904, avg=0.823
```

**Interpretation:**
- Shows improvement across generations
- Best score increased from 0.820 to 0.904
- Average quality also improved

#### 8. Pattern Storage Confirmation
```
Stored Prompt DNA artifacts count=6
```

**Interpretation:**
- System extracted 6 reusable patterns
- These will be available for future warm-starts
- Library continuously grows with each run

---

## Database & Artifacts

### SQLite Database: `warmstart.db`

**Tables:**

1. **artifacts** - Reusable prompt patterns
2. **evolution_runs** - Optimization session metadata
3. **prompt_candidates** - Individual prompts tested
4. **generations** - Per-generation statistics

### Artifact Types

#### 1. PATTERN
**What**: Structural patterns identified in successful prompts

**Examples:**
- `CLEAR_TASK_DEFINITION` - Explicitly states what to do
- `OUTPUT_FORMATTING` - Specifies response structure
- `DOMAIN_SPECIFICITY` - Includes domain-specific terminology
- `EXAMPLE_PROVIDING` - Shows expected output examples

**Storage:**
- `artifact_type = "pattern"`
- `content = "PATTERN_NAME"`
- `description = "What this pattern does"`
- Stored with quality score from originating prompt

#### 2. MUTATION
**What**: Transformation operators applied during evolution

**Examples:**
- `CLARIFICATION_IMPROVEMENT`
- `STRUCTURE_ENHANCEMENT`
- `CONTEXT_EXPANSION`

**Storage:**
- `artifact_type = "mutation"`
- Tracks which mutation types worked well
- Used to prioritize mutation strategies

#### 3. CHAMPION
**What**: The best prompt from a completed optimization run

**Storage:**
- `artifact_type = "champion"`
- Full prompt text stored in `content`
- Fitness score in `effectiveness_score`
- Can be reused in future optimizations

### Artifact Metadata

Each artifact includes:

| Field | Description |
|-------|-------------|
| `id` | Primary key |
| `artifact_type` | pattern / mutation / champion |
| `name` | Human-readable name |
| `content` | The actual pattern or prompt text |
| `description` | Explanation of what it does |
| `domain_tags` | List of applicable domains |
| `task_type` | What kind of task it's for |
| `effectiveness_score` | Quality score (0-1) |
| `usage_count` | How many times retrieved |
| `success_count` | How many times led to good results |
| `source_run_id` | Foreign key to EvolutionRun (provenance) |
| `created_at` | Timestamp |

### Provenance Tracking

**What**: Links artifacts back to the optimization run that created them

**How it works:**
1. When evolution starts, system creates `EvolutionRun` record
2. Run gets unique `run_id` (e.g., `run_general_f1586955`)
3. After evolution, patterns are extracted and stored
4. Each artifact's `source_run_id` links to the run's database ID
5. Inspection tools show: "From Run: run_general_f1586955 | Domain: general"

**Benefits:**
- Understand which optimizations contributed patterns
- Audit trail for all artifacts
- Debug pattern quality issues by reviewing source runs

### Evolution Run Metadata

**Table**: `evolution_runs`

**Fields:**
- `run_id`: Unique identifier (e.g., `run_legal_a1b2c3d4`)
- `domain`: Domain name
- `task_description`: What was being optimized
- `status`: `running`, `completed`, `failed`
- `used_warm_start`: Boolean - was RAG used?
- `current_generation`: Last generation completed
- `best_training_score`: Highest fitness achieved
- `total_candidates_evaluated`: Number of prompts tested
- `started_at`: Start timestamp
- `completed_at`: End timestamp

**Use Cases:**
- Track all optimization sessions
- Analyze which domains/tasks benefit most from warm-start
- Calculate average runtime and costs

---

## Advanced Topics

### Prompt DNA: Reusable Knowledge

**Concept**: Just like biological DNA encodes traits, "Prompt DNA" encodes successful prompt characteristics

**Two Components:**

1. **PATTERN**: Structural elements (what the prompt contains)
   - Example: `OUTPUT_FORMATTING` - "Specifies JSON response structure"

2. **MUTATION**: Transformations (how the prompt was improved)
   - Example: `CLARIFICATION_IMPROVEMENT` - "Added specific examples"

**Extraction Process:**
1. After evolution completes, champion prompt is analyzed
2. LLM judge identifies patterns in the successful prompt
3. Patterns stored in library with effectiveness score
4. Future runs retrieve and reuse these patterns

### RAG Strategy: Triple Fallback

**Primary: ChromaDB (Vector Search)**
- Semantic similarity using sentence-transformers
- Finds conceptually related patterns even with different wording
- Best retrieval quality

**Fallback 1: SQL LIKE (Text Matching)**
- If Chroma unavailable or returns no results
- Uses database `LIKE` queries on content/description
- Fast but less intelligent

**Fallback 2: JSONL (Substring Search)**
- If SQL also fails
- Loads pattern_library/prompt_dna.jsonl
- Simple substring matching
- Always works but least sophisticated

**Configuration:**
```python
# In simple_cli.py, RAG is controlled by:
--rag-off         # Disable warm-start completely
--rag-top-k 10    # Retrieve more/fewer patterns (default: 5)
```

### Cost Management

**Tiered LLM Strategy:**

| Tier | Model | Use Case | Cost |
|------|-------|----------|------|
| Tier 1 | GPT-4 | Pattern extraction, critical evaluations | High |
| Tier 2 | GPT-4o-mini | Quality evaluation (judge) | Medium |
| Tier 3 | GPT-4o-mini | Mutations, fast operations | Low |

**Cost Tracking:**
- System tracks tokens and costs per run
- Shows total cost in final summary
- Default max: No hard limit (set in config if needed)

**Optimization Tips:**
- Use smaller populations for testing (--population 4)
- Fewer generations for quick iterations (--generations 3)
- Mock mode for development (--mock)

### Windows Compatibility

**Emoji Issues:**
- PowerShell may not render emojis correctly
- Use `--no-emoji` flag for clean output
- Environment variable: `WARMSTART_NO_EMOJI=1`

**PowerShell Quoting:**
```powershell
# ✅ Correct - use double quotes for prompts with spaces
python simple_cli.py "Your prompt here"

# ❌ Wrong - will fail
python simple_cli.py Your prompt here
```

### Mock Mode (Testing)

**Purpose**: Test system without API costs

```powershell
python simple_cli.py "Test prompt" --mock
```

**Behavior:**
- Returns random quality scores
- No actual OpenAI API calls
- Fast execution for development
- Useful for testing CLI flags and workflow

---

## Troubleshooting

### Common Issues

#### 1. "ChromaDB not available"
- **Cause**: ChromaDB or sentence-transformers installation failed
- **Fix**: Reinstall dependencies: `pip install -r requirements.txt`
- **Workaround**: System will fall back to SQL/JSONL automatically

#### 2. "OpenAI API key not found"
- **Cause**: `.env` file missing or incorrect
- **Fix**: Create `.env` with `OPENAI_API_KEY=sk-...`

#### 3. Low quality scores even after optimization
- **Cause**: Initial prompt too vague or cold start with no patterns
- **Fix**: 
  - Add more context with `--context`
  - Run larger optimization: `--population 15 --generations 10`
  - After first run, warm-start will improve subsequent runs

#### 4. Evolution stops at Generation 0
- **Cause**: Target score reached immediately (good!)
- **Not an error**: System found high-quality prompt quickly
- **Verify**: Check champion fitness score in output

---

## Best Practices

### 1. Provide Good Context
```powershell
# ❌ Vague
python simple_cli.py "Summarize"

# ✅ Specific
python simple_cli.py "Summarize medical reports" --domain medical --context "Hospital discharge summaries for insurance claims"
```

### 2. Match Domain to Task
- Use `--domain legal` for legal documents
- Use `--domain medical` for healthcare tasks
- Use `--domain code` for programming tasks
- Use `--domain general` for everything else

### 3. Build Your Library
- Run optimizations regularly
- System learns from each run
- After 5-10 runs, warm-start becomes very effective

### 4. Iterate and Refine
```powershell
# First run - cold start
python simple_cli.py "Your task" --output run1.json

# Review results, adjust context
python simple_cli.py "Your task" --context "More specific details" --output run2.json

# Now benefits from warm-start with patterns from run1
```

### 5. Inspect Your Library
```powershell
# After several runs, check what patterns were learned
python tools/inspect_library.py

# Full database audit
python tools/show_all_db.py
```

---

## Summary

**WarmStart** is a production-ready prompt optimization system that:

✅ **No test cases needed** - Uses LLM judges  
✅ **Learns from experience** - RAG-powered warm-start  
✅ **Domain-aware** - Optimizes for specific contexts  
✅ **Cost-effective** - Tiered LLM strategy  
✅ **Transparent** - Full provenance tracking  
✅ **Production-ready** - SQL + Vector DB storage  

**Key Files to Remember:**

| Command | Purpose |
|---------|---------|
| `python simple_cli.py "prompt"` | Optimize a prompt |
| `python tools/inspect_library.py` | View pattern library |
| `python tools/show_all_db.py` | Complete database report |

**Start Optimizing:**
```powershell
.\venv\Scripts\Activate.ps1
python simple_cli.py "Your prompt here" --domain general --context "Your use case"
```

---

## Further Reading

- `SIMPLE_MODE.md` - Original design document
- `MUTATION_FEEDBACK.md` - Details on mutation types
- `README.md` - Quick start guide
- Source code in `src/` - Well-documented implementation

---

**Happy Optimizing! 🚀**
