# 🚀 Getting Started with WarmStart

**Complete setup guide for someone starting from scratch**

---

## Prerequisites

Before you start, make sure you have:

### 1. Python 3.11 or Higher
```powershell
# Check your Python version
python --version
```

**Don't have Python?**
- Download from: https://www.python.org/downloads/
- ⚠️ **IMPORTANT**: During installation, check **"Add Python to PATH"**

### 2. OpenAI API Key
- Sign up at: https://platform.openai.com/
- Go to **API Keys** section
- Click **"Create new secret key"**
- Copy the key (starts with `sk-...`)
- You'll need billing set up (costs ~$0.01-0.05 per optimization)

### 3. Terminal/Shell
**Windows:**
- PowerShell or Command Prompt (already included)
- Open: Press `Win + X` → choose "Windows PowerShell" or "Terminal"

**Linux/macOS:**
- Terminal (already included)
- Open: Press `Ctrl + Alt + T` (Linux) or `Cmd + Space` → type "Terminal" (macOS)

### 4. Git (Optional)
- Only needed if cloning from repository
- Download from: https://git-scm.com/downloads/

---

## Step-by-Step Setup

### Step 1: Get the Project Files

**If you already have the folder:**

**Windows:**
```powershell
# Navigate to the project folder
cd C:\Users\YourName\Desktop\WarmStart
```

**Linux/macOS:**
```bash
# Navigate to the project folder
cd ~/Desktop/WarmStart
```

**If downloading from GitHub:**

**Windows:**
```powershell
# Clone the repository
git clone https://github.com/your-repo/WarmStart.git
cd WarmStart
```

**Linux/macOS:**
```bash
# Clone the repository
git clone https://github.com/your-repo/WarmStart.git
cd WarmStart
```

**If you received a ZIP file:**
1. Extract the ZIP to your desired location
2. Open your terminal (PowerShell/Terminal)
3. Navigate to that folder

---

### Step 2: Run the Setup Script

The setup script will create a Python virtual environment and install all dependencies automatically.

**Windows:**
```powershell
# Run the setup script
.\setup.ps1
```

**Linux/macOS:**
```bash
# Make the script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

**What this does:**
- ✅ Creates a virtual environment in `venv/`
- ✅ Installs all required packages (ChromaDB, OpenAI, sentence-transformers, etc.)
- ✅ Verifies the installation

**Windows - If you see a "cannot be loaded" error:**

This is a PowerShell security feature. Run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try `.\setup.ps1` again.

**Linux/macOS - If you see permission errors:**

```bash
# Make sure the script is executable
chmod +x setup.sh

# Or run with bash directly
bash setup.sh
```

---

### Step 3: Configure Your API Key

**Windows:**
```powershell
# Copy the example environment file
copy .env.example .env

# Open it in Notepad
notepad .env
```

**Linux/macOS:**
```bash
# Copy the example environment file
cp .env.example .env

# Open it in your favorite editor
nano .env
# Or: vim .env
# Or: code .env (if you have VS Code)
```

**Edit the file to add your OpenAI API key:**
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important:**
- No spaces around the `=` sign
- No quotes around the key
- Just: `OPENAI_API_KEY=sk-proj-...`

Save the file and close the editor.

**Saving in different editors:**
- Notepad (Windows): `Ctrl+S`, then close
- nano (Linux): `Ctrl+O` (save), `Enter`, `Ctrl+X` (exit)
- vim (Linux): Press `Esc`, type `:wq`, press `Enter`
- VS Code: `Ctrl+S` or `Cmd+S`

---

### Step 4: Activate the Virtual Environment

Every time you open a new terminal, activate the virtual environment:

**Windows:**
```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# You should see (venv) at the start of your prompt:
# (venv) PS C:\...\WarmStart>
```

**If activation fails with an error on Windows:**
```powershell
# Alternative: Use the full path to Python
.\venv\Scripts\python.exe simple_cli.py "test"
```

**Linux/macOS:**
```bash
# Activate the virtual environment
source venv/bin/activate

# You should see (venv) at the start of your prompt:
# (venv) user@machine:~/WarmStart$
```

**If activation fails on Linux/macOS:**
```bash
# Try with explicit bash
bash -c "source venv/bin/activate"

# Or use Python directly
./venv/bin/python simple_cli.py "test"
```

---

### Step 5: Verify Everything Works

**All platforms:**
```bash
# Test that all dependencies are installed correctly
python -c "import chromadb; import openai; import sentence_transformers; print('✓ Success! All dependencies installed.')"
```

**Expected output:** 
```
✓ Success! All dependencies installed.
```

**If you see import errors:**

**Windows:**
```powershell
# Reinstall dependencies
pip install -r requirements.txt

# Or with full path:
.\venv\Scripts\pip.exe install -r requirements.txt
```

**Linux/macOS:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or with full path:
./venv/bin/pip install -r requirements.txt
```

---

### Step 6: Run Your First Optimization! 🎉

Now you're ready to optimize your first prompt:

**All platforms:**
```bash
# Simple test run
python simple_cli.py "Summarize customer feedback" --no-emoji
```

**What to expect:**
- ⏱️ Takes 15-30 seconds
- 💬 Shows evolution progress in real-time
- 📊 Displays before/after prompts with quality scores
- 💰 Costs approximately $0.01-0.03

**This is a "cold start"** - no warm-start patterns available yet (it's your first run!)

**Output will look like:**
```
====================================================================
WarmStart - Quick Prompt Optimization (No Test Cases Required)
Domain: general
====================================================================

Configuration:
  Population: 8
  Generations: 5
  Target Score: 0.90

PatternLibrary initialized (ChromaDB)
Warm-start seeding: no artifacts found
Cold start: Population will start with initial prompt only

====================================================================
🚀 STARTING OPTIMIZATION
====================================================================

Generation 0 Evaluation:
  Evaluating 8 candidates...
  ✓ Candidate 1/8: fitness=0.750
  ✓ Candidate 2/8: fitness=0.820
  ...

====================================================================
✅ OPTIMIZATION COMPLETE!
====================================================================

[Shows your original vs optimized prompt]
```

---

### Step 7: Check What Was Stored

After your first run, the system extracted patterns and stored them in the database:

**Windows:**
```powershell
# View the pattern library
python tools\inspect_library.py
```

**Linux/macOS:**
```bash
# View the pattern library
python tools/inspect_library.py
```

**You should see several patterns**, like:
```
PROMPT PATTERN LIBRARY - Total artifacts: 6

1. CLEAR_TASK_DEFINITION (pattern) - Score: 0.850
   Domains: general
   Description: Specifies the exact task to be performed...

2. OUTPUT_FORMATTING (pattern) - Score: 0.850
   Domains: general
   Description: Dictates the response format...
```

**Congratulations!** These patterns will now be available for warm-start on your next run.

---

### Step 8: Run Again with Warm Start! 🔥

Now run a similar optimization and see warm-start in action:

**All platforms:**
```bash
# Run with a similar prompt
python simple_cli.py "Analyze customer reviews" --no-emoji
```

**This time you should see:**
```
PatternLibrary initialized (ChromaDB)
Retrieved 5 warm-start artifacts:
  • CLEAR_TASK_DEFINITION (score: 0.850)
  • OUTPUT_FORMATTING (score: 0.850)
  • DOMAIN_SPECIFICITY (score: 0.820)
  ...
Seeding population with 5 high-quality patterns...
```

**🎉 You're now using warm-start optimization!**

The system is learning from your previous runs and starting from a better baseline.

---

## Quick Reference Card

After setup, here are the commands you'll use regularly:

**Windows:**
```powershell
# ---- EVERY TIME YOU OPEN A NEW TERMINAL ----
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# ---- OPTIMIZATION ----
# Basic optimization
python simple_cli.py "Your prompt here"

# With domain and context (recommended)
python simple_cli.py "Your prompt" --domain legal --context "Contract analysis"

# Full control
python simple_cli.py "Your prompt" --domain general --population 10 --generations 8 --output results.json

# ---- INSPECTION TOOLS ----
# View pattern library
python tools\inspect_library.py

# View complete database
python tools\show_all_db.py

# ---- WHEN DONE ----
# Deactivate virtual environment
deactivate
```

**Linux/macOS:**
```bash
# ---- EVERY TIME YOU OPEN A NEW TERMINAL ----
# Activate the virtual environment
source venv/bin/activate

# ---- OPTIMIZATION ----
# Basic optimization
python simple_cli.py "Your prompt here"

# With domain and context (recommended)
python simple_cli.py "Your prompt" --domain legal --context "Contract analysis"

# Full control
python simple_cli.py "Your prompt" --domain general --population 10 --generations 8 --output results.json

# ---- INSPECTION TOOLS ----
# View pattern library
python tools/inspect_library.py

# View complete database
python tools/show_all_db.py

# ---- WHEN DONE ----
# Deactivate virtual environment
deactivate
```

---

## Common Parameters

| Flag | Default | Description | Example |
|------|---------|-------------|---------|
| `--domain` | `general` | Domain context | `--domain legal` |
| `--context` | None | Specific use case details | `--context "Real estate contracts"` |
| `--population` | `8` | Prompts per generation | `--population 12` |
| `--generations` | `5` | Max generations | `--generations 10` |
| `--output` | None | Save to JSON file | `--output result.json` |
| `--no-emoji` | False | Windows-safe output | `--no-emoji` |
| `--rag-off` | False | Disable warm-start | `--rag-off` |
| `--rag-top-k` | `5` | Patterns to retrieve | `--rag-top-k 10` |

---

## Troubleshooting

### "Python is not recognized"

**Windows Solution:**
```powershell
# Use full path to Python
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe simple_cli.py "test"

# Or add Python to PATH:
# 1. Search "Environment Variables" in Windows Start menu
# 2. Click "Edit the system environment variables"
# 3. Click "Environment Variables" button
# 4. Under "User variables", select "Path" and click "Edit"
# 5. Click "New" and add Python installation directory
# 6. Click OK on all windows
# 7. Close and reopen PowerShell
```

**Linux/macOS Solution:**
```bash
# Python might be called python3
python3 simple_cli.py "test"

# Or add alias to your shell config
echo "alias python=python3" >> ~/.bashrc  # For bash
echo "alias python=python3" >> ~/.zshrc   # For zsh
source ~/.bashrc  # or ~/.zshrc

# Or install python-is-python3 package (Ubuntu/Debian)
sudo apt install python-is-python3
```

### "pip is not recognized"

**Windows Solution:**
```powershell
# Use python -m pip instead
python -m pip install -r requirements.txt
```

**Linux/macOS Solution:**
```bash
# Use python3 -m pip or pip3
python3 -m pip install -r requirements.txt
# Or
pip3 install -r requirements.txt
```

### "OpenAI API error: Authentication failed"

**All platforms:**
```bash
# Verify your .env file
cat .env  # Linux/macOS
# Or
Get-Content .env  # Windows PowerShell
```

Should show:
```
OPENAI_API_KEY=sk-...
```

**If not, edit it:**

**Windows:**
```powershell
notepad .env
```

**Linux/macOS:**
```bash
nano .env
# Or: vim .env, code .env, etc.
```

**Common mistakes:**
- ❌ `OPENAI_API_KEY = sk-...` (spaces around =)
- ❌ `OPENAI_API_KEY="sk-..."` (quotes)
- ✅ `OPENAI_API_KEY=sk-...` (correct)

### "ChromaDB installation failed" or "Could not build wheels for hnswlib"

**Windows Solution:**

ChromaDB requires C++ build tools on Windows.

**Option 1: Install Visual C++ Build Tools** (Recommended)
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer
3. Select "Desktop development with C++"
4. Install (may take 10-20 minutes)
5. Then reinstall: `pip install chromadb sentence-transformers`

**Option 2: Use pre-built wheels**
```powershell
pip install --upgrade pip setuptools wheel
pip install chromadb sentence-transformers
```

**Linux/macOS Solution:**

Install build dependencies:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3-dev build-essential
pip install chromadb sentence-transformers
```

**Fedora/RHEL/CentOS:**
```bash
sudo yum install -y python3-devel gcc gcc-c++
pip install chromadb sentence-transformers
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Then install packages
pip install chromadb sentence-transformers
```

**Option 3 (All platforms): Run without ChromaDB**

The system will automatically fall back to SQL-based pattern matching:
```bash
# Just run normally - fallback happens automatically
python simple_cli.py "Your prompt"

# You'll see: "ChromaDB not available, using SQL fallback"
```

### "Rate limit exceeded" from OpenAI

**All platforms:**

Your OpenAI account has rate limits based on your tier.

```bash
# Option 1: Use smaller runs
python simple_cli.py "test" --population 4 --generations 2

# Option 2: Use mock mode (no API calls, for testing)
python simple_cli.py "test" --mock

# Option 3: Wait 60 seconds between runs
```

### "Cannot activate virtual environment"

**Windows Solution:**
```powershell
# Alternative activation syntax
& .\venv\Scripts\Activate.ps1

# Or just use Python directly without activation
.\venv\Scripts\python.exe simple_cli.py "test"
```

**Linux/macOS Solution:**
```bash
# Try with explicit bash/zsh
bash -c "source venv/bin/activate"

# Or use Python directly without activation
./venv/bin/python simple_cli.py "test"
```

### "Permission denied" (Linux/macOS)

**Linux/macOS Solution:**
```bash
# Make sure scripts are executable
chmod +x setup.sh
chmod +x venv/bin/activate

# If you get permission errors during pip install
pip install --user -r requirements.txt
```

### Low quality scores or no improvement

**Solution:**

The system needs better input or more context:

```powershell
# ❌ Too vague
python simple_cli.py "Analyze data"

# ✅ Specific with context
python simple_cli.py "Analyze customer churn data" --context "SaaS product, focus on behavioral patterns and usage metrics" --domain general

# ✅ More generations for thorough optimization
python simple_cli.py "Your prompt" --population 12 --generations 8
```

---

## Next Steps

### 1. Read the Full Documentation
```powershell
# Open in VS Code and press Ctrl+Shift+V for preview
code DOCUMENTATION.md
```

The full documentation covers:
- How cold start vs warm start works
- Detailed explanation of mutations and tournaments
- Understanding the output
- Advanced topics

### 2. Try Different Domains

```powershell
# Legal domain
python simple_cli.py "Extract contract terms" --domain legal

# Medical domain
python simple_cli.py "Summarize patient records" --domain medical

# Code domain
python simple_cli.py "Review code security" --domain code
```

### 3. Experiment with Parameters

```powershell
# Fast iteration (testing)
python simple_cli.py "test" --population 4 --generations 3

# Balanced (recommended)
python simple_cli.py "test" --population 8 --generations 5

# Thorough (maximum quality)
python simple_cli.py "test" --population 15 --generations 10
```

### 4. Build Your Pattern Library

Run optimizations regularly - the more you use it, the better it gets!

Each run adds patterns to the library, improving future optimizations.

---

## Getting Help

### View Built-in Help
```powershell
python simple_cli.py --help
```

### Check Database Contents
```powershell
# Quick view of patterns
python tools\inspect_library.py

# Complete database dump
python tools\show_all_db.py
```

### Test Without API Costs
```powershell
# Mock mode (no real API calls)
python simple_cli.py "test" --mock --no-emoji
```

---

## Success Checklist

After setup, you should be able to:

- ✅ Activate virtual environment: `.\venv\Scripts\Activate.ps1`
- ✅ Run optimization: `python simple_cli.py "test" --no-emoji`
- ✅ See "OPTIMIZATION COMPLETE!" message
- ✅ View patterns: `python tools\inspect_library.py`
- ✅ See warm-start on second run: "Retrieved X warm-start artifacts"

**If all checked, you're ready to go! 🎉**

---

## Quick Example Session

Here's a complete example session from start to finish:

**Windows:**
```powershell
# 1. Navigate to project
cd C:\Users\YourName\Desktop\WarmStart

# 2. Activate environment
.\venv\Scripts\Activate.ps1

# 3. Run first optimization (cold start)
python simple_cli.py "Summarize meeting notes" --domain general --no-emoji

# Expected: Takes 20-30 seconds, cold start (no artifacts found)

# 4. Check what was stored
python tools\inspect_library.py

# Expected: Shows 5-6 patterns extracted

# 5. Run similar optimization (warm start!)
python simple_cli.py "Create meeting summary" --domain general --no-emoji

# Expected: "Retrieved 5 warm-start artifacts..." - faster, better baseline!

# 6. View complete history
python tools\show_all_db.py

# Expected: Shows 2 runs, ~10-12 artifacts total

# 7. When done
deactivate
```

**Linux/macOS:**
```bash
# 1. Navigate to project
cd ~/Desktop/WarmStart

# 2. Activate environment
source venv/bin/activate

# 3. Run first optimization (cold start)
python simple_cli.py "Summarize meeting notes" --domain general --no-emoji

# Expected: Takes 20-30 seconds, cold start (no artifacts found)

# 4. Check what was stored
python tools/inspect_library.py

# Expected: Shows 5-6 patterns extracted

# 5. Run similar optimization (warm start!)
python simple_cli.py "Create meeting summary" --domain general --no-emoji

# Expected: "Retrieved 5 warm-start artifacts..." - faster, better baseline!

# 6. View complete history
python tools/show_all_db.py

# Expected: Shows 2 runs, ~10-12 artifacts total

# 7. When done
deactivate
```

---

**You're all set! Start optimizing your prompts! 🚀**

For detailed documentation, see `DOCUMENTATION.md`.
