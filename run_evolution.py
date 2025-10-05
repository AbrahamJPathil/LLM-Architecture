#!/usr/bin/env python3
"""
Main entry point for PromptEvolve
Run this script from the project root to start prompt evolution
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from promptevolve.prompt_evolution import main

if __name__ == "__main__":
    main()
