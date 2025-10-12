#!/usr/bin/env python3
"""
PromptEvolve CLI Wrapper

Simple wrapper to make the CLI easily accessible.
Run: python promptevolve.py evolve "your prompt"
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from promptevolve.cli import main

if __name__ == '__main__':
    main()
