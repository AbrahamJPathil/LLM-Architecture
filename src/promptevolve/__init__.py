"""
PromptEvolve: Self-Improving Prompt Engineering System
========================================================

A sophisticated prompt optimization framework using R-Zero co-evolutionary algorithms.

Main Components:
- PromptEvolution: Main evolution engine
- PromptChallenger: Generates prompt variations
- PromptSolver: Executes prompts against test scenarios
- PromptVerifier: Evaluates and scores prompt performance
- TaskDefiner: Defines tasks and generates test scenarios

Usage:
    from promptevolve import PromptEvolution, TestScenario
    
    system = PromptEvolution(config_path="../config/config.yaml")
    scenarios = [TestScenario(...)]
    result = system.evolve_prompt(base_prompt="...", test_scenarios=scenarios)
"""

__version__ = "0.1.0"
__author__ = "Abraham J Pathil"

from .prompt_evolution import (
    PromptEvolution,
    TestScenario,
    PromptResult,
    PromptState,
    DomainConfig,
)

__all__ = [
    "PromptEvolution",
    "TestScenario",
    "PromptResult",
    "PromptState",
    "DomainConfig",
]
