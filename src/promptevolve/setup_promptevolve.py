#!/usr/bin/env python3
"""
PromptEvolve Setup and Validation Script

This script helps set up the PromptEvolve environment and validates
that all components are correctly installed and configured.
"""

import os
import sys
from pathlib import Path


def print_header(message):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {message}")
    print("="*80 + "\n")


def print_status(message, status="INFO"):
    """Print a status message."""
    icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌"
    }
    icon = icons.get(status, "ℹ️")
    print(f"{icon}  {message}")


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} detected", "SUCCESS")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor} detected - requires Python 3.9+", "ERROR")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print_header("Checking Dependencies")
    
    required_packages = [
        'pydantic',
        'yaml',
        'openai',
        'pycontractor'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"{package} is installed", "SUCCESS")
        except ImportError:
            print_status(f"{package} is NOT installed", "ERROR")
            missing.append(package)
    
    if missing:
        print_status(f"Missing packages: {', '.join(missing)}", "ERROR")
        print_status("Install with: uv pip install -e .", "INFO")
        return False
    
    return True


def check_api_key():
    """Check if OpenAI API key is configured."""
    print_header("Checking OpenAI API Key")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
        print_status(f"API key found: {masked_key}", "SUCCESS")
        return True
    else:
        print_status("OPENAI_API_KEY environment variable not set", "ERROR")
        print_status("Set it with: export OPENAI_API_KEY='your-key-here'", "INFO")
        return False


def create_directories():
    """Create necessary directory structure."""
    print_header("Creating Directory Structure")
    
    directories = [
        'logs',
        'data/synthetic',
        'data/history',
        'data/test_scenarios',
        'prompts',
        'results'
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_status(f"Created: {directory}", "SUCCESS")
        else:
            print_status(f"Already exists: {directory}", "INFO")
    
    return True


def validate_config():
    """Validate the configuration file."""
    print_header("Validating Configuration")
    
    config_path = Path('config.yaml')
    if not config_path.exists():
        print_status("config.yaml not found", "ERROR")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check essential sections
        essential_sections = ['models', 'evolution', 'thresholds', 'domains']
        for section in essential_sections:
            if section in config:
                print_status(f"Section '{section}' found", "SUCCESS")
            else:
                print_status(f"Section '{section}' missing", "ERROR")
                return False
        
        return True
        
    except Exception as e:
        print_status(f"Error parsing config.yaml: {e}", "ERROR")
        return False


def run_basic_tests():
    """Run basic import tests."""
    print_header("Running Basic Tests")
    
    try:
        from prompt_evolution import (
            PromptEvolution,
            PromptChallenger,
            PromptSolver,
            PromptVerifier,
            EvolutionEngine,
            TestScenario,
            PromptResult
        )
        print_status("Core components imported successfully", "SUCCESS")
        
        from data_generator import SyntheticDataGenerator, GenerationRequest
        print_status("Data generator imported successfully", "SUCCESS")
        
        # Test creating basic data structures
        scenario = TestScenario(
            input_message="Test input",
            desired_output="Test output"
        )
        print_status("TestScenario creation works", "SUCCESS")
        
        return True
        
    except Exception as e:
        print_status(f"Import test failed: {e}", "ERROR")
        return False


def run_unit_tests():
    """Run the full unit test suite."""
    print_header("Running Unit Tests")
    
    try:
        import test_evolution_engine
        success = test_evolution_engine.run_tests()
        
        if success:
            print_status("All unit tests passed", "SUCCESS")
        else:
            print_status("Some unit tests failed", "WARNING")
        
        return success
        
    except Exception as e:
        print_status(f"Unit test execution failed: {e}", "ERROR")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_header("Next Steps")
    
    print("1. Review and customize config.yaml for your use case")
    print("2. Generate synthetic test data:")
    print("   python data_generator.py")
    print()
    print("3. Run a basic evolution test:")
    print("   python prompt_evolution.py")
    print()
    print("4. Integrate with your Task Definer System:")
    print("   from prompt_evolution import PromptEvolution")
    print()
    print("5. Read the full documentation:")
    print("   cat README_PROMPTEVOLVE.md")
    print()


def main():
    """Main setup and validation routine."""
    print_header("PromptEvolve Setup & Validation")
    
    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "API Key": check_api_key(),
        "Directories": create_directories(),
        "Configuration": validate_config(),
        "Basic Tests": run_basic_tests()
    }
    
    # Summary
    print_header("Setup Summary")
    
    all_passed = True
    for check, passed in results.items():
        status = "SUCCESS" if passed else "ERROR"
        print_status(f"{check}: {'PASSED' if passed else 'FAILED'}", status)
        if not passed:
            all_passed = False
    
    if all_passed:
        print_status("\n🎉 Setup complete! PromptEvolve is ready to use.", "SUCCESS")
        
        # Optionally run unit tests
        print("\nWould you like to run the full unit test suite? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response == 'y':
                run_unit_tests()
        except (KeyboardInterrupt, EOFError):
            print("\nSkipping unit tests.")
        
        print_next_steps()
        return 0
    else:
        print_status("\n⚠️  Setup incomplete. Please fix the errors above.", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
