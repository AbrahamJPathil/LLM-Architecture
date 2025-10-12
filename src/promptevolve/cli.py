"""
PromptEvolve CLI - Command-line interface for prompt evolution.

A modern, type-safe CLI using Typer for the PromptEvolve system.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from promptevolve.task_analyzer import SimpleTaskAnalyzer, TaskAnalysis
from promptevolve.scenario_generator import ScenarioGenerator
from promptevolve.prompt_enhancer import PromptEnhancer
from promptevolve.prompt_evolution import PromptEvolution, TestScenario as EvolutionTestScenario

# Initialize Typer app and Rich console
app = typer.Typer(
    name="promptevolve",
    help="🚀 PromptEvolve - AI-Powered Prompt Optimization System",
    add_completion=True,
)
console = Console()


def check_api_key() -> str:
    """Check if OpenAI API key is set, exit with helpful message if not."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("\n[bold red]❌ Error: OPENAI_API_KEY not set[/bold red]\n")
        console.print("Please set your OpenAI API key:\n")
        console.print("  1. Create/edit .env file:")
        console.print("     [cyan]cp .env.example .env[/cyan]")
        console.print("     [cyan]# Add: OPENAI_API_KEY=sk-proj-...[/cyan]\n")
        console.print("  2. Or export directly:")
        console.print("     [cyan]export OPENAI_API_KEY='your-key-here'[/cyan]\n")
        raise typer.Exit(1)
    return api_key


@app.command()
def define(
    prompt: str = typer.Argument(..., help="The vague prompt to enhance"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save enhanced prompt to file"),
):
    """
    Define and enhance a vague prompt into a complete system prompt.
    
    This command uses TaskDefiner logic to:
    1. Analyze your vague input
    2. Fill gaps with intelligent defaults
    3. Add domain knowledge and best practices
    4. Generate a complete, structured system prompt
    
    Use this to preview what will be fed into evolution, or to get
    a ready-to-use prompt without running the full evolution process.
    """
    api_key = check_api_key()
    
    console.print("\n[bold cyan]📋 Task Definition & Prompt Enhancement[/bold cyan]\n")
    console.print("=" * 60)
    
    # Step 1: Analyze
    console.print("\n[bold]Step 1:[/bold] Analyzing your input...")
    with console.status("[bold green]Analyzing...", spinner="dots"):
        analyzer = SimpleTaskAnalyzer(api_key=api_key)
        task_analysis = analyzer.analyze_prompt(prompt)
    
    console.print(f"   Domain: [cyan]{task_analysis.domain}[/cyan]")
    console.print(f"   Complexity: [cyan]{task_analysis.complexity}[/cyan]")
    console.print(f"   Objectives identified: [cyan]{len(task_analysis.objectives)}[/cyan]")
    
    # Step 2: Enhance
    console.print("\n[bold]Step 2:[/bold] Enhancing prompt with domain knowledge...")
    with console.status("[bold green]Creating enhanced prompt...", spinner="dots"):
        enhancer = PromptEnhancer(api_key=api_key)
        enhanced_prompt = enhancer.enhance_prompt(task_analysis)
    
    # Display result
    console.print()
    console.print(Panel(
        f"[bold]{enhanced_prompt}[/bold]",
        title="[bold cyan]✨ ENHANCED INITIAL PROMPT[/bold cyan]",
        border_style="cyan",
        padding=(1, 2)
    ))
    
    # Stats
    console.print(f"\n[dim]Original: {len(prompt)} chars → Enhanced: {len(enhanced_prompt)} chars[/dim]")
    console.print(f"[dim]Improvement: +{len(enhanced_prompt) - len(prompt)} characters[/dim]")
    
    # Save if requested
    if output:
        output_path = PROJECT_ROOT / output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"# Enhanced Prompt\n\n")
            f.write(f"**Original Input:** {prompt}\n\n")
            f.write(f"**Domain:** {task_analysis.domain}\n")
            f.write(f"**Complexity:** {task_analysis.complexity}\n\n")
            f.write(f"---\n\n")
            f.write(enhanced_prompt)
        
        console.print(f"\n💾 Enhanced prompt saved to: [cyan]{output_path}[/cyan]")
    
    console.print("\n[dim]💡 Tip: Use 'promptevolve evolve' to optimize this further![/dim]\n")


@app.command()
def analyze(
    prompt: str = typer.Argument(..., help="The prompt to analyze"),
):
    """
    Analyze a prompt and generate task definition without evolution.
    
    [DEPRECATED] Use 'define' command instead for better results.
    This command is kept for backwards compatibility.
    """
    console.print("[yellow]⚠️  Note: 'analyze' is deprecated. Use 'define' for enhanced prompts.[/yellow]\n")
    
    api_key = check_api_key()
    
    console.print("\n[bold cyan]📊 Task Analysis[/bold cyan]\n")
    console.print("=" * 60)
    
    # Initialize analyzer
    with console.status("[bold green]Analyzing prompt...", spinner="dots"):
        analyzer = SimpleTaskAnalyzer(api_key=api_key)
        task_analysis = analyzer.analyze_prompt(prompt)
    
    # Display analysis results
    console.print(f"\n[bold]Domain:[/bold] {task_analysis.domain}")
    console.print(f"[bold]Complexity:[/bold] {task_analysis.complexity}")
    console.print(f"\n[bold]Objectives:[/bold]")
    for i, obj in enumerate(task_analysis.objectives, 1):
        console.print(f"  {i}. {obj}")
    
    if task_analysis.constraints:
        console.print(f"\n[bold]Constraints:[/bold]")
        for constraint in task_analysis.constraints:
            console.print(f"  • {constraint}")
    
    if task_analysis.required_skills:
        console.print(f"\n[bold]Required Skills:[/bold]")
        for skill in task_analysis.required_skills:
            console.print(f"  • {skill}")
    
    console.print(f"\n[bold]Context:[/bold]\n{task_analysis.context}")
    
    # Generate scenarios
    console.print("\n[bold cyan]🎯 Generating Test Scenarios[/bold cyan]\n")
    
    with console.status("[bold green]Creating scenarios...", spinner="dots"):
        generator = ScenarioGenerator(api_key=api_key)
        scenarios = generator.generate_from_task(task_analysis)
    
    console.print(f"Generated {len(scenarios)} test scenarios:\n")
    
    for i, scenario in enumerate(scenarios, 1):
        table = Table(title=f"Scenario {i}: {scenario.description}", show_header=False)
        table.add_row("[green]✓ Desired Output[/green]", scenario.desired_output[:150] + "...")
        table.add_row("[red]✗ Bad Output[/red]", scenario.bad_output[:150] + "...")
        console.print(table)
        console.print()


@app.command()
def evolve(
    prompt: str = typer.Argument(..., help="The vague prompt to enhance and evolve"),
    iterations: int = typer.Option(10, "--iterations", "-i", help="Maximum evolution iterations"),
    population: int = typer.Option(5, "--population", "-p", help="Population size for evolution"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (JSON)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
):
    """
    Full optimization pipeline: define → enhance → evolve.
    
    This command:
    1. Analyzes your vague input (domain, objectives, constraints)
    2. Generates an enhanced, complete system prompt
    3. Creates test scenarios automatically
    4. Evolves the enhanced prompt using genetic algorithm
    5. Returns the final optimized, production-ready prompt
    
    This is the main command you should use for prompt optimization.
    """
    api_key = check_api_key()
    
    # Header
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🚀 PromptEvolve - Full Optimization Pipeline[/bold cyan]",
        border_style="cyan"
    ))
    console.print()
    
    # Step 1: Analyze
    console.print("[bold]📊 Step 1:[/bold] Analyzing your input...")
    
    with console.status("[bold green]Analyzing...", spinner="dots"):
        analyzer = SimpleTaskAnalyzer(api_key=api_key)
        task_analysis = analyzer.analyze_prompt(prompt)
    
    console.print(f"   Domain: [cyan]{task_analysis.domain}[/cyan]")
    console.print(f"   Complexity: [cyan]{task_analysis.complexity}[/cyan]")
    console.print(f"   Objectives: [cyan]{len(task_analysis.objectives)}[/cyan]")
    
    # Step 2: Enhance
    console.print("\n[bold]✨ Step 2:[/bold] Enhancing prompt with domain knowledge...")
    
    with console.status("[bold green]Creating enhanced prompt...", spinner="dots"):
        enhancer = PromptEnhancer(api_key=api_key)
        enhanced_prompt = enhancer.enhance_prompt(task_analysis)
    
    console.print(f"   Enhanced prompt: [cyan]{len(enhanced_prompt)} characters[/cyan]")
    
    # Always show enhanced prompt so user sees what's being optimized
    console.print()
    console.print(Panel(
        f"{enhanced_prompt}",
        title="[bold cyan]✨ ENHANCED INITIAL PROMPT[/bold cyan]",
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print()
    
    # Step 3: Generate scenarios
    console.print("\n[bold]🎯 Step 3:[/bold] Generating test scenarios...")
    
    with console.status("[bold green]Creating scenarios...", spinner="dots"):
        generator = ScenarioGenerator(api_key=api_key)
        scenarios = generator.generate_from_task(task_analysis)
    
    console.print(f"   Created [cyan]{len(scenarios)}[/cyan] test scenarios")
    
    # Prepare scenarios for evolution (convert to PromptEvolution format)
    test_scenarios_evolution = []
    for scenario in scenarios:
        # Convert our ScenarioGenerator format to PromptEvolution TestScenario format
        evolution_scenario = EvolutionTestScenario(
            input_message=scenario.input_data.get('query', scenario.description),
            existing_memories="",
            desired_output=scenario.desired_output,
            bad_output=scenario.bad_output,
            metadata=scenario.input_data
        )
        test_scenarios_evolution.append(evolution_scenario)
    
    # Step 4: Evolution
    console.print("\n[bold]🧬 Step 4:[/bold] Evolving the enhanced prompt...")
    console.print(f"   Running up to [cyan]{iterations}[/cyan] iterations with population size [cyan]{population}[/cyan]")
    console.print("   [dim]This may take a few minutes...[/dim]\n")
    
    config_path = str(PROJECT_ROOT / "config" / "config.yaml")
    evolution_system = PromptEvolution(config_path=config_path)
    
    # Override config with CLI parameters
    evolution_system.config['evolution']['max_iterations'] = iterations
    evolution_system.config['evolution']['population_size'] = population
    
    try:
        # Feed ENHANCED prompt into evolution (not the original vague input!)
        results = evolution_system.evolve_prompt(
            base_prompt=enhanced_prompt,  # ← Enhanced prompt as the starting point
            test_scenarios=test_scenarios_evolution
        )
        
        # Step 5: Display results
        console.print()
        console.print(Panel.fit(
            "✨ [bold green]Evolution Complete![/bold green]",
            border_style="green"
        ))
        console.print()
        
        console.print(Panel(
            f"[bold]{results.current_prompt}[/bold]",
            title="[bold cyan]🏆 FINAL OPTIMIZED PROMPT[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        # Metrics table
        metrics = Table(show_header=False, box=None)
        
        # Access PromptState attributes
        if results.results:  # Check if PromptResult is available
            composite_score = results.results.get_composite_score()
            metrics.add_row("[bold]Score:[/bold]", f"[green]{composite_score:.4f}[/green]")
            metrics.add_row("[bold]Success Rate:[/bold]", f"[cyan]{results.results.success_rate:.2%}[/cyan]")
            metrics.add_row("[bold]Quality Score:[/bold]", f"[cyan]{results.results.quality_score:.4f}[/cyan]")
        
        metrics.add_row("[bold]Generations:[/bold]", f"[cyan]{results.generation}[/cyan]")
        metrics.add_row("[bold]Termination Reason:[/bold]", f"[yellow]{results.termination_reason or 'Max iterations'}[/yellow]")
        console.print(metrics)
        console.print()
        
        # Prepare full output
        output_data = {
            "original_input": prompt,
            "task_analysis": {
                "domain": task_analysis.domain,
                "complexity": task_analysis.complexity,
                "objectives": task_analysis.objectives,
                "constraints": task_analysis.constraints,
                "context": task_analysis.context
            },
            "enhanced_prompt": enhanced_prompt,
            "test_scenarios": [
                {
                    "input_message": s.input_message,
                    "desired_output": s.desired_output,
                    "bad_output": s.bad_output,
                    "metadata": s.metadata
                }
                for s in test_scenarios_evolution
            ],
            "evolution_results": {
                "final_prompt": results.current_prompt,
                "generation": results.generation,
                "termination_reason": results.termination_reason,
                "changelog": results.changelog,
                "results": {
                    "success_rate": results.results.success_rate if results.results else 0.0,
                    "quality_score": results.results.quality_score if results.results else 0.0,
                    "composite_score": results.results.get_composite_score() if results.results else 0.0
                } if results.results else None
            }
        }
        
        # Save to file if requested
        if output:
            output_path = PROJECT_ROOT / output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            console.print(f"💾 Full results saved to: [cyan]{output_path}[/cyan]\n")
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error during evolution:[/bold red] {e}\n")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print("\n[bold cyan]PromptEvolve[/bold cyan] version [green]0.1.0[/green]")
    console.print("Self-Improving Prompt Engineering Agent\n")


if __name__ == "__main__":
    app()
