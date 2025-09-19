#!/usr/bin/env python3
"""
Main entry point for the Agentic Research Framework.

This provides both CLI and web interfaces for running experiments
and testing the system design use case.
"""

import asyncio
import typer
from pathlib import Path
from typing import Optional
import structlog
from rich.console import Console
from rich.table import Table

from config.settings import settings
from research.experiment_orchestrator import GenericExperimentRunner, ExperimentConfig
from use_cases.system_design.orchestrator import SystemDesignOrchestrator
from use_cases.system_design.config import USE_CASE_CONFIG, TEST_SCENARIOS

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
console = Console()

app = typer.Typer(
    name="research-framework",
    help="Agentic Research Framework for testing reflection vs model capability",
    add_completion=False
)

@app.command()
def serve_api(
    host: str = typer.Option("0.0.0.0", help="Host to bind the API server"),
    port: int = typer.Option(8000, help="Port to bind the API server"),
    reload: bool = typer.Option(True, help="Enable auto-reload for development")
):
    """
    Start the FastAPI server for research framework.
    
    This provides REST API endpoints for frontend integration.
    """
    import uvicorn
    console.print(f"[bold blue]Starting Research Framework API Server[/bold blue]")
    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    console.print(f"Docs: http://{host}:{port}/docs")
    console.print()
    
    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except Exception as e:
        console.print(f"[red]Failed to start API server: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def test_system_design(
    scenario: str = typer.Option(
        "simple", 
        help="Test scenario: simple, medium, or complex"
    ),
    model: str = typer.Option(
        "gemini-2.5-flash-lite",
        help="Model to use for testing"
    ),
    reflection_iterations: int = typer.Option(
        0,
        help="Number of reflection iterations (0 = baseline)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """
    Test the system design use case with a single scenario.
    
    This is useful for development and debugging before running full experiments.
    """
    if verbose:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))
    
    console.print(f"[bold blue]Testing System Design Use Case[/bold blue]")
    console.print(f"Scenario: {scenario}")
    console.print(f"Model: {model}")
    console.print(f"Reflection iterations: {reflection_iterations}")
    console.print()
    
    # Get test scenario
    if scenario in TEST_SCENARIOS:
        test_input = TEST_SCENARIOS[scenario][0]["input"]
    else:
        console.print(f"[red]Unknown scenario: {scenario}[/red]")
        console.print(f"Available scenarios: {list(TEST_SCENARIOS.keys())}")
        return
    
    # Run test
    asyncio.run(_run_system_design_test(test_input, model, reflection_iterations))

async def _run_system_design_test(test_input: str, model: str, reflection_iterations: int):
    """Run a single system design test."""
    try:
        # Initialize orchestrator
        orchestrator = SystemDesignOrchestrator(USE_CASE_CONFIG)
        
        # Create appropriate pipeline
        if reflection_iterations == 0:
            pipeline = orchestrator.create_baseline_pipeline(model)
            console.print("[yellow]Running baseline pipeline (no reflection)[/yellow]")
        else:
            pipeline = orchestrator.create_reflection_pipeline(
                producer_model=model,
                max_iterations=reflection_iterations
            )
            console.print(f"[green]Running reflection pipeline ({reflection_iterations} max iterations)[/green]")
        
        console.print("\n[bold]Input:[/bold]")
        console.print(test_input)
        console.print("\n[bold]Processing...[/bold]")
        
        # Execute pipeline
        result = await pipeline.run({"input": test_input})
        
        # Display results
        console.print("\n[bold green]Results:[/bold green]")
        if isinstance(result, dict):
            for key, value in result.items():
                console.print(f"[blue]{key}:[/blue] {str(value)[:200]}...")
        else:
            console.print(str(result))
            
        console.print(f"\n[bold]Test completed successfully![/bold]")
        
    except Exception as e:
        console.print(f"[red]Test failed: {str(e)}[/red]")
        logger.error("System design test failed", error=str(e))
        raise typer.Exit(1)

@app.command()
def run_experiment(
    config_file: Path = typer.Argument(
        ...,
        help="Path to experiment configuration JSON file",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Override output directory for results"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """
    Run a complete research experiment from configuration file.
    
    This executes the full experimental protocol for comparing
    reflection vs model capability.
    """
    if verbose:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))
    
    console.print(f"[bold blue]Running Research Experiment[/bold blue]")
    console.print(f"Configuration: {config_file}")
    if output_dir:
        console.print(f"Output directory: {output_dir}")
    console.print()
    
    # Run experiment
    asyncio.run(_run_research_experiment(config_file, output_dir))

async def _run_research_experiment(config_file: Path, output_dir: Optional[Path]):
    """Run a complete research experiment."""
    try:
        import json
        
        # Load configuration
        with open(config_file, "r") as f:
            config_data = json.load(f)
        
        config = ExperimentConfig(**config_data)
        if output_dir:
            config.output_dir = str(output_dir)
        
        # Initialize experiment runner
        runner = GenericExperimentRunner()
        
        console.print(f"[yellow]Starting experiment: {config.experiment_id}[/yellow]")
        console.print(f"Use case: {config.use_case}")
        console.print(f"Models: {config.models_to_test}")
        console.print(f"Reflection configs: {config.reflection_configs}")
        console.print(f"Scenarios: {len(config.test_scenarios)}")
        console.print(f"Repetitions: {config.repetitions}")
        console.print()
        
        # Run experiment
        results = await runner.run_experiment(config)
        
        # Display summary
        console.print("\n[bold green]Experiment Completed![/bold green]")
        console.print(f"Total experiments: {len(results['results'])}")
        
        # Show summary table
        summary = results['summary']
        if 'by_model' in summary:
            table = Table(title="Results by Model")
            table.add_column("Model", style="cyan")
            table.add_column("Mean Quality", style="green")
            table.add_column("Count", style="blue")
            
            for model, stats in summary['by_model'].items():
                table.add_row(
                    model,
                    f"{stats['mean']:.3f}",
                    str(stats['count'])
                )
            
            console.print(table)
        
        console.print(f"\nResults saved to: {results.get('output_dir', 'default location')}")
        
    except Exception as e:
        console.print(f"[red]Experiment failed: {str(e)}[/red]")
        logger.error("Research experiment failed", error=str(e))
        raise typer.Exit(1)

@app.command()
def list_scenarios():
    """List available test scenarios for system design."""
    console.print("[bold blue]Available Test Scenarios[/bold blue]\n")
    
    for complexity, scenarios in TEST_SCENARIOS.items():
        console.print(f"[bold]{complexity.upper()}[/bold]")
        for scenario in scenarios:
            console.print(f"  ID: {scenario['id']}")
            console.print(f"  Input: {scenario['input'][:100]}...")
            console.print()

@app.command()
def check_config():
    """Check framework configuration and dependencies."""
    console.print("[bold blue]Framework Configuration Check[/bold blue]\n")
    
    # Check API key
    if settings.google_api_key:
        console.print("[green]✓[/green] Google API key configured")
    else:
        console.print("[red]✗[/red] Google API key not found")
    
    # Check models
    console.print(f"[blue]Available models:[/blue] {', '.join(settings.available_models)}")
    console.print(f"[blue]Default model:[/blue] {settings.default_model}")
    console.print(f"[blue]Pro model:[/blue] {settings.pro_model}")
    
    # Check directories
    settings.experiment_output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[blue]Output directory:[/blue] {settings.experiment_output_dir}")
    
    # Check research mode
    if settings.enable_research_mode:
        console.print("[green]✓[/green] Research mode enabled")
    else:
        console.print("[yellow]![/yellow] Research mode disabled")
    
    console.print("\n[bold green]Configuration check completed![/bold green]")

def cli_app():
    """Entry point for CLI application."""
    app()

if __name__ == "__main__":
    cli_app()

