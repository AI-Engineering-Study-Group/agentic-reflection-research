import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import structlog
import importlib
from framework.base_orchestrator import BaseUseCaseOrchestrator
from framework.base_evaluator import BaseUseCaseEvaluator
from config.settings import settings

logger = structlog.get_logger(__name__)

@dataclass
class ExperimentConfig:
    """
    Configuration for a research experiment.
    
    Why structured configuration:
    1. Ensures reproducible experiments
    2. Enables systematic parameter sweeps
    3. Provides clear documentation of experimental conditions
    4. Supports research paper methodology section
    """
    experiment_id: str
    use_case: str
    test_scenarios: List[str]
    models_to_test: List[str]
    reflection_configs: List[int]  # [0, 2, 5] = baseline, light reflection, deep reflection
    complexity_levels: List[str]
    repetitions: int = 1  # For statistical significance
    enable_expert_evaluation: bool = False
    output_dir: Optional[str] = None

@dataclass
class ExperimentResult:
    """Structured experiment result for analysis."""
    experiment_id: str
    use_case: str
    scenario: str
    model: str
    reflection_iterations: int
    complexity_level: str
    repetition: int
    
    # Performance metrics
    execution_time_seconds: float
    token_usage: Dict[str, int]
    api_cost_usd: float
    
    # Quality metrics
    quality_scores: List[Dict[str, Any]]
    overall_quality_score: float
    
    # Output data
    agent_output: Dict[str, Any]
    evaluation_details: Dict[str, Any]
    
    timestamp: str

class GenericExperimentRunner:
    """
    Runs systematic experiments across any use case.
    
    This is the core research infrastructure that enables the
    "Iterative Reflection vs. Model Capability" study.
    """
    
    def __init__(self):
        self.results: List[ExperimentResult] = []
        self.use_case_cache = {}
        self.cost_tracker = CostTracker()
        
        # Ensure output directory exists
        self.base_output_dir = Path(settings.experiment_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Experiment runner initialized",
            output_dir=str(self.base_output_dir)
        )
    
    def load_use_case(self, use_case_name: str) -> tuple[BaseUseCaseOrchestrator, BaseUseCaseEvaluator]:
        """
        Dynamically load use case components.
        
        Why dynamic loading:
        1. Enables easy addition of new use cases
        2. Keeps framework core independent of specific use cases
        3. Allows use case-specific optimizations
        4. Supports modular development and testing
        """
        if use_case_name in self.use_case_cache:
            return self.use_case_cache[use_case_name]
        
        try:
            # Dynamic import of use case modules
            orchestrator_module = importlib.import_module(f"use_cases.{use_case_name}.orchestrator")
            evaluator_module = importlib.import_module(f"use_cases.{use_case_name}.evaluator")
            config_module = importlib.import_module(f"use_cases.{use_case_name}.config")
            
            # Get classes (assuming naming convention)
            orchestrator_class_name = f"{use_case_name.title().replace('_', '')}Orchestrator"
            evaluator_class_name = f"{use_case_name.title().replace('_', '')}Evaluator"
            
            orchestrator_class = getattr(orchestrator_module, orchestrator_class_name)
            evaluator_class = getattr(evaluator_module, evaluator_class_name)
            
            # Instantiate components
            orchestrator = orchestrator_class(config_module.USE_CASE_CONFIG)
            evaluator = evaluator_class(use_case_name)
            
            # Cache for reuse
            self.use_case_cache[use_case_name] = (orchestrator, evaluator)
            
            logger.info(
                "Use case loaded successfully",
                use_case=use_case_name,
                orchestrator=orchestrator_class_name,
                evaluator=evaluator_class_name
            )
            
            return orchestrator, evaluator
            
        except Exception as e:
            logger.error(
                "Failed to load use case",
                use_case=use_case_name,
                error=str(e)
            )
            raise
    
    async def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Run a complete experiment configuration.
        
        This method implements the core research methodology:
        1. Load use case components
        2. For each test scenario and model:
           a. Run baseline (no reflection)
           b. Run reflection with varying iteration counts
        3. Evaluate all outputs with consistent criteria
        4. Store results for analysis
        """
        logger.info(
            "Starting experiment",
            experiment_id=config.experiment_id,
            use_case=config.use_case,
            scenarios=len(config.test_scenarios),
            models=len(config.models_to_test)
        )
        
        # Load use case components
        orchestrator, evaluator = self.load_use_case(config.use_case)
        
        # Create experiment output directory
        experiment_dir = self.base_output_dir / config.experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Store experiment configuration
        with open(experiment_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        experiment_results = []
        
        # Run experiments for each scenario
        for scenario_idx, scenario in enumerate(config.test_scenarios):
            logger.info(
                "Processing scenario",
                scenario_index=scenario_idx + 1,
                total_scenarios=len(config.test_scenarios)
            )
            
            for model in config.models_to_test:
                for reflection_iterations in config.reflection_configs:
                    for repetition in range(config.repetitions):
                        
                        # Run single experiment instance
                        result = await self._run_single_experiment(
                            config=config,
                            orchestrator=orchestrator,
                            evaluator=evaluator,
                            scenario=scenario,
                            model=model,
                            reflection_iterations=reflection_iterations,
                            repetition=repetition
                        )
                        
                        experiment_results.append(result)
                        
                        # Save intermediate results
                        result_file = experiment_dir / f"result_{len(experiment_results)}.json"
                        with open(result_file, "w") as f:
                            json.dump(asdict(result), f, indent=2, default=str)
        
        # Compile final results
        final_results = {
            "experiment_id": config.experiment_id,
            "config": asdict(config),
            "results": [asdict(r) for r in experiment_results],
            "summary": self._generate_experiment_summary(experiment_results),
            "completed_at": datetime.now().isoformat()
        }
        
        # Save final results
        with open(experiment_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(
            "Experiment completed",
            experiment_id=config.experiment_id,
            total_results=len(experiment_results),
            output_dir=str(experiment_dir)
        )
        
        return final_results
    
    async def _run_single_experiment(self, 
                                   config: ExperimentConfig,
                                   orchestrator: BaseUseCaseOrchestrator,
                                   evaluator: BaseUseCaseEvaluator,
                                   scenario: str,
                                   model: str,
                                   reflection_iterations: int,
                                   repetition: int) -> ExperimentResult:
        """
        Run a single experiment instance.
        
        This is where the core research question is tested:
        Does reflection improve quality compared to baseline?
        """
        start_time = datetime.now()
        
        logger.debug(
            "Running single experiment",
            model=model,
            reflection_iterations=reflection_iterations,
            repetition=repetition
        )
        
        try:
            # Track costs and performance
            with self.cost_tracker.track_experiment(
                f"{config.experiment_id}_{model}_{reflection_iterations}_{repetition}"
            ):
                
                if reflection_iterations == 0:
                    # Baseline: no reflection
                    pipeline = orchestrator.create_baseline_pipeline(model)
                else:
                    # Reflection: iterative improvement
                    pipeline = orchestrator.create_reflection_pipeline(
                        producer_model=model,
                        max_iterations=reflection_iterations
                    )
                
                # Execute pipeline
                agent_output = await pipeline.run({"input": scenario})
                
                # Evaluate output quality
                quality_scores = await evaluator.evaluate_output(
                    agent_output, 
                    scenario,
                    context={"model": model, "reflection_iterations": reflection_iterations}
                )
                
                overall_quality = evaluator.calculate_overall_score(quality_scores)
            
            # Get performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            cost_data = self.cost_tracker.get_experiment_cost(
                f"{config.experiment_id}_{model}_{reflection_iterations}_{repetition}"
            )
            
            # Create result record
            result = ExperimentResult(
                experiment_id=config.experiment_id,
                use_case=config.use_case,
                scenario=scenario,
                model=model,
                reflection_iterations=reflection_iterations,
                complexity_level=self._determine_complexity(scenario),
                repetition=repetition,
                execution_time_seconds=execution_time,
                token_usage=cost_data.get("token_usage", {}),
                api_cost_usd=cost_data.get("cost_usd", 0.0),
                quality_scores=[asdict(score) for score in quality_scores],
                overall_quality_score=overall_quality,
                agent_output=agent_output,
                evaluation_details={"evaluator_model": evaluator.evaluator_model},
                timestamp=datetime.now().isoformat()
            )
            
            logger.debug(
                "Single experiment completed",
                model=model,
                reflection_iterations=reflection_iterations,
                overall_quality=overall_quality,
                execution_time=execution_time
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Single experiment failed",
                model=model,
                reflection_iterations=reflection_iterations,
                error=str(e)
            )
            raise
    
    def _determine_complexity(self, scenario: str) -> str:
        """Determine complexity level of scenario for analysis."""
        scenario_lower = scenario.lower()
        
        if any(word in scenario_lower for word in ["simple", "basic", "small"]):
            return "simple"
        elif any(word in scenario_lower for word in ["complex", "large", "enterprise", "black friday"]):
            return "complex"
        else:
            return "medium"
    
    def _generate_experiment_summary(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate summary statistics for the experiment."""
        if not results:
            return {}
        
        # Group results by key dimensions
        by_model = {}
        by_reflection = {}
        by_complexity = {}
        
        for result in results:
            # By model
            if result.model not in by_model:
                by_model[result.model] = []
            by_model[result.model].append(result.overall_quality_score)
            
            # By reflection iterations
            if result.reflection_iterations not in by_reflection:
                by_reflection[result.reflection_iterations] = []
            by_reflection[result.reflection_iterations].append(result.overall_quality_score)
            
            # By complexity
            if result.complexity_level not in by_complexity:
                by_complexity[result.complexity_level] = []
            by_complexity[result.complexity_level].append(result.overall_quality_score)
        
        # Calculate summary statistics
        def calc_stats(values):
            if not values:
                return {}
            return {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        summary = {
            "total_experiments": len(results),
            "by_model": {model: calc_stats(scores) for model, scores in by_model.items()},
            "by_reflection": {str(refl): calc_stats(scores) for refl, scores in by_reflection.items()},
            "by_complexity": {comp: calc_stats(scores) for comp, scores in by_complexity.items()},
            "overall_stats": calc_stats([r.overall_quality_score for r in results])
        }
        
        return summary

class CostTracker:
    """Track API costs and token usage for research analysis."""
    
    def __init__(self):
        self.experiment_costs = {}
    
    def track_experiment(self, experiment_key: str):
        """Context manager for tracking experiment costs."""
        return ExperimentCostTracker(self, experiment_key)
    
    def get_experiment_cost(self, experiment_key: str) -> Dict[str, Any]:
        """Get cost data for an experiment."""
        return self.experiment_costs.get(experiment_key, {
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
            "cost_usd": 0.0
        })

class ExperimentCostTracker:
    """Context manager for tracking individual experiment costs."""
    
    def __init__(self, cost_tracker: CostTracker, experiment_key: str):
        self.cost_tracker = cost_tracker
        self.experiment_key = experiment_key
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # In real implementation, integrate with actual cost tracking
        # For now, use mock data
        self.cost_tracker.experiment_costs[self.experiment_key] = {
            "token_usage": {
                "input_tokens": 1500,  # Mock values
                "output_tokens": 800
            },
            "cost_usd": 0.023  # Mock cost
        }

# CLI interface for running experiments
async def main():
    """Main CLI entry point for running experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run agentic research experiments")
    parser.add_argument("--config", required=True, help="Path to experiment configuration file")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))  # INFO level
    
    # Load experiment configuration
    with open(args.config, "r") as f:
        config_data = json.load(f)
    
    config = ExperimentConfig(**config_data)
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Run experiment
    runner = GenericExperimentRunner()
    results = await runner.run_experiment(config)
    
    print(f"Experiment completed: {config.experiment_id}")
    print(f"Results saved to: {results.get('output_dir', 'default location')}")
    print(f"Total experiments: {len(results['results'])}")

if __name__ == "__main__":
    asyncio.run(main())

