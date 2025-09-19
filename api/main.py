#!/usr/bin/env python3
"""
FastAPI server for the Agentic Research Framework.

Provides REST API endpoints for:
1. Research chat interface
2. Experiment execution
3. Results analysis
4. Expert evaluation integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import structlog
import asyncio
from datetime import datetime
import uuid

from config.settings import settings
from research.experiment_orchestrator import GenericExperimentRunner, ExperimentConfig
from use_cases.system_design.orchestrator import SystemDesignOrchestrator
from use_cases.system_design.evaluator import SystemDesignEvaluator
from use_cases.system_design.config import USE_CASE_CONFIG, TEST_SCENARIOS

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Research Framework API",
    description="REST API for researching iterative reflection vs model capability",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message or system design question")
    mode: str = Field(default="chat", description="Mode: 'chat', 'baseline', or 'reflection'")
    model: str = Field(default="gemini-2.5-flash-lite", description="Model to use")
    reflection_iterations: int = Field(default=3, description="Number of reflection iterations")
    use_case: str = Field(default="system_design", description="Use case to apply")

class ChatResponse(BaseModel):
    response: str
    mode_used: str
    model_used: str
    session_id: str
    quality_score: Optional[float] = None
    reflection_iterations_used: Optional[int] = None
    processing_time_seconds: float
    metadata: Dict[str, Any] = {}

class ExperimentRequest(BaseModel):
    experiment_name: str = Field(..., description="Name for the experiment")
    scenarios: List[str] = Field(..., description="Test scenarios to run")
    models: List[str] = Field(default=["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"], description="Models to test")
    reflection_configs: List[int] = Field(default=[0, 2, 3], description="Reflection iteration counts")
    repetitions: int = Field(default=1, description="Number of repetitions per configuration")

class ComparisonRequest(BaseModel):
    message: str = Field(..., description="Message to compare across modes")
    model: str = Field(default="gemini-2.5-flash-lite", description="Model to use for comparison")
    use_case: str = Field(default="system_design", description="Use case for comparison")

# Global state for sessions (in production, use proper session management)
active_sessions = {}
active_experiments = {}

@app.get("/")
async def root():
    """Health check and API information."""
    return {
        "message": "Agentic Research Framework API",
        "version": "0.1.0",
        "status": "running",
        "available_endpoints": [
            "/chat",
            "/chat/compare", 
            "/experiments/run",
            "/experiments/status",
            "/use-cases",
            "/models"
        ]
    }

@app.get("/models")
async def get_available_models():
    """Get available models for research."""
    return {
        "available_models": settings.available_models,
        "default_model": settings.default_model,
        "pro_model": settings.pro_model,
        "model_info": {
            "gemini-2.5-flash-lite": {
                "description": "Lightweight model without thinking capability (DEFAULT)",
                "thinking_budget": "Model does not think by default (can enable with 512 to 24576 tokens)",
                "best_for": "Research baseline, cost efficiency, testing reflection effectiveness",
                "research_note": "Perfect for testing if reflection can compensate for lower base capability"
            },
            "gemini-2.5-flash": {
                "description": "Fast, cost-efficient model with adaptive thinking",
                "thinking_budget": "Dynamic thinking (0 to 24576 tokens)",
                "best_for": "General tasks, balanced cost and capability"
            },
            "gemini-2.5-pro": {
                "description": "Enhanced reasoning and advanced capabilities",
                "thinking_budget": "Dynamic thinking (128 to 32768 tokens)", 
                "best_for": "Complex reasoning, highest quality output"
            }
        }
    }

@app.get("/use-cases")
async def get_use_cases():
    """Get available use cases and their test scenarios."""
    return {
        "available_use_cases": {
            "system_design": {
                "description": USE_CASE_CONFIG["description"],
                "evaluation_dimensions": USE_CASE_CONFIG["evaluation_dimensions"],
                "test_scenarios": {
                    complexity: [scenario["id"] for scenario in scenarios]
                    for complexity, scenarios in TEST_SCENARIOS.items()
                }
            }
        },
        "chat_modes": {
            "chat": "Normal conversational mode",
            "baseline": "Single-pass generation for research comparison",
            "reflection": "Iterative improvement with reflection pattern"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def research_chat(request: ChatRequest):
    """
    Chat with research framework agents.
    
    Supports three modes:
    - chat: Normal conversation
    - baseline: Single-pass for research 
    - reflection: Iterative improvement
    """
    start_time = datetime.now()
    session_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Chat request: mode={request.mode}, model={request.model}")
        
        # Load use case components
        runner = GenericExperimentRunner()
        orchestrator, evaluator = runner.load_use_case(request.use_case)
        
        # Execute based on mode
        quality_score = None
        reflection_iterations_used = None
        
        if request.mode == "chat":
            # Normal chat - use producer agent directly
            producer = orchestrator.get_agent("producer")
            producer.update_model(request.model)
            result = await producer.run({"input": request.message})
            
        elif request.mode == "baseline":
            # Baseline - single pass
            pipeline = orchestrator.create_baseline_pipeline(request.model)
            result = await pipeline.run({"input": request.message})
            
            # Evaluate quality
            quality_scores = await evaluator.evaluate_output(result, request.message)
            quality_score = evaluator.calculate_overall_score(quality_scores)
            
        elif request.mode == "reflection":
            # Reflection - iterative improvement
            pipeline = orchestrator.create_reflection_pipeline(
                producer_model=request.model,
                max_iterations=request.reflection_iterations
            )
            result = await pipeline.run({"input": request.message})
            reflection_iterations_used = request.reflection_iterations
            
            # Evaluate quality
            quality_scores = await evaluator.evaluate_output(result, request.message)
            quality_score = evaluator.calculate_overall_score(quality_scores)
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")
        
        # Extract response text
        if isinstance(result, dict):
            response_text = result.get("response", str(result))
        else:
            response_text = str(result)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store session for potential follow-up
        active_sessions[session_id] = {
            "last_request": request.dict(),
            "last_response": response_text,
            "created_at": start_time,
            "orchestrator": orchestrator,
            "evaluator": evaluator
        }
        
        return ChatResponse(
            response=response_text,
            mode_used=request.mode,
            model_used=request.model,
            session_id=session_id,
            quality_score=quality_score,
            reflection_iterations_used=reflection_iterations_used,
            processing_time_seconds=processing_time,
            metadata={
                "use_case": request.use_case,
                "timestamp": start_time.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/compare")
async def compare_modes(request: ComparisonRequest):
    """
    Compare baseline vs reflection modes side-by-side.
    
    Useful for demonstrating the effectiveness of reflection.
    """
    try:
        logger.info(f"Mode comparison request for: {request.message[:100]}...")
        
        # Load use case
        runner = GenericExperimentRunner()
        orchestrator, evaluator = runner.load_use_case(request.use_case)
        
        # Run baseline
        baseline_pipeline = orchestrator.create_baseline_pipeline(request.model)
        baseline_result = await baseline_pipeline.run({"input": request.message})
        baseline_quality = await evaluator.evaluate_output(baseline_result, request.message)
        
        # Run reflection
        reflection_pipeline = orchestrator.create_reflection_pipeline(
            producer_model=request.model,
            max_iterations=3
        )
        reflection_result = await reflection_pipeline.run({"input": request.message})
        reflection_quality = await evaluator.evaluate_output(reflection_result, request.message)
        
        # Compare results
        comparison = evaluator.compare_designs(baseline_quality, reflection_quality)
        
        return {
            "baseline": {
                "response": str(baseline_result),
                "quality_score": evaluator.calculate_overall_score(baseline_quality),
                "quality_breakdown": [
                    {
                        "dimension": score.dimension,
                        "score": score.score,
                        "reasoning": score.reasoning
                    }
                    for score in baseline_quality
                ]
            },
            "reflection": {
                "response": str(reflection_result),
                "quality_score": evaluator.calculate_overall_score(reflection_quality),
                "quality_breakdown": [
                    {
                        "dimension": score.dimension,
                        "score": score.score,
                        "reasoning": score.reasoning
                    }
                    for score in reflection_quality
                ]
            },
            "comparison": {
                "overall_improvement": comparison["overall_improvement"],
                "improvement_percentage": f"{comparison['overall_improvement'] * 100:.1f}%",
                "significant_improvements": comparison["significant_improvements"],
                "dimension_improvements": comparison["dimension_improvements"],
                "winner": "reflection" if comparison["overall_improvement"] > 0 else "baseline"
            }
        }
        
    except Exception as e:
        logger.error(f"Mode comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiments/run")
async def run_experiment(request: ExperimentRequest, background_tasks: BackgroundTasks):
    """
    Run a complete research experiment in the background.
    
    Returns experiment ID for status tracking.
    """
    experiment_id = f"exp_{int(datetime.now().timestamp())}"
    
    try:
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_id=experiment_id,
            use_case="system_design",
            test_scenarios=request.scenarios,
            models_to_test=request.models,
            reflection_configs=request.reflection_configs,
            complexity_levels=["simple", "medium", "complex"],
            repetitions=request.repetitions
        )
        
        # Initialize experiment state
        active_experiments[experiment_id] = {
            "status": "starting",
            "config": config,
            "started_at": datetime.now(),
            "progress": 0,
            "total_runs": len(request.scenarios) * len(request.models) * len(request.reflection_configs) * request.repetitions,
            "completed_runs": 0,
            "results": None
        }
        
        # Run experiment in background
        background_tasks.add_task(execute_experiment, experiment_id, config)
        
        return {
            "experiment_id": experiment_id,
            "status": "started",
            "estimated_runs": active_experiments[experiment_id]["total_runs"],
            "check_status_url": f"/experiments/status/{experiment_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_experiment(experiment_id: str, config: ExperimentConfig):
    """Background task to execute the experiment."""
    try:
        active_experiments[experiment_id]["status"] = "running"
        
        runner = GenericExperimentRunner()
        results = await runner.run_experiment(config)
        
        active_experiments[experiment_id].update({
            "status": "completed",
            "completed_at": datetime.now(),
            "results": results,
            "progress": 100
        })
        
        logger.info(f"Experiment {experiment_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}")
        active_experiments[experiment_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now()
        })

@app.get("/experiments/status/{experiment_id}")
async def get_experiment_status(experiment_id: str):
    """Get the status of a running experiment."""
    if experiment_id not in active_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = active_experiments[experiment_id]
    
    response = {
        "experiment_id": experiment_id,
        "status": experiment["status"],
        "started_at": experiment["started_at"].isoformat(),
        "progress_percentage": experiment.get("progress", 0)
    }
    
    if experiment["status"] == "completed":
        response["completed_at"] = experiment["completed_at"].isoformat()
        response["results_summary"] = experiment["results"]["summary"]
        
    elif experiment["status"] == "failed":
        response["error"] = experiment["error"]
        response["failed_at"] = experiment["failed_at"].isoformat()
    
    return response

@app.get("/experiments/results/{experiment_id}")
async def get_experiment_results(experiment_id: str):
    """Get detailed results from a completed experiment."""
    if experiment_id not in active_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = active_experiments[experiment_id]
    
    if experiment["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Experiment status is {experiment['status']}")
    
    return experiment["results"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
