#!/usr/bin/env python3
"""
Use Case-Specific API Server

This server runs independently for each use case, allowing:
1. Independent scaling and resource allocation
2. Isolated experiment execution
3. Use case-specific optimizations
4. Parallel research across different domains
"""

import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import structlog
import asyncio
from datetime import datetime
import uuid

from config.settings import settings

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

# Get use case from environment
USE_CASE = os.getenv("USE_CASE", "system_design")
API_PORT = int(os.getenv("API_PORT", "8001"))

# Initialize FastAPI app for specific use case
app = FastAPI(
    title=f"Agentic Research Framework - {USE_CASE.title()} Use Case",
    description=f"Independent API server for {USE_CASE} research experiments",
    version="0.1.0",
    docs_url=f"/docs",
    redoc_url=f"/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UseCaseChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    mode: str = Field(default="chat", description="Mode: 'chat', 'baseline', or 'reflection'")
    model: str = Field(default="gemini-2.5-flash-lite", description="Model to use")
    reflection_iterations: int = Field(default=3, description="Number of reflection iterations")

class UseCaseChatResponse(BaseModel):
    response: str
    mode_used: str
    model_used: str
    session_id: str
    use_case: str
    quality_score: Optional[float] = None
    reflection_iterations_used: Optional[int] = None
    processing_time_seconds: float
    resource_usage: Dict[str, Any] = {}

# Global state for this use case instance
active_sessions = {}
active_experiments = {}

# Dynamic import of use case components
def load_use_case_components():
    """Dynamically load the use case components based on USE_CASE environment variable."""
    try:
        from research.experiment_orchestrator import GenericExperimentRunner
        
        runner = GenericExperimentRunner()
        orchestrator, evaluator = runner.load_use_case(USE_CASE)
        
        logger.info(f"Use case components loaded successfully for {USE_CASE}")
        return orchestrator, evaluator
        
    except Exception as e:
        logger.error(f"Failed to load use case components for {USE_CASE}: {e}")
        raise

# Load components at startup
try:
    orchestrator, evaluator = load_use_case_components()
except Exception as e:
    logger.error(f"Failed to initialize use case {USE_CASE}: {e}")
    orchestrator, evaluator = None, None

@app.get("/")
async def root():
    """Health check and use case information."""
    return {
        "message": f"Agentic Research Framework - {USE_CASE.title()} Use Case",
        "use_case": USE_CASE,
        "api_port": API_PORT,
        "version": "0.1.0",
        "status": "running",
        "available_endpoints": [
            "/health",
            "/chat",
            "/chat/compare",
            "/experiments/run",
            "/info"
        ]
    }

@app.get("/health")
async def health_check():
    """Kubernetes-style health check."""
    try:
        # Check if use case components are loaded
        if orchestrator is None or evaluator is None:
            raise HTTPException(status_code=503, detail="Use case components not loaded")
        
        return {
            "status": "healthy",
            "use_case": USE_CASE,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "orchestrator": "loaded",
                "evaluator": "loaded"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/info")
async def get_use_case_info():
    """Get detailed information about this use case instance."""
    try:
        # Get use case configuration
        config_module = __import__(f"use_cases.{USE_CASE}.config", fromlist=["USE_CASE_CONFIG"])
        use_case_config = getattr(config_module, "USE_CASE_CONFIG", {})
        
        return {
            "use_case": USE_CASE,
            "api_port": API_PORT,
            "description": use_case_config.get("description", "No description available"),
            "evaluation_dimensions": use_case_config.get("evaluation_dimensions", []),
            "available_agents": orchestrator.get_available_agents() if orchestrator else [],
            "default_model": settings.default_model,
            "available_models": settings.available_models,
            "resource_allocation": {
                "dedicated_container": True,
                "isolated_experiments": True,
                "independent_scaling": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get use case info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=UseCaseChatResponse)
async def use_case_chat(request: UseCaseChatRequest):
    """
    Chat endpoint specific to this use case.
    
    Runs in isolation with dedicated resources for accurate measurement.
    """
    start_time = datetime.now()
    session_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Use case chat request: {USE_CASE}, mode={request.mode}, model={request.model}")
        
        if orchestrator is None or evaluator is None:
            raise HTTPException(status_code=503, detail="Use case components not loaded")
        
        # Execute based on mode
        quality_score = None
        reflection_iterations_used = None
        
        if request.mode == "chat":
            # Normal chat - use producer agent directly
            producer = orchestrator.get_agent("producer")
            producer.update_model(request.model)
            result = await producer.run({"input": request.message})
            
        elif request.mode == "baseline":
            # Baseline - single pass using producer agent directly
            producer = orchestrator.get_agent("producer")
            producer.update_model(request.model)
            result = await producer.run({"input": request.message})
            
            # Evaluate quality
            quality_scores = await evaluator.evaluate_output(result, request.message)
            quality_score = evaluator.calculate_overall_score(quality_scores)
            
        elif request.mode == "reflection":
            # Reflection - manual implementation of producer-critic pattern
            producer = orchestrator.get_agent("producer")
            critic = orchestrator.get_agent("critic")
            producer.update_model(request.model)
            critic.update_model(request.model)
            
            # Initial production
            current_input = request.message
            reflection_iterations_used = 0
            
            for iteration in range(request.reflection_iterations):
                # Producer: Generate/refine design
                producer_result = await producer.run({"input": current_input})
                producer_response = producer_result.get("response", str(producer_result))
                
                # Critic: Evaluate the design
                critic_input = f"Review this system design:\n\n{producer_response}\n\nOriginal requirements: {request.message}"
                critic_result = await critic.run({"input": critic_input})
                critic_response = critic_result.get("response", str(critic_result))
                
                reflection_iterations_used += 1
                
                # Log iteration details for research analysis
                logger.info(
                    "Reflection iteration completed",
                    iteration=iteration + 1,
                    producer_response_length=len(producer_response),
                    critic_response_preview=critic_response[:200] + "..." if len(critic_response) > 200 else critic_response
                )
                
                # Check if critic approves (simple termination condition)
                termination_keywords = ["DESIGN_APPROVED", "EXCELLENT", "APPROVED", "PERFECT"]
                critic_upper = critic_response.upper()
                terminated_by = None
                
                for keyword in termination_keywords:
                    if keyword in critic_upper:
                        terminated_by = keyword
                        break
                
                if terminated_by:
                    logger.info(
                        "Reflection terminated by critic approval",
                        iteration=iteration + 1,
                        termination_keyword=terminated_by,
                        total_iterations_used=reflection_iterations_used
                    )
                    result = producer_result
                    break
                
                # Prepare input for next iteration
                current_input = f"Refine this system design based on the critique:\n\nOriginal requirements: {request.message}\n\nCurrent design:\n{producer_response}\n\nCritique:\n{critic_response}\n\nProvide an improved design."
                result = producer_result  # Keep the last producer result
            
            # Evaluate final quality
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
        
        # Store session for this use case instance
        active_sessions[session_id] = {
            "last_request": request.dict(),
            "last_response": response_text,
            "created_at": start_time,
            "use_case": USE_CASE
        }
        
        return UseCaseChatResponse(
            response=response_text,
            mode_used=request.mode,
            model_used=request.model,
            session_id=session_id,
            use_case=USE_CASE,
            quality_score=quality_score,
            reflection_iterations_used=reflection_iterations_used,
            processing_time_seconds=processing_time,
            resource_usage={
                "container_port": API_PORT,
                "isolated_execution": True,
                "dedicated_resources": True
            }
        )
        
    except Exception as e:
        logger.error(f"Use case chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/compare")
async def compare_modes(request: UseCaseChatRequest):
    """
    Compare baseline vs reflection modes for this specific use case.
    
    Runs in dedicated container for accurate resource measurement.
    """
    try:
        logger.info(f"Mode comparison for use case {USE_CASE}: {request.message[:100]}...")
        
        if orchestrator is None or evaluator is None:
            raise HTTPException(status_code=503, detail="Use case components not loaded")
        
        # Run baseline
        baseline_pipeline = orchestrator.create_baseline_pipeline(request.model)
        baseline_result = await baseline_pipeline.run({"input": request.message})
        baseline_quality = await evaluator.evaluate_output(baseline_result, request.message)
        
        # Run reflection
        reflection_pipeline = orchestrator.create_reflection_pipeline(
            producer_model=request.model,
            max_iterations=request.reflection_iterations
        )
        reflection_result = await reflection_pipeline.run({"input": request.message})
        reflection_quality = await evaluator.evaluate_output(reflection_result, request.message)
        
        # Compare results
        comparison = evaluator.compare_designs(baseline_quality, reflection_quality)
        
        return {
            "use_case": USE_CASE,
            "container_port": API_PORT,
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
            },
            "resource_info": {
                "isolated_execution": True,
                "dedicated_container": True,
                "use_case_specific": True
            }
        }
        
    except Exception as e:
        logger.error(f"Mode comparison failed for {USE_CASE}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_active_sessions():
    """Get active sessions for this use case instance."""
    return {
        "use_case": USE_CASE,
        "container_port": API_PORT,
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": sid,
                "created_at": session["created_at"].isoformat(),
                "use_case": session["use_case"]
            }
            for sid, session in active_sessions.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.use_case_server:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=True,
        log_level="info"
    )
