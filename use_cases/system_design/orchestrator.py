from typing import Dict, Any
import structlog
from framework.base_orchestrator import BaseUseCaseOrchestrator
from .agents import SystemDesignProducer, SystemDesignCritic
from .config import USE_CASE_CONFIG

logger = structlog.get_logger(__name__)

class SystemDesignOrchestrator(BaseUseCaseOrchestrator):
    """
    Orchestrates system design workflow with producer-critic pattern.
    
    Why this orchestration:
    1. Implements the reflection pattern for system design
    2. Manages termination conditions specific to architecture review
    3. Handles state management between producer and critic
    4. Enables research comparison between baseline and reflection
    """
    
    def __init__(self, use_case_config: Dict[str, Any] = None):
        config = use_case_config or USE_CASE_CONFIG
        super().__init__(config)
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize system design specific agents."""
        # Start with default model, will be updated during experiments
        default_model = "gemini-2.5-flash-lite"
        
        agents = {
            "producer": SystemDesignProducer(default_model),
            "critic": SystemDesignCritic(default_model)
        }
        
        logger.info(
            "System design agents initialized",
            producer_model=agents["producer"].model,
            critic_model=agents["critic"].model
        )
        
        return agents
    
    def _should_terminate(self, state: Dict[str, Any]) -> bool:
        """
        Determine when system design reflection should terminate.
        
        Termination conditions specific to system design:
        1. Critic approves the design (DESIGN_APPROVED)
        2. No significant improvements in quality score
        3. All critical issues resolved
        4. Maximum iterations reached (handled by LoopAgent)
        """
        # Check if critic approved the design
        critic_output = state.get("critic_output", {})
        if isinstance(critic_output, str) and "DESIGN_APPROVED" in critic_output:
            logger.info("Design approved by critic, terminating reflection")
            return True
        
        # Check for structured critic response
        if isinstance(critic_output, dict):
            overall_assessment = critic_output.get("overall_assessment", "")
            if overall_assessment == "EXCELLENT":
                logger.info("Design rated as excellent, terminating reflection")
                return True
            
            # Check if no critical issues remain
            critical_issues = critic_output.get("critical_issues", [])
            if not critical_issues:
                high_issues = critic_output.get("high_issues", [])
                if len(high_issues) <= 1:  # Allow 1 minor high-priority issue
                    logger.info("No critical issues and minimal high issues, terminating reflection")
                    return True
        
        # Check improvement stagnation
        iteration_count = state.get("reflection:iteration", 0)
        if iteration_count >= 2:  # Only check after a few iterations
            quality_history = state.get("quality_history", [])
            if len(quality_history) >= 2:
                recent_improvement = quality_history[-1] - quality_history[-2]
                if recent_improvement < 0.05:  # Less than 5% improvement
                    logger.info(
                        "Quality improvement stagnated, terminating reflection",
                        recent_improvement=recent_improvement
                    )
                    return True
        
        # Continue reflection
        logger.debug(
            "Continuing reflection",
            iteration=iteration_count,
            critic_assessment=critic_output.get("overall_assessment", "unknown")
        )
        return False
    
    def prepare_producer_input(self, original_input: str, 
                             critique: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare input for producer agent, incorporating critique if available.
        
        Why this method: Enables iterative improvement by incorporating
        previous critique into the next generation cycle.
        """
        producer_input = {
            "requirements": original_input,
            "iteration_context": {
                "is_refinement": critique is not None,
                "previous_critique": critique
            }
        }
        
        if critique:
            # Add specific improvement guidance
            producer_input["improvement_focus"] = [
                "Address critical issues: " + ", ".join(critique.get("critical_issues", [])),
                "Resolve high priority issues: " + ", ".join(critique.get("high_issues", [])),
                "Implement recommendations: " + ", ".join(critique.get("recommendations", []))
            ]
            
            producer_input["refinement_instructions"] = """
            This is a refinement iteration. Focus on:
            1. Addressing all critical and high-priority issues from the critique
            2. Implementing specific recommendations provided
            3. Maintaining the good aspects of the previous design
            4. Improving overall quality while preserving working components
            """
        
        return producer_input
    
    def extract_quality_metrics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract quality metrics from agent outputs for research tracking.
        
        Why needed: Research requires consistent quality measurement
        across iterations and experiments.
        """
        metrics = {}
        
        # Extract from critic output
        critic_output = state.get("critic_output", {})
        if isinstance(critic_output, dict):
            # Map assessment to numeric score
            assessment_scores = {
                "EXCELLENT": 0.9,
                "GOOD": 0.7,
                "NEEDS_IMPROVEMENT": 0.5,
                "POOR": 0.3
            }
            
            overall_assessment = critic_output.get("overall_assessment", "NEEDS_IMPROVEMENT")
            metrics["overall_quality"] = assessment_scores.get(overall_assessment, 0.5)
            
            # Extract specific dimension scores if available
            metrics["technical_accuracy"] = critic_output.get("technical_accuracy_score", 0.5)
            metrics["cost_optimization"] = critic_output.get("cost_optimization_score", 0.5)
            metrics["security_posture"] = critic_output.get("security_score", 0.5)
            metrics["scalability_design"] = critic_output.get("scalability_score", 0.5)
            
            # Count issues by severity
            metrics["critical_issues_count"] = len(critic_output.get("critical_issues", []))
            metrics["high_issues_count"] = len(critic_output.get("high_issues", []))
            metrics["medium_issues_count"] = len(critic_output.get("medium_issues", []))
        
        return metrics

