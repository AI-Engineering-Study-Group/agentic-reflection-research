from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from google.adk.agents import SequentialAgent, LoopAgent
import structlog
from .base_agent import BaseUseCaseAgent

logger = structlog.get_logger(__name__)

class BaseUseCaseOrchestrator(ABC):
    """
    Abstract orchestrator for managing agent workflows within a use case.
    
    Why this design:
    1. Encapsulates use case specific workflow logic
    2. Provides both baseline and reflection pipelines
    3. Handles termination conditions for reflection
    4. Enables consistent experiment execution across use cases
    """
    
    def __init__(self, use_case_config: Dict[str, Any]):
        self.use_case = use_case_config["name"]
        self.config = use_case_config
        self.agents = self._initialize_agents()
        
        logger.info(
            "Orchestrator initialized",
            use_case=self.use_case,
            agent_count=len(self.agents)
        )
    
    @abstractmethod
    def _initialize_agents(self) -> Dict[str, BaseUseCaseAgent]:
        """
        Initialize all agents needed for this use case.
        
        Why abstract: Each use case has different agent requirements.
        Common patterns:
        - "producer": Creates initial output
        - "critic": Evaluates and suggests improvements
        - "refiner": Applies improvements based on critique
        """
        pass
    
    def create_baseline_pipeline(self, model: str) -> SequentialAgent:
        """
        Create single-pass pipeline without reflection.
        
        Why needed: Research baseline for comparison with reflection.
        This represents the "traditional" approach of using a single
        high-capability model pass.
        """
        # Update producer model
        producer = self.agents["producer"]
        producer.update_model(model)
        
        pipeline = SequentialAgent(
            name=f"{self.use_case}_baseline",
            sub_agents=[producer.agent]
        )
        
        logger.info(
            "Baseline pipeline created",
            use_case=self.use_case,
            model=model,
            pipeline_name=pipeline.name
        )
        
        return pipeline
    
    def create_reflection_pipeline(self, 
                                 producer_model: str,
                                 critic_model: Optional[str] = None,
                                 max_iterations: int = 3) -> LoopAgent:
        """
        Create reflection-enabled pipeline with iterative improvement.
        
        Why this design:
        1. Tests the core research hypothesis: can reflection improve quality?
        2. Allows different models for producer and critic
        3. Configurable iteration count for research
        4. Implements termination conditions to prevent infinite loops
        """
        critic_model = critic_model or producer_model
        
        # Update agent models
        producer = self.agents["producer"]
        critic = self.agents["critic"]
        
        producer.update_model(producer_model)
        critic.update_model(critic_model)
        
        # Create reflection pipeline using LoopAgent
        # LoopAgent will execute sub_agents in sequence for each iteration
        reflection_pipeline = LoopAgent(
            name=f"{self.use_case}_reflection",
            description=f"Reflection pipeline for {self.use_case} with producer-critic pattern",
            sub_agents=[producer.agent, critic.agent],
            max_iterations=max_iterations
        )
        
        logger.info(
            "Reflection pipeline created",
            use_case=self.use_case,
            producer_model=producer_model,
            critic_model=critic_model,
            max_iterations=max_iterations
        )
        
        return reflection_pipeline
    
    @abstractmethod
    def _should_terminate(self, state: Dict[str, Any]) -> bool:
        """
        Determine when reflection iterations should stop.
        
        Why abstract: Termination conditions are use case specific.
        
        Common termination conditions:
        1. Quality threshold reached
        2. No significant improvement detected
        3. Critic approves the output
        4. Specific issues resolved
        """
        pass
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent roles for this use case."""
        return list(self.agents.keys())
    
    def get_agent(self, role: str) -> BaseUseCaseAgent:
        """Get specific agent by role."""
        if role not in self.agents:
            raise ValueError(f"Agent role '{role}' not found. Available: {list(self.agents.keys())}")
        return self.agents[role]

