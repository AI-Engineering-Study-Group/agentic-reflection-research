from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import structlog
from google.adk.agents import LlmAgent
from config.settings import settings

logger = structlog.get_logger(__name__)

@dataclass
class QualityDimension:
    """
    Represents a single quality evaluation dimension.
    
    Why structured approach:
    1. Enables weighted scoring across multiple criteria
    2. Provides clear evaluation framework
    3. Allows use case specific quality definitions
    4. Supports research analysis and comparison
    """
    name: str
    weight: float
    description: str
    scale_description: str  # What do scores 0-1 mean?

@dataclass
class QualityScore:
    """Individual quality score for a dimension."""
    dimension: str
    score: float  # 0.0 to 1.0
    reasoning: str
    specific_issues: List[str]
    improvement_suggestions: List[str]

class BaseUseCaseEvaluator(ABC):
    """
    Abstract evaluator for assessing output quality across multiple dimensions.
    
    Why multi-dimensional evaluation:
    1. Real-world quality is multi-faceted
    2. Enables research into which aspects benefit from reflection
    3. Provides actionable feedback for improvement
    4. Supports expert validation from your community
    """
    
    def __init__(self, use_case: str, evaluator_model: str = None):
        self.use_case = use_case
        self.evaluator_model = evaluator_model or settings.pro_model
        self.dimensions = self._get_quality_dimensions()
        self.evaluation_agent = self._create_evaluation_agent()
        
        logger.info(
            "Evaluator initialized",
            use_case=self.use_case,
            evaluator_model=self.evaluator_model,
            dimension_count=len(self.dimensions)
        )
    
    @abstractmethod
    def _get_quality_dimensions(self) -> List[QualityDimension]:
        """
        Define use case specific quality dimensions.
        
        Why abstract: Each use case has different quality criteria.
        Examples:
        - System Design: technical_accuracy, cost_optimization, security_posture
        - Content Generation: clarity, engagement, factual_accuracy
        - Code Review: correctness, maintainability, security
        """
        pass
    
    def _create_evaluation_agent(self) -> LlmAgent:
        """
        Create ADK agent for quality evaluation.
        
        Why separate evaluation agent:
        1. Consistent evaluation across experiments
        2. Uses high-capability model for accurate assessment
        3. Can be validated against expert evaluations
        """
        evaluation_instructions = f"""
        You are an expert evaluator for {self.use_case} outputs.
        
        Your role is to assess quality across these dimensions:
        {self._format_dimensions_for_instructions()}
        
        For each dimension:
        1. Provide a score from 0.0 to 1.0
        2. Explain your reasoning
        3. List specific issues found
        4. Suggest concrete improvements
        
        Be thorough, objective, and consistent in your evaluations.
        """
        
        return LlmAgent(
            model=self.evaluator_model,
            name=f"{self.use_case}_evaluator",
            description=f"Quality evaluator for {self.use_case} outputs",
            instruction=evaluation_instructions,
            tools=[]  # Evaluators typically don't need tools
        )
    
    def _format_dimensions_for_instructions(self) -> str:
        """Format quality dimensions for agent instructions."""
        formatted = []
        for dim in self.dimensions:
            formatted.append(f"- {dim.name} (weight: {dim.weight}): {dim.description}")
        return "\n".join(formatted)
    
    @abstractmethod
    async def evaluate_output(self, output: Dict[str, Any], 
                            original_input: str,
                            context: Optional[Dict[str, Any]] = None) -> List[QualityScore]:
        """
        Evaluate output quality across all dimensions.
        
        Why abstract: Evaluation logic is use case specific.
        Each use case needs to:
        1. Extract relevant parts of output for evaluation
        2. Apply domain-specific evaluation criteria
        3. Generate structured quality scores
        """
        pass
    
    def calculate_overall_score(self, scores: List[QualityScore]) -> float:
        """
        Calculate weighted overall quality score.
        
        Why weighted: Different quality aspects have different importance.
        Research insight: Can identify which aspects benefit most from reflection.
        """
        if not scores:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for score in scores:
            # Find dimension weight
            dimension = next(
                (d for d in self.dimensions if d.name == score.dimension), 
                None
            )
            if dimension:
                total_score += score.score * dimension.weight
                total_weight += dimension.weight
        
        overall = total_score / total_weight if total_weight > 0 else 0.0
        
        logger.debug(
            "Overall score calculated",
            use_case=self.use_case,
            individual_scores=len(scores),
            overall_score=overall
        )
        
        return overall
    
    def get_dimension_scores(self, scores: List[QualityScore]) -> Dict[str, float]:
        """Extract dimension scores as dictionary for analysis."""
        return {score.dimension: score.score for score in scores}
    
    def identify_improvement_opportunities(self, scores: List[QualityScore]) -> List[str]:
        """
        Identify top improvement opportunities based on scores.
        
        Why needed: Helps prioritize reflection focus areas.
        Research insight: Which dimensions benefit most from iteration?
        """
        # Sort by lowest scores (biggest improvement opportunities)
        sorted_scores = sorted(scores, key=lambda s: s.score)
        
        opportunities = []
        for score in sorted_scores[:3]:  # Top 3 opportunities
            if score.score < 0.7:  # Only if below threshold
                opportunities.extend(score.improvement_suggestions)
        
        return opportunities

