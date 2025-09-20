from typing import Dict, Any, List, Optional
import structlog
from framework.base_evaluator import BaseUseCaseEvaluator, QualityScore
from .config import QUALITY_DIMENSIONS
import json

logger = structlog.get_logger(__name__)

class SystemDesignEvaluator(BaseUseCaseEvaluator):
    """
    Evaluates system design quality across multiple dimensions.
    
    Why multi-dimensional evaluation for system design:
    1. Architecture quality is inherently multi-faceted
    2. Enables research into which aspects benefit from reflection
    3. Provides actionable feedback for improvement
    4. Matches how your engineering community actually evaluates designs
    """
    
    def __init__(self, use_case: str = "system_design", evaluator_model: str = None):
        super().__init__(use_case, evaluator_model)
    
    def _get_quality_dimensions(self):
        """Return system design specific quality dimensions."""
        return QUALITY_DIMENSIONS
    
    async def evaluate_output(self, output: Dict[str, Any], 
                            original_input: str,
                            context: Optional[Dict[str, Any]] = None) -> List[QualityScore]:
        """
        Evaluate system design output across all quality dimensions.
        
        This is the core evaluation logic that enables research comparison
        between baseline and reflection approaches.
        """
        try:
            # Prepare evaluation input for the ADK agent
            evaluation_input = {
                "original_requirements": original_input,
                "system_design": output,
                "evaluation_context": context or {}
            }
            
            # Use evaluation agent to score each dimension
            scores = []
            
            for dimension in self.dimensions:
                score = await self._evaluate_dimension(
                    evaluation_input, 
                    dimension
                )
                scores.append(score)
            
            logger.info(
                "System design evaluation completed",
                dimension_count=len(scores),
                overall_score=self.calculate_overall_score(scores)
            )
            
            return scores
            
        except Exception as e:
            logger.error(
                "System design evaluation failed",
                error=str(e),
                output_keys=list(output.keys()) if isinstance(output, dict) else "non-dict"
            )
            raise
    
    async def _evaluate_dimension(self, evaluation_input: Dict[str, Any],
                                dimension) -> QualityScore:
        """Evaluate a specific quality dimension."""
        
        # Create dimension-specific evaluation prompt
        dimension_prompt = f"""
        Evaluate the system design for {dimension.name}.
        
        Dimension: {dimension.name}
        Description: {dimension.description}
        Scale: {dimension.scale_description}
        Weight: {dimension.weight}
        
        Original Requirements:
        {evaluation_input['original_requirements']}
        
        System Design to Evaluate:
        {evaluation_input['system_design']}
        
        Provide your evaluation in this exact JSON format:
        {{
            "score": <float between 0.0 and 1.0>,
            "reasoning": "<detailed explanation of the score>",
            "specific_issues": ["<issue1>", "<issue2>", ...],
            "improvement_suggestions": ["<suggestion1>", "<suggestion2>", ...]
        }}
        
        Be thorough, objective, and specific in your evaluation.
        Focus only on the {dimension.name} aspect.
        """
        
        try:
            # Use evaluation agent with proper ADK Runner pattern
            from google.adk.runners import Runner
            from google.adk.sessions import DatabaseSessionService
            from google.genai import types
            from config.settings import settings
            
            # Create session service
            session_service = DatabaseSessionService(db_url=settings.database_url or "sqlite:///research.db")
            
            # Create session
            user_id = "evaluation_user"
            session = await session_service.create_session(
                app_name=f"{self.use_case}_evaluator",
                user_id=user_id
            )
            
            # Create runner with proper app_name matching
            runner = Runner(
                app_name=f"{self.use_case}_evaluator",
                agent=self.evaluation_agent,
                session_service=session_service
            )
            
            # Create proper ADK Content object
            content = types.Content(role="user", parts=[types.Part(text=dimension_prompt)])
            
            # Execute agent using run_async
            response_text = ""
            async for evt in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=content
            ):
                if hasattr(evt, 'content') and evt.content:
                    if hasattr(evt.content, 'parts'):
                        for part in evt.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
            
            # Try to parse JSON from response
            if response_text:
                try:
                    result_data = json.loads(response_text)
                except json.JSONDecodeError:
                    # If not JSON, create a fallback structure
                    result_data = {
                        "score": 0.5,
                        "reasoning": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                        "specific_issues": [],
                        "improvement_suggestions": []
                    }
            else:
                # No response received
                result_data = {
                    "score": 0.5,
                    "reasoning": "No response received from evaluation agent",
                    "specific_issues": ["Evaluation agent did not respond"],
                    "improvement_suggestions": ["Check evaluation agent configuration"]
                }
            
            return QualityScore(
                dimension=dimension.name,
                score=float(result_data.get("score", 0.5)),
                reasoning=result_data.get("reasoning", "No reasoning provided"),
                specific_issues=result_data.get("specific_issues", []),
                improvement_suggestions=result_data.get("improvement_suggestions", [])
            )
            
        except Exception as e:
            logger.warning(
                "Dimension evaluation failed, using fallback",
                dimension=dimension.name,
                error=str(e)
            )
            
            # Fallback evaluation based on heuristics
            return self._fallback_dimension_evaluation(
                evaluation_input, 
                dimension
            )
    
    def _fallback_dimension_evaluation(self, evaluation_input: Dict[str, Any],
                                     dimension) -> QualityScore:
        """
        Fallback evaluation when LLM evaluation fails.
        
        Why needed: Ensures research can continue even if evaluation
        agent encounters issues.
        """
        design = evaluation_input['system_design']
        
        # Simple heuristic-based evaluation
        score = 0.5  # Default neutral score
        issues = []
        suggestions = []
        reasoning = f"Fallback evaluation for {dimension.name}"
        
        if dimension.name == "technical_accuracy":
            # Check for presence of key technical components
            if isinstance(design, dict):
                if "architecture" in design or "components" in design:
                    score = 0.7
                    reasoning = "Design includes architectural components"
                else:
                    score = 0.3
                    issues.append("Missing architectural components")
                    suggestions.append("Include detailed component architecture")
        
        elif dimension.name == "cost_optimization":
            # Check for cost considerations
            if isinstance(design, dict):
                if "cost" in str(design).lower() or "pricing" in str(design).lower():
                    score = 0.6
                    reasoning = "Design mentions cost considerations"
                else:
                    score = 0.4
                    issues.append("No cost optimization mentioned")
                    suggestions.append("Include cost analysis and optimization")
        
        elif dimension.name == "security_posture":
            # Check for security mentions
            if isinstance(design, dict):
                security_keywords = ["security", "authentication", "encryption", "firewall", "vpc"]
                if any(keyword in str(design).lower() for keyword in security_keywords):
                    score = 0.6
                    reasoning = "Design includes security considerations"
                else:
                    score = 0.3
                    issues.append("Limited security considerations")
                    suggestions.append("Include comprehensive security design")
        
        return QualityScore(
            dimension=dimension.name,
            score=score,
            reasoning=reasoning,
            specific_issues=issues,
            improvement_suggestions=suggestions
        )
    
    def compare_designs(self, design1_scores: List[QualityScore], 
                       design2_scores: List[QualityScore]) -> Dict[str, Any]:
        """
        Compare two designs for research analysis.
        
        Why needed: Core functionality for reflection vs baseline comparison.
        """
        comparison = {
            "overall_improvement": 0.0,
            "dimension_improvements": {},
            "significant_improvements": [],
            "regressions": []
        }
        
        # Calculate overall improvement
        overall1 = self.calculate_overall_score(design1_scores)
        overall2 = self.calculate_overall_score(design2_scores)
        comparison["overall_improvement"] = overall2 - overall1
        
        # Compare each dimension
        scores1_dict = {s.dimension: s.score for s in design1_scores}
        scores2_dict = {s.dimension: s.score for s in design2_scores}
        
        for dimension in scores1_dict:
            if dimension in scores2_dict:
                improvement = scores2_dict[dimension] - scores1_dict[dimension]
                comparison["dimension_improvements"][dimension] = improvement
                
                if improvement >= 0.1:  # 10% improvement threshold
                    comparison["significant_improvements"].append({
                        "dimension": dimension,
                        "improvement": improvement,
                        "before": scores1_dict[dimension],
                        "after": scores2_dict[dimension]
                    })
                elif improvement <= -0.1:  # 10% regression threshold
                    comparison["regressions"].append({
                        "dimension": dimension,
                        "regression": improvement,
                        "before": scores1_dict[dimension],
                        "after": scores2_dict[dimension]
                    })
        
        return comparison

