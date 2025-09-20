# Use Case Extension Guide: Adding New Research Domains

## 🎯 **Overview**

This guide explains how to extend the Agentic Research Framework to new use cases beyond system design. The framework is designed to be modular, allowing researchers to easily add new domains while maintaining consistent experiment methodology and quality evaluation.

**Perfect for:** Researchers, developers, and AI enthusiasts who want to build on this framework - **no prior agent development experience required!**

## 🎓 **Learning Path for Beginners**

### **New to AI Agents? Start Here:**

#### **1. Understanding AI Agents (15 minutes)**
- 📖 **[What are AI Agents?](https://python.langchain.com/docs/concepts/agents/)** - LangChain introduction
- 🎥 **[AI Agents Explained](https://www.youtube.com/results?search_query=AI+agents+explained+tutorial)** - YouTube tutorials
- 📚 **[Agent Design Patterns](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)** - DeepLearning.AI course

#### **2. Producer-Critic Pattern (10 minutes)**
- 📖 **Read:** `Chapter 4_ Reflection.txt` in this repository
- 🧠 **Key Concept:** Separate agents for generation (Producer) and evaluation (Critic)
- 💡 **Why It Works:** Prevents cognitive bias of self-review

#### **3. Google ADK Basics (20 minutes)**
- 📖 **[Official ADK Documentation](https://google.github.io/adk-docs/)**
- 🚀 **[Getting Started Guide](https://google.github.io/adk-docs/getting-started/)**
- 🛠️ **[Agent Development Tutorial](https://google.github.io/adk-docs/tutorials/)**
- 📝 **[ADK Python Examples](https://github.com/google/adk/tree/main/examples)**

#### **4. Docker Basics (30 minutes)**
- 📖 **[Docker for Beginners](https://docs.docker.com/get-started/)**
- 🎥 **[Docker Tutorial](https://www.youtube.com/watch?v=fqMOX6JJhGo)** - FreeCodeCamp
- 🐳 **[Docker Compose Guide](https://docs.docker.com/compose/gettingstarted/)**

#### **5. FastAPI Basics (20 minutes)**
- 📖 **[FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)**
- 🎥 **[FastAPI Crash Course](https://www.youtube.com/results?search_query=FastAPI+tutorial+beginner)**

### **Recommended Learning Order:**
1. **Start with AI Agents concepts** (understand the big picture)
2. **Read Chapter 4** (understand reflection pattern)
3. **Explore ADK documentation** (understand the tools)
4. **Follow this guide** (build your first use case)
5. **Experiment and iterate** (learn by doing)

## 📋 **Prerequisites**

### **Required Knowledge:**
- ✅ **Basic Python** (functions, classes, async/await)
- ✅ **Basic understanding of AI/LLMs** (what they do, how they work)
- ✅ **Command line comfort** (running commands, file navigation)

### **Helpful but Not Required:**
- 🔄 **Docker experience** (we provide complete examples)
- 🔄 **API development** (FastAPI patterns are provided)
- 🔄 **Agent frameworks** (ADK patterns are documented)

### **Before Starting, Read:**
- 📖 **Producer-Critic reflection pattern** (see `Chapter 4_ Reflection.txt`)
- 📊 **Research methodology** (see `Iterative Reflection vs.txt`)
- 🏗️ **Framework architecture** (see `COMPREHENSIVE_BUILD_SUMMARY.md`)
- 🐳 **Docker setup** (see main `README.md`)
- 🔧 **Recent debugging fixes** (see `COMPREHENSIVE_BUILD_SUMMARY.md` - Final Debugging Session)

## 🏗️ **Framework Architecture**

### **Modular Design Philosophy**
```
Framework Core (Generic)
├── Base Classes (Agent, Orchestrator, Evaluator)
├── Experiment Runner
├── Quality Assessment Framework
└── Performance Metrics

Use Case Modules (Pluggable)
├── System Design ✅ (Reference Implementation - FULLY WORKING)
├── Code Review 📝 (Your new use case)
├── Content Generation 📝 (Your new use case)
├── Strategic Planning 📝 (Your new use case)
└── [Any Domain] 📝 (Your new use case)
```

### **✅ Current Status: FULLY OPERATIONAL**
The framework has been thoroughly debugged and all critical issues resolved:
- ✅ **All models working**: Flash-Lite, Flash, Pro
- ✅ **All modes working**: chat, baseline, reflection  
- ✅ **Quality evaluation functional**: Multi-dimensional scoring
- ✅ **No runtime errors**: All edge cases handled
- ✅ **Research ready**: Systematic experiments possible

### **Consistent Research Methodology**
Each use case follows the same pattern:
1. **Producer Agent**: Generates initial output
2. **Critic Agent**: Evaluates and suggests improvements
3. **Orchestrator**: Manages reflection cycles and termination
4. **Evaluator**: Assesses quality across multiple dimensions
5. **Tools**: Domain-specific capabilities
6. **Configuration**: Test scenarios and quality dimensions

---

## 📁 **Step-by-Step Implementation Guide**

### **Step 1: Create Use Case Directory Structure**

```bash
# Create the basic directory structure
mkdir -p use_cases/your_use_case/{agents,tools,orchestrator,evaluator,docker}
mkdir -p use_cases/your_use_case/docker

# Create required files
touch use_cases/your_use_case/__init__.py
touch use_cases/your_use_case/config.py
touch use_cases/your_use_case/agents.py
touch use_cases/your_use_case/orchestrator.py
touch use_cases/your_use_case/evaluator.py
touch use_cases/your_use_case/tools/__init__.py
```

**Expected Structure:**
```
use_cases/your_use_case/
├── __init__.py
├── config.py                    # Use case configuration
├── agents.py                    # Producer & Critic agents
├── orchestrator.py              # Workflow management
├── evaluator.py                 # Quality assessment
├── tools/                       # Domain-specific tools
│   ├── __init__.py
│   └── your_domain_tools.py
└── docker/                      # Container configuration
    ├── Dockerfile
    ├── docker-compose.yml
    └── entrypoint.sh
```

### **Step 2: Define Use Case Configuration**

> 💡 **For Beginners:** This file defines what your use case does, how to measure quality, and what test scenarios to use. Think of it as the "blueprint" for your domain.

**📚 Helpful Resources:**
- **[Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/)** - Understanding data validation
- **[Type Hints in Python](https://docs.python.org/3/library/typing.html)** - Python typing basics

Create `use_cases/your_use_case/config.py`:

```python
from typing import Dict, Any, List
from framework.base_evaluator import QualityDimension

# Use case configuration
USE_CASE_CONFIG = {
    "name": "your_use_case",  # e.g., "code_review", "content_generation"
    "description": "Brief description of what this use case does",
    "complexity_levels": ["simple", "medium", "complex"],
    "evaluation_dimensions": [
        "accuracy",      # Domain-specific quality aspects
        "completeness", 
        "clarity",
        "efficiency",
        "creativity",    # If applicable
        "compliance"     # If applicable
    ]
}

# Quality dimensions for your domain
QUALITY_DIMENSIONS = [
    QualityDimension(
        name="accuracy",
        weight=0.30,  # Adjust weights based on domain importance
        description="Correctness and factual accuracy of the output",
        scale_description="0.0 = Major errors, 1.0 = Completely accurate"
    ),
    QualityDimension(
        name="completeness",
        weight=0.25,
        description="Coverage of all requirements and aspects",
        scale_description="0.0 = Missing key components, 1.0 = Comprehensive"
    ),
    QualityDimension(
        name="clarity",
        weight=0.20,
        description="Clarity of explanations and structure",
        scale_description="0.0 = Unclear/confusing, 1.0 = Crystal clear"
    ),
    QualityDimension(
        name="efficiency",
        weight=0.15,
        description="Efficiency and optimization of the solution",
        scale_description="0.0 = Inefficient approach, 1.0 = Highly optimized"
    ),
    QualityDimension(
        name="creativity",
        weight=0.10,
        description="Innovation and creative problem-solving",
        scale_description="0.0 = Generic solution, 1.0 = Highly innovative"
    )
]

# Test scenarios for research (complexity progression)
TEST_SCENARIOS = {
    "simple": [
        {
            "id": "simple_example",
            "input": "Simple test case for your domain...",
            "expected_components": ["component1", "component2"]
        }
    ],
    "medium": [
        {
            "id": "medium_example", 
            "input": "Medium complexity test case...",
            "expected_components": ["component1", "component2", "component3"]
        }
    ],
    "complex": [
        {
            "id": "complex_example",
            "input": "Complex test case with multiple requirements...",
            "expected_components": ["component1", "component2", "component3", "component4"]
        }
    ]
}
```

### **Step 3: Implement Producer and Critic Agents**

> 💡 **For Beginners:** Agents are like specialized AI assistants. The **Producer** creates content (like a writer), and the **Critic** reviews it (like an editor). This separation prevents the AI from being too lenient on its own work.

**📚 Essential Resources:**
- **[Google ADK Agent Guide](https://google.github.io/adk-docs/agents/)** - Official agent documentation
- **[LLM Agent Patterns](https://blog.langchain.dev/reflection-agents/)** - LangChain blog on reflection
- **[Prompt Engineering Guide](https://www.promptingguide.ai/)** - Writing effective AI instructions
- **[Python Async/Await Tutorial](https://realpython.com/async-io-python/)** - Understanding async programming

**🎯 Key Concepts:**
- **Producer Agent**: Generates initial output (like a domain expert)
- **Critic Agent**: Reviews and suggests improvements (like a quality reviewer)
- **Tools**: Special functions agents can use (like calculators or databases)
- **Instructions**: The "personality" and expertise you give each agent

Create `use_cases/your_use_case/agents.py`:

```python
from typing import Dict, Any, List
from framework.base_agent import BaseUseCaseAgent
from google.adk.tools import FunctionTool
from .tools.your_domain_tools import (
    # Import your domain-specific tools
    domain_specific_function_1,
    domain_specific_function_2,
    analysis_function,
    validation_function
)
import structlog

logger = structlog.get_logger(__name__)

class YourUseCaseProducer(BaseUseCaseAgent):
    """
    Generates high-quality outputs for your specific domain.
    
    Design principles:
    1. Domain expertise and specialized knowledge
    2. Structured output format for evaluation
    3. Tool integration for enhanced capabilities
    4. Research-oriented logging and metrics
    """
    
    def __init__(self, model: str):
        super().__init__(model, "your_use_case", "producer")
    
    def _initialize_tools(self) -> List[FunctionTool]:
        """Initialize domain-specific tools for the producer."""
        return [
            FunctionTool(func=domain_specific_function_1),
            FunctionTool(func=domain_specific_function_2),
            FunctionTool(func=analysis_function),
            # Add tools that help generate better outputs
        ]
    
    def _get_instructions(self) -> str:
        """Instructions for the domain-specific producer."""
        return """
        You are a senior expert in [YOUR DOMAIN] with [X]+ years of experience.
        
        EXAMPLE DOMAINS & EXPERTISE:
        
        📝 CONTENT WRITING:
        - Content strategy and audience analysis
        - SEO optimization and keyword research  
        - Brand voice and tone consistency
        - Editorial standards and fact-checking
        
        💻 CODE REVIEW:
        - Software architecture and design patterns
        - Security vulnerability assessment
        - Performance optimization techniques
        - Code maintainability and readability
        
        📊 DATA ANALYSIS:
        - Statistical analysis and hypothesis testing
        - Data visualization and storytelling
        - Machine learning model evaluation
        - Business intelligence and insights
        
        🎯 STRATEGIC PLANNING:
        - Business strategy and competitive analysis
        - Market research and customer insights
        - Financial modeling and ROI analysis
        - Risk assessment and mitigation
        
        When generating outputs:
        1. Analyze requirements thoroughly
        2. Apply domain expertise and best practices
        3. Create structured, comprehensive responses
        4. Include specific examples and recommendations
        5. Address potential challenges and solutions
        6. Provide clear rationale for decisions
        
        Your output must be structured with these sections:
        - Executive Summary
        - Requirements Analysis
        - [Domain-Specific Section 1]
        - [Domain-Specific Section 2]
        - [Domain-Specific Section 3]
        - Quality Assurance
        - Implementation Guidance
        - Best Practices
        
        Always justify your decisions with specific reasoning.
        Use your tools to enhance the quality and accuracy of your output.
        """

class YourUseCaseCritic(BaseUseCaseAgent):
    """
    Reviews and critiques outputs for improvements.
    
    Design principles:
    1. Objective evaluation (prevents cognitive bias)
    2. Domain expertise for accurate assessment
    3. Structured feedback for actionable improvements
    4. Research-oriented termination conditions
    """
    
    def __init__(self, model: str):
        super().__init__(model, "your_use_case", "critic")
    
    def _initialize_tools(self) -> List[FunctionTool]:
        """Initialize tools for domain-specific critique."""
        return [
            FunctionTool(func=validation_function),
            FunctionTool(func=analysis_function),
            # Add tools that help evaluate quality
        ]
    
    def _get_instructions(self) -> str:
        """Instructions for the domain-specific critic."""
        return """
        You are a principal expert and technical reviewer in [YOUR DOMAIN] with expertise in:
        - [Domain review expertise 1]
        - [Domain review expertise 2]
        - [Quality assessment standards]
        - [Industry benchmarks]
        - [Best practice validation]
        
        Your role is to critically evaluate outputs and provide structured feedback.
        
        For each review:
        1. Assess accuracy and technical correctness
        2. Evaluate completeness and coverage
        3. Review clarity and structure
        4. Analyze efficiency and optimization
        5. Check for missing components
        6. Validate against best practices
        
        Your critique must be structured with:
        - Overall Assessment (EXCELLENT/GOOD/NEEDS_IMPROVEMENT/POOR)
        - Specific Issues Found (categorized by severity: CRITICAL/HIGH/MEDIUM/LOW)
        - Improvement Recommendations (with specific actions)
        - Missing Components or Considerations
        - Best Practice Violations
        - [Domain-Specific Assessment Areas]
        
        Be thorough, objective, and constructive in your feedback.
        Prioritize issues by impact on quality and effectiveness.
        
        Termination condition: Respond with "OUTPUT_APPROVED" if the output meets all requirements and follows best practices with no critical issues.
        """
```

### **Step 4: Implement Domain-Specific Tools**

> 💡 **For Beginners:** Tools are special functions that agents can use to enhance their capabilities. Think of them as "superpowers" - like giving a writer access to a spell-checker, or giving a code reviewer access to security scanners.

**📚 Tool Development Resources:**
- **[ADK Tools Documentation](https://google.github.io/adk-docs/tools/)** - Official tool guide
- **[Function Tools Tutorial](https://google.github.io/adk-docs/tutorials/adding-tools/)** - Step-by-step tool creation
- **[Tool Best Practices](https://google.github.io/adk-docs/tools/best-practices/)** - Design guidelines

**🛠️ Common Tool Types:**
- **Analysis Tools**: Analyze data, check quality, validate inputs
- **External APIs**: Weather, stock prices, news, databases
- **Calculations**: Math, statistics, financial modeling
- **Validation**: Check formats, verify facts, test compliance

**💡 Tool Design Tips:**
- Keep tools **focused** (one clear purpose)
- Make them **reliable** (handle errors gracefully)
- Include **good documentation** (clear descriptions)
- Return **structured data** (dictionaries with consistent format)

Create `use_cases/your_use_case/tools/your_domain_tools.py`:

```python
from typing import Dict, Any, List, Optional
import structlog

logger = structlog.get_logger(__name__)

def domain_specific_function_1(input_param: str, **kwargs) -> Dict[str, Any]:
    """
    Domain-specific tool function.
    
    Args:
        input_param: Description of the parameter
        **kwargs: Additional context from agent execution
    
    Returns:
        Dict containing tool results and metadata
    """
    try:
        # Implement your domain-specific logic
        result = {
            "success": True,
            "data": "Tool-specific output",
            "metadata": {
                "tool_name": "domain_specific_function_1",
                "execution_time": "measurement if needed",
                "parameters_used": {"input_param": input_param}
            }
        }
        
        logger.info(
            "Domain tool executed successfully",
            tool_name="domain_specific_function_1",
            input_param=input_param
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Domain tool execution failed",
            tool_name="domain_specific_function_1",
            error=str(e)
        )
        return {
            "success": False,
            "error": str(e),
            "data": None
        }

def validation_function(output: str, requirements: str, **kwargs) -> Dict[str, Any]:
    """
    Validate output against requirements.
    
    Args:
        output: The generated output to validate
        requirements: Original requirements
        **kwargs: Additional context
    
    Returns:
        Validation results with scores and feedback
    """
    # Implement domain-specific validation logic
    return {
        "validation_score": 0.8,  # 0.0 - 1.0
        "issues_found": ["List of specific issues"],
        "recommendations": ["List of improvements"],
        "compliance_status": "PASS/FAIL",
        "details": "Detailed validation analysis"
    }

# Add more domain-specific tools as needed
```

### **Step 5: Implement Use Case Orchestrator**

Create `use_cases/your_use_case/orchestrator.py`:

```python
from typing import Dict, Any
import structlog
from framework.base_orchestrator import BaseUseCaseOrchestrator
from .agents import YourUseCaseProducer, YourUseCaseCritic
from .config import USE_CASE_CONFIG

logger = structlog.get_logger(__name__)

class YourUseCaseOrchestrator(BaseUseCaseOrchestrator):
    """
    Orchestrates your domain-specific workflow with producer-critic pattern.
    """
    
    def __init__(self, use_case_config: Dict[str, Any] = None):
        config = use_case_config or USE_CASE_CONFIG
        super().__init__(config)
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize domain-specific agents."""
        default_model = "gemini-2.5-flash-lite"
        
        agents = {
            "producer": YourUseCaseProducer(default_model),
            "critic": YourUseCaseCritic(default_model)
        }
        
        logger.info(
            "Domain agents initialized",
            use_case=self.use_case,
            producer_model=agents["producer"].model,
            critic_model=agents["critic"].model
        )
        
        return agents
    
    def _should_terminate(self, state: Dict[str, Any]) -> bool:
        """
        Domain-specific termination conditions.
        
        Customize based on your domain's quality requirements.
        """
        # Check if critic approved the output
        critic_output = state.get("critic_output", {})
        if isinstance(critic_output, str) and "OUTPUT_APPROVED" in critic_output:
            logger.info("Output approved by critic, terminating reflection")
            return True
        
        # Check for structured critic response
        if isinstance(critic_output, dict):
            overall_assessment = critic_output.get("overall_assessment", "")
            if overall_assessment == "EXCELLENT":
                logger.info("Output rated as excellent, terminating reflection")
                return True
            
            # Check if no critical issues remain
            critical_issues = critic_output.get("critical_issues", [])
            if not critical_issues:
                high_issues = critic_output.get("high_issues", [])
                if len(high_issues) <= 1:  # Domain-specific threshold
                    logger.info("No critical issues, terminating reflection")
                    return True
        
        # Check improvement stagnation (optional)
        iteration_count = state.get("reflection:iteration", 0)
        if iteration_count >= 2:
            quality_history = state.get("quality_history", [])
            if len(quality_history) >= 2:
                recent_improvement = quality_history[-1] - quality_history[-2]
                if recent_improvement < 0.05:  # Less than 5% improvement
                    logger.info("Quality improvement stagnated, terminating reflection")
                    return True
        
        return False
    
    def prepare_producer_input(self, original_input: str, 
                             critique: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare input for producer agent, incorporating critique if available.
        """
        producer_input = {
            "requirements": original_input,
            "iteration_context": {
                "is_refinement": critique is not None,
                "previous_critique": critique
            }
        }
        
        if critique:
            producer_input["improvement_focus"] = [
                "Address critical issues: " + ", ".join(critique.get("critical_issues", [])),
                "Resolve high priority issues: " + ", ".join(critique.get("high_issues", [])),
                "Implement recommendations: " + ", ".join(critique.get("recommendations", []))
            ]
            
            producer_input["refinement_instructions"] = f"""
            This is a refinement iteration for {self.use_case}. Focus on:
            1. Addressing all critical and high-priority issues from the critique
            2. Implementing specific recommendations provided
            3. Maintaining the good aspects of the previous output
            4. Improving overall quality while preserving working components
            """
        
        return producer_input
```

### **Step 6: Implement Quality Evaluator**

> 💡 **For Beginners:** The evaluator measures how good the output is across different aspects (like grading a paper on content, grammar, structure, etc.). This provides the research data to compare different approaches.

**📚 Quality Assessment Resources:**
- **[Multi-Criteria Decision Analysis](https://en.wikipedia.org/wiki/Multiple-criteria_decision_analysis)** - Theory behind multi-dimensional evaluation
- **[Rubric Design Guide](https://www.cmu.edu/teaching/assessment/assesslearning/rubrics.html)** - Creating evaluation criteria
- **[Inter-rater Reliability](https://www.scribbr.com/methodology/inter-rater-reliability/)** - Ensuring consistent evaluation

**🎯 Quality Dimension Examples:**
- **Technical Domains**: Accuracy, completeness, efficiency, security
- **Creative Domains**: Originality, engagement, clarity, relevance  
- **Analytical Domains**: Rigor, insight, actionability, evidence
- **Communication Domains**: Clarity, persuasiveness, tone, structure

**💡 Evaluation Tips:**
- Use **0.0 to 1.0 scale** for consistency
- Define **clear criteria** for each score level
- Weight dimensions by **importance** to your domain
- Include **specific feedback** for improvement

Create `use_cases/your_use_case/evaluator.py`:

```python
from typing import Dict, Any, List, Optional
import structlog
from framework.base_evaluator import BaseUseCaseEvaluator, QualityScore
from .config import QUALITY_DIMENSIONS

logger = structlog.get_logger(__name__)

class YourUseCaseEvaluator(BaseUseCaseEvaluator):
    """
    Evaluates output quality for your specific domain.
    """
    
    def __init__(self, evaluator_model: str = None):
        super().__init__("your_use_case", evaluator_model)
    
    def _get_quality_dimensions(self) -> List[QualityDimension]:
        """Get domain-specific quality dimensions."""
        return QUALITY_DIMENSIONS
    
    async def evaluate_output(self, 
                            output: Dict[str, Any], 
                            original_input: str,
                            context: Optional[Dict[str, Any]] = None) -> List[QualityScore]:
        """
        Evaluate output quality across all dimensions.
        
        Customize this method for your domain's specific evaluation needs.
        """
        try:
            # Extract the actual output content
            output_text = output.get("response", str(output))
            
            # Prepare evaluation prompt for the evaluation agent
            evaluation_prompt = f"""
            Evaluate this {self.use_case} output across multiple quality dimensions.
            
            Original Requirements:
            {original_input}
            
            Generated Output:
            {output_text}
            
            Evaluate across these dimensions: {[d.name for d in self.dimensions]}
            
            For each dimension, provide:
            1. Score (0.0 to 1.0)
            2. Reasoning for the score
            3. Specific issues found
            4. Improvement suggestions
            
            Format your response as a structured analysis for each dimension.
            """
            
            # Execute evaluation (you may need to implement this differently)
            # For now, using fallback scores based on output characteristics
            scores = []
            
            for dimension in self.dimensions:
                # Implement domain-specific scoring logic
                score = self._calculate_dimension_score(dimension, output_text, original_input)
                scores.append(score)
            
            logger.info(
                "Quality evaluation completed",
                use_case=self.use_case,
                dimension_count=len(scores),
                overall_score=self.calculate_overall_score(scores)
            )
            
            return scores
            
        except Exception as e:
            logger.error(
                "Quality evaluation failed",
                use_case=self.use_case,
                error=str(e)
            )
            # Return fallback scores
            return [
                QualityScore(
                    dimension=dim.name,
                    score=0.5,  # Neutral fallback
                    reasoning=f"Evaluation failed: {str(e)}",
                    specific_issues=["Evaluation system error"],
                    improvement_suggestions=["Fix evaluation system"]
                )
                for dim in self.dimensions
            ]
    
    def _calculate_dimension_score(self, 
                                 dimension: QualityDimension,
                                 output_text: str,
                                 original_input: str) -> QualityScore:
        """
        Calculate score for a specific quality dimension.
        
        Implement domain-specific scoring logic here.
        """
        # Example scoring logic - customize for your domain
        if dimension.name == "completeness":
            # Score based on output length and requirement coverage
            score = min(1.0, len(output_text) / 1000)  # Adjust threshold
            reasoning = f"Output length: {len(output_text)} characters"
            issues = ["Too brief"] if score < 0.5 else []
            suggestions = ["Provide more detail"] if score < 0.5 else []
            
        elif dimension.name == "clarity":
            # Score based on structure and readability
            has_sections = any(marker in output_text for marker in ["##", "###", "**", "1.", "2."])
            score = 0.8 if has_sections else 0.4
            reasoning = f"Structured format: {has_sections}"
            issues = ["Lacks clear structure"] if not has_sections else []
            suggestions = ["Add headings and sections"] if not has_sections else []
            
        else:
            # Default scoring - implement specific logic for each dimension
            score = 0.5
            reasoning = f"Default scoring for {dimension.name}"
            issues = []
            suggestions = ["Implement specific scoring logic"]
        
        return QualityScore(
            dimension=dimension.name,
            score=score,
            reasoning=reasoning,
            specific_issues=issues,
            improvement_suggestions=suggestions
        )
```

### **Step 7: Create Docker Configuration**

Create `use_cases/your_use_case/docker/Dockerfile`:

```dockerfile
# Multi-stage build using pip with pyproject.toml (proven approach)
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./

# Install dependencies using pip (which can read pyproject.toml)
RUN pip install --no-cache-dir -e .[dev]

# Production stage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_CASE=your_use_case
ENV API_PORT=8002  # Use next available port

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy the source code
COPY . .

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting Your Use Case API Server"\n\
echo "Port: ${API_PORT:-8002}"\n\
echo "Use Case: ${USE_CASE:-your_use_case}"\n\
\n\
export USE_CASE=your_use_case\n\
export API_PORT=${API_PORT:-8002}\n\
\n\
echo "Python: $(python --version)"\n\
echo "Starting API server on port ${API_PORT}..."\n\
exec uvicorn api.use_case_server:app \\\n\
    --host 0.0.0.0 \\\n\
    --port ${API_PORT} \\\n\
    --log-level info \\\n\
    --access-log' > /entrypoint.sh && chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

EXPOSE ${API_PORT}
CMD ["/entrypoint.sh"]
```

Create `use_cases/your_use_case/docker/docker-compose.yml`:

```yaml
services:
  your_use_case:
    container_name: your_use_case_container
    build:
      context: ../../..  # Build from project root
      dockerfile: use_cases/your_use_case/docker/Dockerfile
    env_file: ../../../.env
    ports:
      - "127.0.0.1:8002:8002"  # Use next available port
    environment:
      - USE_CASE=your_use_case
      - API_PORT=8002
    networks:
      - research_framework_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  your_use_case_research:
    container_name: your_use_case_research
    build:
      context: ../../..
      dockerfile: use_cases/your_use_case/docker/Dockerfile
    env_file: ../../../.env
    ports:
      - "127.0.0.1:8892:8888"  # Jupyter port for this use case
    environment:
      - USE_CASE=your_use_case
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=your_use_case_research_token
    volumes:
      - ../../../research/data:/app/research/data
      - ../../../research/notebooks:/app/research/notebooks
    networks:
      - research_framework_network
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
    profiles: ["research"]

networks:
  research_framework_network:
    external: true
```

### **Step 8: Update Main Docker Compose**

Add your use case to the main `docker-compose.yml`:

```yaml
# Add to the main docker-compose.yml services section
  your_use_case:
    container_name: your_use_case_container
    build:
      context: .
      dockerfile: use_cases/your_use_case/docker/Dockerfile
    env_file: .env
    ports:
      - "127.0.0.1:8002:8002"
    environment:
      - USE_CASE=your_use_case
      - API_PORT=8002
    networks:
      - research_framework_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### **Step 9: Update Port Allocation**

Update `docs/PORT_ALLOCATION.md`:

```markdown
## Port Allocation Strategy

| Port | Service | Use Case | Purpose |
|------|---------|----------|---------|
| 8000 | Main API | Framework | Cross-use-case orchestration |
| 8001 | System Design | system_design | Cloud architecture design |
| 8002 | Your Use Case | your_use_case | [Your domain description] |
| 8003 | Available | - | Next use case |
| 8888 | Jupyter Lab | research | Data analysis |
| 8891 | System Design Research | system_design | Research notebooks |
| 8892 | Your Use Case Research | your_use_case | Research notebooks |
```

---

## 🧪 **Testing Your New Use Case**

### **Step 1: Build and Start**

```bash
# Build your use case container
docker-compose up -d --build your_use_case

# Check health
curl http://localhost:8002/health

# Check available endpoints
curl http://localhost:8002/
```

### **Step 2: Test Basic Functionality**

```bash
# Test chat mode (producer only)
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Your domain-specific test request...",
    "mode": "chat",
    "model": "gemini-2.5-flash-lite"
  }'
```

### **Step 3: Test Baseline Mode**

```bash
# Test baseline mode (with quality evaluation)
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Your domain-specific test request...",
    "mode": "baseline", 
    "model": "gemini-2.5-flash-lite"
  }'
```

### **Step 4: Test Reflection Mode**

```bash
# Test reflection mode (producer-critic iterations)
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Your domain-specific test request...",
    "mode": "reflection",
    "model": "gemini-2.5-flash-lite",
    "reflection_iterations": 3
  }'
```

### **Step 5: Validate Research Metrics**

Check that your response includes:
```json
{
  "response": "Domain-specific output...",
  "mode_used": "reflection",
  "model_used": "gemini-2.5-flash-lite",
  "quality_score": 0.75,
  "reflection_iterations_used": 2,
  "processing_time_seconds": 45.2,
  "session_id": "uuid",
  "use_case": "your_use_case"
}
```

---

## 🎯 **Domain-Specific Examples**

### **Code Review Use Case**

```python
# use_cases/code_review/config.py
QUALITY_DIMENSIONS = [
    QualityDimension("correctness", 0.30, "Code correctness and bug detection"),
    QualityDimension("security", 0.25, "Security vulnerability identification"),
    QualityDimension("maintainability", 0.20, "Code maintainability and readability"),
    QualityDimension("performance", 0.15, "Performance optimization suggestions"),
    QualityDimension("best_practices", 0.10, "Adherence to coding standards")
]

# Producer: Senior software engineer reviewing code
# Critic: Principal engineer validating the review
# Tools: Static analysis, security scanning, performance profiling
```

### **Content Generation Use Case**

```python
# use_cases/content_generation/config.py
QUALITY_DIMENSIONS = [
    QualityDimension("relevance", 0.25, "Relevance to target audience and purpose"),
    QualityDimension("engagement", 0.25, "Engagement and readability"),
    QualityDimension("accuracy", 0.20, "Factual accuracy and credibility"),
    QualityDimension("creativity", 0.15, "Originality and creative approach"),
    QualityDimension("structure", 0.15, "Organization and flow")
]

# Producer: Content strategist and writer
# Critic: Editorial reviewer and fact-checker
# Tools: SEO analysis, readability scoring, fact-checking
```

### **Strategic Planning Use Case**

```python
# use_cases/strategic_planning/config.py
QUALITY_DIMENSIONS = [
    QualityDimension("strategic_alignment", 0.30, "Alignment with business objectives"),
    QualityDimension("feasibility", 0.25, "Practical feasibility and resource requirements"),
    QualityDimension("risk_assessment", 0.20, "Risk identification and mitigation"),
    QualityDimension("innovation", 0.15, "Innovation and competitive advantage"),
    QualityDimension("measurability", 0.10, "Clear metrics and success criteria")
]

# Producer: Senior strategy consultant
# Critic: Executive advisor and risk assessor  
# Tools: Market analysis, competitive intelligence, financial modeling
```

---

## 🔧 **Advanced Customization**

### **Custom Tool Integration**

```python
# For complex tools requiring external APIs or databases
class DomainSpecificTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="domain_specific_tool",
            description="Tool for domain-specific analysis"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Implement complex tool logic
        # Can include external API calls, database queries, etc.
        pass
```

### **Custom Evaluation Logic**

```python
# For domains requiring specialized evaluation
async def custom_evaluation_agent(self, output: str, requirements: str) -> Dict[str, Any]:
    """Use a separate ADK agent for quality evaluation."""
    
    evaluation_agent = LlmAgent(
        model="gemini-2.5-pro",  # Use higher model for evaluation
        name=f"{self.use_case}_evaluator",
        instruction="Detailed evaluation instructions...",
        tools=[]
    )
    
    # Execute evaluation agent
    result = await self._run_agent(evaluation_agent, {
        "output": output,
        "requirements": requirements
    })
    
    return result
```

### **Domain-Specific Termination Conditions**

```python
def _should_terminate(self, state: Dict[str, Any]) -> bool:
    """Customize termination logic for your domain."""
    
    # Example: Code review termination
    if self.use_case == "code_review":
        critic_output = state.get("critic_output", {})
        security_issues = critic_output.get("security_issues", [])
        if security_issues:
            return False  # Never terminate with security issues
    
    # Example: Content generation termination
    elif self.use_case == "content_generation":
        quality_score = state.get("quality_score", 0)
        if quality_score < 0.7:
            return False  # Require higher quality for content
    
    # Call parent termination logic
    return super()._should_terminate(state)
```

---

## 📊 **Research Integration**

### **Experiment Configuration**

Create `config/experiments/your_use_case_pilot.json`:

```json
{
  "name": "your_use_case_pilot_study",
  "description": "Pilot study comparing reflection vs capability for [your domain]",
  "use_case": "your_use_case",
  "models": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
  "modes": ["baseline", "reflection"],
  "reflection_iterations": [1, 2, 3],
  "test_scenarios": ["simple", "medium", "complex"],
  "evaluation_dimensions": [
    "accuracy", "completeness", "clarity", "efficiency", "creativity"
  ],
  "sample_size": 10,
  "randomization": true
}
```

### **Research Notebook Template**

Create `research/notebooks/your_use_case_analysis.ipynb`:

```python
# Jupyter notebook for analyzing your use case results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load experiment results
results_df = pd.read_json('research/data/experiments/your_use_case_results.json')

# Analyze quality vs processing time
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='processing_time_seconds', y='quality_score', 
                hue='mode', style='model')
plt.title('Quality vs Processing Time: Your Use Case')
plt.show()

# Compare baseline vs reflection
baseline_scores = results_df[results_df['mode'] == 'baseline']['quality_score']
reflection_scores = results_df[results_df['mode'] == 'reflection']['quality_score']

print(f"Baseline mean quality: {baseline_scores.mean():.3f}")
print(f"Reflection mean quality: {reflection_scores.mean():.3f}")
print(f"Quality improvement: {(reflection_scores.mean() - baseline_scores.mean()):.3f}")
```

---

## 🚀 **Deployment & Usage**

### **Independent Deployment**

```bash
# Deploy just your use case
cd use_cases/your_use_case
docker-compose up -d

# Or deploy with main framework
cd ../../..
docker-compose up -d your_use_case
```

### **API Usage Examples**

```python
import requests

# Test your use case API
response = requests.post('http://localhost:8002/chat', json={
    "message": "Your domain-specific request...",
    "mode": "reflection",
    "model": "gemini-2.5-flash-lite",
    "reflection_iterations": 2
})

result = response.json()
print(f"Quality Score: {result['quality_score']}")
print(f"Processing Time: {result['processing_time_seconds']}s")
print(f"Iterations Used: {result['reflection_iterations_used']}")
```

### **Research Data Collection**

```python
# Collect data for your research
import asyncio
import json

async def run_experiment(scenarios, models, modes):
    results = []
    
    for scenario in scenarios:
        for model in models:
            for mode in modes:
                result = await test_use_case(scenario, model, mode)
                results.append(result)
    
    # Save results for analysis
    with open(f'research/data/{use_case}_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
```

---

## 🎓 **Best Practices**

### **1. Agent Design**
- **Clear personas**: Define specific expertise and experience levels
- **Structured outputs**: Require consistent formatting for evaluation
- **Domain tools**: Integrate relevant tools for enhanced capabilities
- **Termination keywords**: Use consistent approval language

### **2. Quality Evaluation**
- **Domain relevance**: Choose dimensions that matter for your field
- **Weighted importance**: Reflect real-world priorities
- **Measurable criteria**: Define clear scoring guidelines
- **Expert validation**: Test with domain professionals

### **3. Research Design**
- **Progressive complexity**: Simple → Medium → Complex scenarios
- **Realistic constraints**: Budget, time, resource limitations
- **Professional scenarios**: Real-world applicability
- **Comparative baselines**: Multiple model and mode combinations

### **4. Development Workflow**
- **Start simple**: Basic functionality first, then add complexity
- **Test incrementally**: Validate each component before integration
- **Monitor performance**: Track quality, time, and resource usage
- **Document thoroughly**: Clear instructions for future developers

---

## 🆘 **Beginner-Friendly Troubleshooting**

### **"Help! My Agent Won't Work!" 🤔**

#### **Problem 1: Import Errors**
```bash
# Error: ModuleNotFoundError: No module named 'google.adk'
```
**Solution:**
1. Check your `pyproject.toml` includes `google-adk>=1.14.1`
2. Rebuild your Docker container: `docker-compose up -d --build your_use_case`
3. Verify installation: `docker exec your_container pip list | grep google-adk`

**📚 Learn More:** [Python Import System](https://realpython.com/python-import/)

#### **Problem 2: "Agent Won't Execute"**
```bash
# Error: 'LlmAgent' object has no attribute 'run'
```
**Solution:**
```python
# ❌ Wrong way:
result = await agent.run(input_data)

# ✅ Correct way (use our base_agent.py):
result = await producer.run({"input": "Your message here"})
```

**📚 Learn More:** [Google ADK Runner Documentation](https://google.github.io/adk-docs/runners/)

#### **Problem 3: Docker Build Failures**
```bash
# Error: failed to solve: process "/bin/sh -c pip install..." did not complete
```
**Solution:**
1. Copy the working `Dockerfile` from `use_cases/system_design/docker/Dockerfile`
2. Update only the `USE_CASE` and `API_PORT` environment variables
3. Use the proven pip-based approach (not UV or Poetry)

**📚 Learn More:** [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

#### **Problem 4: API Not Responding**
```bash
# Error: curl: (7) Failed to connect to localhost:8002
```
**Solution:**
1. Check container status: `docker ps`
2. Check logs: `docker logs your_container`
3. Verify port mapping in `docker-compose.yml`
4. Test health endpoint: `curl http://localhost:8002/health`

### **"My Quality Scores Are Wrong!" 📊**

#### **Problem 1: All Scores Are 0.5**
This means your evaluation is using fallback scores.

**Solution:**
1. Check evaluator logs for errors
2. Verify quality dimensions are properly defined
3. Test evaluation logic with simple inputs
4. Use fallback scoring as a starting point

#### **Problem 2: Weights Don't Add Up**
```python
# ❌ Wrong: weights sum to 1.2
QualityDimension("accuracy", 0.4, ...),
QualityDimension("clarity", 0.4, ...),
QualityDimension("completeness", 0.4, ...)

# ✅ Correct: weights sum to 1.0  
QualityDimension("accuracy", 0.5, ...),
QualityDimension("clarity", 0.3, ...),
QualityDimension("completeness", 0.2, ...)
```

### **"My Reflection Isn't Working!" 🔄**

#### **Problem 1: Only 1 Iteration Used**
This is often **correct behavior**! The critic approves good outputs early.

**To Test:**
- Use vague prompts to force multiple iterations
- Check critic instructions include termination keywords
- Look for "DESIGN_APPROVED" or "EXCELLENT" in critic responses

#### **Problem 2: Infinite Iterations**
The critic never approves the output.

**Solution:**
1. Check critic instructions include termination condition
2. Add maximum iteration limits (safety net)
3. Review critic's evaluation criteria (might be too strict)

### **"I Don't Understand the Code!" 😵‍💫**

#### **Start Here:**
1. **Copy the system_design use case** as a template
2. **Change only the domain-specific parts** (instructions, tools)
3. **Test each component separately** before integrating
4. **Use our working examples** as reference

#### **Key Files to Understand:**
```python
# Start with these (easiest to modify):
config.py          # Define what your domain does
agents.py          # Change the instructions and expertise

# Then move to these (more complex):  
tools/             # Add domain-specific capabilities
evaluator.py       # Customize quality measurement
orchestrator.py    # Usually no changes needed
```

## 🔧 **Common Beginner Mistakes & Solutions**

### **Mistake 1: Overcomplicating Instructions**
```python
# ❌ Too complex:
"You are an expert with deep knowledge of advanced methodologies..."

# ✅ Clear and specific:
"You are a senior software engineer. Review code for bugs, security issues, and best practices."
```

### **Mistake 2: Unclear Quality Dimensions**
```python
# ❌ Vague:
QualityDimension("goodness", 0.5, "How good it is")

# ✅ Specific:
QualityDimension("technical_accuracy", 0.5, "Correctness of technical decisions and implementations")
```

### **Mistake 3: Missing Error Handling**
```python
# ❌ No error handling:
def my_tool(input_data):
    return expensive_api_call(input_data)

# ✅ Robust error handling:
def my_tool(input_data):
    try:
        result = expensive_api_call(input_data)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Tool failed: {e}")
        return {"success": False, "error": str(e)}
```

### **Mistake 4: Forgetting to Test**
```bash
# ✅ Always test each step:
# 1. Test agent creation
curl http://localhost:8002/health

# 2. Test basic chat
curl -X POST http://localhost:8002/chat -d '{"message":"test","mode":"chat"}'

# 3. Test baseline mode  
curl -X POST http://localhost:8002/chat -d '{"message":"test","mode":"baseline"}'

# 4. Test reflection mode
curl -X POST http://localhost:8002/chat -d '{"message":"test","mode":"reflection"}'
```

## 🎓 **Learning Resources by Experience Level**

### **Complete Beginner (Never built AI agents)**
1. **[AI Agents 101](https://www.deeplearning.ai/short-courses/)** - DeepLearning.AI courses
2. **[Python for AI](https://realpython.com/learning-paths/python-ai-machine-learning/)** - Python skills
3. **[Our Chapter 4](./Chapter%204_%20Reflection.txt)** - Reflection pattern explained
4. **Copy system_design** and modify step by step

### **Some Python Experience**
1. **[Google ADK Quickstart](https://google.github.io/adk-docs/getting-started/)**
2. **[FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/first-steps/)**
3. **[Docker for Python](https://testdriven.io/blog/dockerizing-flask-with-postgres-gunicorn-and-nginx/)**
4. **Follow this guide** step by step

### **Experienced Developer**
1. **[ADK Advanced Patterns](https://google.github.io/adk-docs/agents/multi-agents/)**
2. **[Research Methodology](./Iterative%20Reflection%20vs.txt)**
3. **[Framework Architecture](./COMPREHENSIVE_BUILD_SUMMARY.md)**
4. **Customize and extend** based on your needs

## 🎯 **Success Path for Beginners**

### **Week 1: Learn the Basics**
- [ ] Read Chapter 4_ Reflection.txt (understand the pattern)
- [ ] Watch AI agents tutorial videos (understand the concepts)
- [ ] Set up development environment (Docker, Python, IDE)
- [ ] Run the existing system_design use case (see it work)

### **Week 2: Copy and Modify**
- [ ] Copy system_design folder to your_use_case
- [ ] Change the domain in config.py (your expertise area)
- [ ] Modify agent instructions (your domain knowledge)
- [ ] Test basic functionality (chat mode)

### **Week 3: Customize and Enhance**
- [ ] Add domain-specific tools (if needed)
- [ ] Customize quality dimensions (what matters in your domain)
- [ ] Test baseline and reflection modes
- [ ] Debug and refine based on results

### **Week 4: Research and Deploy**
- [ ] Run comparison experiments (baseline vs reflection)
- [ ] Analyze results and gather insights
- [ ] Deploy for expert evaluation
- [ ] Document findings and share with community

## 📞 **Getting Help**

### **When You're Stuck:**

#### **1. Check the Reference Implementation**
- Look at `use_cases/system_design/` files
- See how system design solves similar problems
- Copy working patterns and adapt them

#### **2. Use the Debug Approach**
```bash
# Test each component separately:
# 1. Can you create the agent?
# 2. Can you execute it in chat mode?
# 3. Does the quality evaluation work?
# 4. Does reflection mode work?
```

#### **3. Start Simple, Add Complexity**
```python
# Begin with minimal implementation:
# - No tools initially
# - Simple quality dimensions  
# - Basic agent instructions
# - Add features incrementally
```

#### **4. Community Resources**
- **[Google ADK GitHub](https://github.com/google/adk)** - Issues and discussions
- **[ADK Community](https://google.github.io/adk-docs/community/)** - Official community
- **[AI Agent Discord/Slack](https://discord.gg/langchain)** - LangChain community
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/google-adk)** - Technical questions

## 🎉 **Beginner Success Stories**

### **"I Built My First Use Case!"**
*"I followed the guide to create a content generation use case. The Producer writes blog posts, and the Critic reviews them for SEO and engagement. It works great!"*

**Key Success Factors:**
- Started with system_design as template
- Focused on clear instructions first
- Added tools gradually
- Tested each step thoroughly

### **"My Research Results Are Amazing!"**
*"I compared reflection vs baseline for code review. Reflection found 40% more bugs with the same model! Publishing at next conference."*

**Key Success Factors:**
- Chose measurable quality dimensions
- Ran systematic experiments
- Validated with expert developers
- Documented methodology clearly

---

## 📚 **Reference Implementation**

The **System Design use case** serves as the reference implementation. Study these files:

- `use_cases/system_design/agents.py` - Producer-Critic agent implementation
- `use_cases/system_design/orchestrator.py` - Workflow management
- `use_cases/system_design/evaluator.py` - Quality assessment
- `use_cases/system_design/config.py` - Configuration and test scenarios
- `use_cases/system_design/tools/cloud_pricing.py` - Domain tools
- `use_cases/system_design/docker/Dockerfile` - Container configuration

---

## 🎯 **Success Criteria**

Your new use case is ready when:

- ✅ **Producer agent** generates domain-appropriate outputs
- ✅ **Critic agent** provides objective evaluation and improvement suggestions
- ✅ **Quality evaluation** works across your defined dimensions
- ✅ **Reflection mode** shows iterative improvement
- ✅ **Baseline comparison** demonstrates research capabilities
- ✅ **Docker deployment** works independently
- ✅ **API endpoints** respond with structured research data
- ✅ **Performance metrics** are captured accurately

---

## 🚀 **Contributing Back**

When you create new use cases:

1. **Document your approach** in this guide
2. **Share quality dimensions** that work well for your domain
3. **Contribute tools** that might be useful for other domains
4. **Report issues** and improvements to the framework
5. **Publish research results** to advance the field

---

## 📞 **Support**

For questions about extending the framework:

1. **Study the reference implementation** (system_design use case)
2. **Check the comprehensive build summary** for technical details
3. **Review ADK documentation** for agent development patterns
4. **Test incrementally** and validate each component
5. **Use the debugging techniques** documented in the main guide

---

## 🎉 **Conclusion**

This framework makes it easy to extend research to new domains while maintaining:
- **Consistent methodology** across use cases
- **Comparable results** for cross-domain analysis  
- **Professional quality** outputs for real-world validation
- **Research rigor** with measurable outcomes

**Your new use case will contribute to the broader understanding of reflection vs capability trade-offs across different AI application domains!** 🎯✨

---

## 🚀 **Complete Beginner Tutorial: Content Generation Use Case**

### **Let's Build Your First Use Case Together!**

We'll create a **blog post generation** use case step-by-step. Perfect for beginners!

#### **Step 1: Set Up the Files**
```bash
# Create the directory structure
mkdir -p use_cases/content_generation/{tools,docker}
cd use_cases/content_generation

# Create the required files
touch __init__.py config.py agents.py orchestrator.py evaluator.py
touch tools/__init__.py tools/content_tools.py
touch docker/Dockerfile docker/docker-compose.yml
```

#### **Step 2: Define What We're Building (config.py)**
```python
# use_cases/content_generation/config.py
from typing import Dict, Any, List
from framework.base_evaluator import QualityDimension

# What does this use case do?
USE_CASE_CONFIG = {
    "name": "content_generation",
    "description": "AI-powered blog post and article generation with editorial review",
    "complexity_levels": ["simple", "medium", "complex"],
    "evaluation_dimensions": ["relevance", "engagement", "accuracy", "creativity", "structure"]
}

# How do we measure quality? (Like a grading rubric)
QUALITY_DIMENSIONS = [
    QualityDimension(
        name="relevance",
        weight=0.25,  # 25% of total score
        description="How well the content matches the target audience and purpose",
        scale_description="0.0 = Completely off-topic, 1.0 = Perfectly relevant"
    ),
    QualityDimension(
        name="engagement", 
        weight=0.25,  # 25% of total score
        description="How engaging and interesting the content is to read",
        scale_description="0.0 = Boring/hard to read, 1.0 = Highly engaging"
    ),
    QualityDimension(
        name="accuracy",
        weight=0.20,  # 20% of total score
        description="Factual accuracy and credibility of information",
        scale_description="0.0 = Major factual errors, 1.0 = Completely accurate"
    ),
    QualityDimension(
        name="creativity",
        weight=0.15,  # 15% of total score
        description="Originality and creative approach to the topic",
        scale_description="0.0 = Generic/cliché, 1.0 = Highly original"
    ),
    QualityDimension(
        name="structure",
        weight=0.15,  # 15% of total score
        description="Organization, flow, and readability of the content",
        scale_description="0.0 = Poor structure, 1.0 = Excellent organization"
    )
]

# What scenarios will we test?
TEST_SCENARIOS = {
    "simple": [
        {
            "id": "basic_blog_post",
            "input": "Write a 500-word blog post about the benefits of remote work for productivity.",
            "expected_components": ["introduction", "main_points", "conclusion", "engaging_tone"]
        }
    ],
    "medium": [
        {
            "id": "technical_article", 
            "input": "Write a 1000-word technical article explaining machine learning to business executives, including practical applications and ROI considerations.",
            "expected_components": ["executive_summary", "technical_explanation", "business_value", "examples", "action_items"]
        }
    ],
    "complex": [
        {
            "id": "thought_leadership",
            "input": "Write a 1500-word thought leadership article on the future of AI in healthcare, including current challenges, emerging opportunities, regulatory considerations, and a 5-year outlook.",
            "expected_components": ["industry_analysis", "trend_identification", "regulatory_landscape", "predictions", "strategic_recommendations"]
        }
    ]
}
```

#### **Step 3: Create the AI Agents (agents.py)**
```python
# use_cases/content_generation/agents.py
from typing import Dict, Any, List
from framework.base_agent import BaseUseCaseAgent
from google.adk.tools import FunctionTool
import structlog

logger = structlog.get_logger(__name__)

class ContentGenerationProducer(BaseUseCaseAgent):
    """
    The "Writer" - Creates blog posts and articles.
    Think of this as your AI content writer.
    """
    
    def __init__(self, model: str):
        super().__init__(model, "content_generation", "producer")
    
    def _initialize_tools(self) -> List[FunctionTool]:
        """Tools the writer can use (start with none for simplicity)"""
        return []  # We'll add tools later
    
    def _get_instructions(self) -> str:
        """Instructions that define the writer's personality and expertise"""
        return """
        You are a senior content strategist and writer with 10+ years of experience.
        
        Your expertise includes:
        - Content strategy and audience analysis
        - SEO optimization and keyword research
        - Brand voice and tone development
        - Editorial standards and best practices
        - Engagement optimization and conversion writing
        
        When creating content:
        1. Analyze the target audience and purpose
        2. Create compelling headlines and introductions
        3. Structure content for maximum readability
        4. Include relevant examples and actionable insights
        5. Optimize for engagement and shareability
        6. Ensure factual accuracy and credibility
        
        Your output must be structured with these sections:
        - Compelling Headline
        - Executive Summary (for longer pieces)
        - Well-organized Main Content
        - Key Takeaways
        - Call to Action (when appropriate)
        
        Always write in an engaging, professional tone that matches the intended audience.
        """

class ContentGenerationCritic(BaseUseCaseAgent):
    """
    The "Editor" - Reviews and improves content.
    Think of this as your AI editor and fact-checker.
    """
    
    def __init__(self, model: str):
        super().__init__(model, "content_generation", "critic")
    
    def _initialize_tools(self) -> List[FunctionTool]:
        """Tools the editor can use for review"""
        return []  # We'll add tools later
    
    def _get_instructions(self) -> str:
        """Instructions that define the editor's review criteria"""
        return """
        You are a principal editor and content strategist with expertise in:
        - Editorial review and content optimization
        - Fact-checking and accuracy verification
        - Audience engagement and readability analysis
        - SEO and content performance optimization
        - Brand consistency and voice guidelines
        
        Your role is to critically evaluate content and provide structured feedback.
        
        For each review:
        1. Assess relevance to target audience and purpose
        2. Evaluate engagement and readability
        3. Check factual accuracy and credibility
        4. Analyze creativity and originality
        5. Review structure and organization
        6. Identify missing elements or improvements
        
        Your critique must be structured with:
        - Overall Assessment (EXCELLENT/GOOD/NEEDS_IMPROVEMENT/POOR)
        - Specific Issues Found (categorized by severity: CRITICAL/HIGH/MEDIUM/LOW)
        - Improvement Recommendations (with specific actions)
        - Missing Elements
        - Engagement Opportunities
        - SEO and Optimization Suggestions
        
        Be thorough, objective, and constructive in your feedback.
        
        Termination condition: Respond with "CONTENT_APPROVED" if the content meets all requirements and editorial standards with no critical issues.
        """
```

#### **Step 4: Copy the Working Framework Files**

The orchestrator and evaluator are complex - let's copy and modify the working ones:

```bash
# Copy the working orchestrator (minimal changes needed)
cp use_cases/system_design/orchestrator.py use_cases/content_generation/orchestrator.py

# Copy the working evaluator (we'll customize it)
cp use_cases/system_design/evaluator.py use_cases/content_generation/evaluator.py
```

Then make these simple changes:

```python
# In orchestrator.py, change these lines:
from .agents import ContentGenerationProducer, ContentGenerationCritic  # Line ~4
# Change "DESIGN_APPROVED" to "CONTENT_APPROVED" in termination condition

# In evaluator.py, change:
super().__init__("content_generation", evaluator_model)  # In __init__
```

#### **Step 5: Create Docker Configuration**

Copy the working Docker setup:
```bash
# Copy the proven Dockerfile
cp use_cases/system_design/docker/Dockerfile use_cases/content_generation/docker/Dockerfile

# Edit these lines in the Dockerfile:
ENV USE_CASE=content_generation
ENV API_PORT=8002
```

#### **Step 6: Test Your New Use Case**

```bash
# Build and start your use case
docker-compose up -d --build content_generation

# Test it works
curl http://localhost:8002/health

# Test content generation
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a blog post about the benefits of remote work for productivity.",
    "mode": "chat",
    "model": "gemini-2.5-flash-lite"
  }'

# Test reflection mode
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a blog post about the benefits of remote work for productivity.",
    "mode": "reflection", 
    "model": "gemini-2.5-flash-lite",
    "reflection_iterations": 2
  }'
```

**🎉 Congratulations! You've built your first AI agent use case!**

---

## 📖 **Additional Learning Resources**

### **Understanding AI Agent Concepts**
- **[Anthropic's Guide to AI Agents](https://docs.anthropic.com/claude/docs/guide-to-anthropics-prompt-engineering-resources)** - Prompt engineering
- **[OpenAI Agent Patterns](https://platform.openai.com/docs/guides/function-calling)** - Function calling and tools
- **[Multi-Agent Systems](https://arxiv.org/abs/2308.00352)** - Academic paper on agent coordination

### **Advanced Agent Development**
- **[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)** - Alternative agent framework
- **[CrewAI Framework](https://docs.crewai.com/)** - Another multi-agent approach
- **[AutoGen Framework](https://microsoft.github.io/autogen/)** - Microsoft's agent framework

### **Research and Evaluation**
- **[Evaluating LLM Applications](https://www.anthropic.com/research/evaluating-ai-systems)** - Anthropic's evaluation guide
- **[AI Safety and Alignment](https://www.alignmentforum.org/)** - Safety considerations
- **[Empirical Methods in AI](https://aclanthology.org/venues/emnlp/)** - Academic conference for methodology

### **Docker and Deployment**
- **[Docker for Data Science](https://docker-curriculum.com/)** - Docker tutorial
- **[FastAPI Production Guide](https://fastapi.tiangolo.com/deployment/)** - Deployment best practices
- **[Container Security](https://sysdig.com/blog/dockerfile-best-practices/)** - Security considerations

### **Python Development**
- **[Real Python](https://realpython.com/)** - Comprehensive Python tutorials
- **[Python Type Hints](https://mypy.readthedocs.io/en/stable/)** - Type checking with mypy
- **[Async Programming](https://docs.python.org/3/library/asyncio.html)** - Official asyncio documentation

---

## 🎯 **Quick Start for Different Backgrounds**

### **If You're a Researcher**
1. **Focus on methodology** - understand the research design first
2. **Study the quality dimensions** - how we measure improvement
3. **Customize evaluation criteria** - what matters in your field
4. **Run systematic experiments** - collect publishable data

**Key Resources:**
- `Iterative Reflection vs.txt` - Research methodology
- `COMPREHENSIVE_BUILD_SUMMARY.md` - Technical implementation
- Research notebooks in `research/` folder

### **If You're a Developer**
1. **Understand the architecture** - modular, containerized design
2. **Study the ADK patterns** - agent creation and execution
3. **Focus on code quality** - error handling, logging, testing
4. **Optimize performance** - Docker builds, API response times

**Key Resources:**
- `framework/` folder - Core implementation patterns
- `use_cases/system_design/` - Reference implementation
- Docker configurations and best practices

### **If You're a Domain Expert**
1. **Define your expertise** - what knowledge should the agent have?
2. **Create evaluation criteria** - how do you judge quality in your field?
3. **Design test scenarios** - what challenges should the agent handle?
4. **Validate results** - does the output meet professional standards?

**Key Resources:**
- Agent instruction templates in this guide
- Quality dimension examples
- Test scenario frameworks

---

## 📋 **Quick Start Checklist**

### **Phase 1: Setup (Day 1)**
- [ ] 📖 Read Chapter 4_ Reflection.txt (understand the pattern)
- [ ] 🎥 Watch AI agents tutorial video (understand concepts)
- [ ] 🐳 Install Docker and test system_design use case
- [ ] 📁 Create your use case directory structure

### **Phase 2: Basic Implementation (Day 2-3)**
- [ ] 📝 Define your domain in config.py
- [ ] 🤖 Create Producer agent instructions (your domain expert)
- [ ] 🔍 Create Critic agent instructions (your domain reviewer)
- [ ] 🐳 Copy and modify Docker configuration
- [ ] ✅ Test basic chat mode functionality

### **Phase 3: Quality & Reflection (Day 4-5)**
- [ ] 📊 Define quality dimensions for your domain
- [ ] 🔄 Test baseline mode (with quality evaluation)
- [ ] 🔁 Test reflection mode (producer-critic iterations)
- [ ] 🐛 Debug and fix any issues
- [ ] 📈 Validate quality measurements

### **Phase 4: Research & Deploy (Day 6-7)**
- [ ] 🧪 Run comparison experiments (baseline vs reflection)
- [ ] 📊 Analyze results and insights
- [ ] 👥 Deploy for expert evaluation (if applicable)
- [ ] 📝 Document your findings
- [ ] 🌟 Share with the community

**Total Time Investment: ~1 week for beginners, 2-3 days for experienced developers**

---

## 🆘 **Emergency Help Section**

### **"Nothing Works!" Panic Guide**

#### **Step 1: Check the Basics**
```bash
# Is Docker running?
docker --version

# Is the main framework working?
curl http://localhost:8001/health

# Are there any containers running?
docker ps
```

#### **Step 2: Copy What Works**
```bash
# Copy the entire system_design folder
cp -r use_cases/system_design use_cases/my_test_case

# Change just the name and port
# In config.py: change "system_design" to "my_test_case"  
# In docker files: change port 8001 to 8003
# Test if it works before customizing
```

#### **Step 3: Get Help**
- **Discord/Slack**: Join AI development communities
- **Stack Overflow**: Tag questions with `google-adk`, `ai-agents`
- **GitHub Issues**: Check existing issues and discussions
- **Email**: Reach out to the framework maintainers

### **"I'm Lost!" Recovery Guide**

1. **Take a break** - Complex systems take time to understand
2. **Start smaller** - Copy system_design and change just the instructions
3. **Focus on one thing** - Get chat mode working before reflection
4. **Ask for help** - The community is friendly and helpful
5. **Learn incrementally** - You don't need to understand everything at once

**Remember: Every expert was once a beginner! 🌟**

---

## 🎓 **Graduation: You're Ready When...**

- ✅ Your use case responds to chat requests
- ✅ Quality evaluation returns meaningful scores  
- ✅ Reflection mode shows iterative improvement
- ✅ You understand what each file does
- ✅ You can debug issues using logs and testing
- ✅ You're excited to try new domains and experiments!

**Happy researching!** 🔬✨

---

## 📞 **Community & Support**

### **Join the Community**
- **[GitHub Discussions](https://github.com/google/adk/discussions)** - ADK community
- **[Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)** - ML research community
- **[AI Research Discord](https://discord.gg/artificial-intelligence)** - Real-time help
- **[LangChain Community](https://discord.gg/langchain)** - Agent development

### **Contributing Back**
When you succeed:
- 🌟 **Star this repository** - Help others find it
- 📝 **Document your use case** - Share your approach
- 🐛 **Report issues** - Help improve the framework
- 📊 **Share results** - Contribute to research knowledge
- 🎓 **Mentor others** - Help the next generation of researchers

**Building on this framework makes you part of advancing AI agent research! 🚀**
