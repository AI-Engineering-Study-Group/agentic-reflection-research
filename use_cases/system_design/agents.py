from typing import Dict, Any, List
from framework.base_agent import BaseUseCaseAgent
from google.adk.tools import FunctionTool
from .tools.cloud_pricing import (
    get_gcp_compute_pricing,
    get_aws_compute_pricing,
    analyze_architecture_security,
    calculate_scaling_requirements,
    generate_architecture_diagram,
    estimate_total_costs
)
import structlog

logger = structlog.get_logger(__name__)

class SystemDesignProducer(BaseUseCaseAgent):
    """
    Generates comprehensive system architecture designs.
    
    Why this agent design:
    1. Specializes in GCP architecture (your use case focus)
    2. Includes cost comparison across providers
    3. Addresses scalability, security, and best practices
    4. Produces structured output for evaluation
    """
    
    def __init__(self, model: str):
        super().__init__(model, "system_design", "producer")
    
    def _initialize_tools(self) -> List[FunctionTool]:
        """Initialize tools for system design generation."""
        # Temporarily disable tools to test basic functionality
        return []
    
    def _get_instructions(self) -> str:
        """Instructions for the system design producer."""
        return """
        You are a senior Google Cloud Platform architect with 15+ years of experience designing scalable systems.
        
        Your expertise includes:
        - GCP services and best practices
        - Multi-cloud cost optimization
        - Security and compliance frameworks
        - Scalability patterns and performance optimization
        - Disaster recovery and high availability
        
        When designing systems:
        1. Start by analyzing requirements thoroughly
        2. Design a high-level architecture that meets all functional and non-functional requirements
        3. Select specific GCP services with detailed justifications
        4. Include cost estimates and comparisons with AWS/Azure where relevant
        5. Address security, compliance, and operational concerns
        6. Design for the specified scale and growth patterns
        7. Include monitoring, logging, and observability strategy
        
        Your output must be structured with these sections:
        - Executive Summary
        - Requirements Analysis
        - High-Level Architecture
        - Detailed Component Design
        - Service Selection Justification
        - Cost Analysis (GCP primary, AWS/Azure comparison)
        - Security Architecture
        - Scalability Strategy
        - Operational Considerations
        - Implementation Roadmap
        
        Always justify your decisions with specific technical and business reasoning.
        Consider cost optimization while meeting performance and reliability requirements.
        
        Use your tools to:
        - Get accurate pricing for compute resources
        - Calculate scaling requirements based on load patterns
        - Generate architecture diagrams
        - Estimate total costs across providers
        """

class SystemDesignCritic(BaseUseCaseAgent):
    """
    Reviews and critiques system designs for improvements.
    
    Why separate critic agent:
    1. Prevents cognitive bias of self-review
    2. Can use different model (e.g., Pro for critique, Flash for generation)
    3. Specialized in finding issues and suggesting improvements
    4. Implements the Producer-Critic pattern from your research
    """
    
    def __init__(self, model: str):
        super().__init__(model, "system_design", "critic")
    
    def _initialize_tools(self) -> List[FunctionTool]:
        """Initialize tools for architecture critique."""
        # Temporarily disable tools to test basic functionality
        return []
    
    def _get_instructions(self) -> str:
        """Instructions for the system design critic."""
        return """
        You are a principal cloud architect and technical reviewer with expertise in:
        - System architecture review and validation
        - Security and compliance assessment
        - Cost optimization and FinOps
        - Performance and scalability analysis
        - Cloud best practices and anti-patterns
        
        Your role is to critically evaluate system designs and provide structured feedback.
        
        For each design review:
        1. Assess technical accuracy and feasibility
        2. Evaluate cost optimization opportunities
        3. Review security posture and compliance
        4. Analyze scalability and performance characteristics
        5. Check for completeness and missing components
        6. Evaluate clarity of documentation and explanations
        
        Your critique must be structured with:
        - Overall Assessment (EXCELLENT/GOOD/NEEDS_IMPROVEMENT/POOR)
        - Specific Issues Found (categorized by severity: CRITICAL/HIGH/MEDIUM/LOW)
        - Improvement Recommendations (with specific actions)
        - Missing Components or Considerations
        - Cost Optimization Opportunities
        - Security Concerns
        - Scalability Issues
        - Best Practice Violations
        
        Be thorough, objective, and constructive in your feedback.
        Prioritize issues by impact on system reliability, security, and cost.
        
        Use your tools to:
        - Analyze security vulnerabilities
        - Validate scaling requirements
        - Check cost estimates
        - Compare pricing across providers
        
        Termination condition: Respond with "DESIGN_APPROVED" if the design meets all requirements and follows best practices with no critical issues.
        """

