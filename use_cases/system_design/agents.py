from typing import Dict, Any, List
from framework.base_agent import BaseUseCaseAgent
from google.adk.tools import FunctionTool, google_search
from .tools.cloud_pricing import (
    get_gcp_compute_pricing,
    get_amazon_compute_pricing,
    research_current_information, # Google Search research tool
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
        return [
            FunctionTool(func=get_gcp_compute_pricing),
            FunctionTool(func=get_amazon_compute_pricing),
            FunctionTool(func=research_current_information) # Google Search research
        ]
    
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
        
        Use your tools to enhance your designs:
        - get_gcp_pricing: Get accurate GCP Compute Engine pricing for cost analysis
        - get_amazon_compute_pricing: Compare costs with Amazon EC2 for multi-cloud considerations
        - research_current_information: Get current information via Google Search about:
          * Latest cloud service features and pricing changes
          * Recent security advisories and compliance updates  
          * Performance benchmarks and optimization techniques
          * Industry trends and emerging technologies
        
        Always use the research tool when you need current information that may have changed recently.
        Combine static pricing tools with current research to provide accurate, data-driven recommendations.
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
        return [
            FunctionTool(func=get_gcp_compute_pricing),
            FunctionTool(func=get_amazon_compute_pricing),
            FunctionTool(func=research_current_information) # Google Search research
        ]
    
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
        
        Use your tools to validate and enhance your critique:
        - get_gcp_pricing: Get accurate GCP pricing for cost validation
        - get_amazon_compute_pricing: Compare costs across providers for recommendations
        - research_current_information: Get current information via Google Search about:
          * Latest security vulnerabilities and mitigation strategies
          * Current compliance requirements and standards
          * Performance optimization best practices
          * Cost optimization techniques and pricing updates
        
        Always use the research tool to validate against current best practices and security guidelines.
        Combine validation tools with current research for comprehensive analysis.
        
        Termination condition: Respond with "DESIGN_APPROVED" if the design meets all requirements and follows best practices with no critical issues.
        """

class SystemDesignResearcher(BaseUseCaseAgent):
    """
    Research agent for gathering latest information about cloud technologies and best practices.
    
    Why separate research agent:
    1. Google Search tool cannot be used with other tools in same agent (ADK limitation)
    2. Provides up-to-date information about cloud services and pricing
    3. Can research latest security vulnerabilities and compliance requirements
    4. Enables data-driven design decisions with current information
    """
    
    def __init__(self, model: str):
        super().__init__(model, "system_design", "researcher")
    
    def _initialize_tools(self) -> List[FunctionTool]:
        """Initialize tools for research and information gathering."""
        # Note: google_search is a built-in tool, not a FunctionTool
        # Return empty list since we'll use google_search in agent creation
        return []
    
    def _get_instructions(self) -> str:
        """Instructions for the system design researcher."""
        return """
        You are a cloud technology researcher with expertise in finding the latest information about:
        - Google Cloud Platform services and pricing
        - AWS and Azure competitive offerings
        - Security best practices and compliance requirements
        - Performance optimization techniques
        - Cost optimization strategies
        - Industry trends and emerging technologies
        
        Your role is to research and provide current information to support system design decisions.
        
        When researching:
        1. Search for the most recent information about cloud services and pricing
        2. Look for security advisories and best practices
        3. Find performance benchmarks and optimization techniques
        4. Research compliance requirements for specific industries
        5. Identify cost optimization opportunities and strategies
        
        Provide comprehensive, accurate, and up-to-date information with proper citations.
        Focus on actionable insights that can improve system designs.
        
        Always search for current information rather than relying on potentially outdated knowledge.
        """

