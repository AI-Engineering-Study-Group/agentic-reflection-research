from typing import Dict, Any, List
from framework.base_evaluator import QualityDimension

# Use case configuration
USE_CASE_CONFIG = {
    "name": "system_design",
    "description": "Cloud system architecture design with multi-provider cost comparison",
    "complexity_levels": ["simple", "medium", "complex"],
    "evaluation_dimensions": [
        "technical_accuracy",
        "cost_optimization", 
        "security_posture",
        "scalability_design",
        "completeness",
        "clarity"
    ]
}

# Quality dimensions for system design
QUALITY_DIMENSIONS = [
    QualityDimension(
        name="technical_accuracy",
        weight=0.25,
        description="Correctness of technical decisions and service selections",
        scale_description="0.0 = Major technical errors, 1.0 = All technical decisions are sound"
    ),
    QualityDimension(
        name="cost_optimization",
        weight=0.20,
        description="Cost efficiency and optimization across cloud providers",
        scale_description="0.0 = No cost consideration, 1.0 = Optimal cost for requirements"
    ),
    QualityDimension(
        name="security_posture",
        weight=0.20,
        description="Security best practices and compliance considerations",
        scale_description="0.0 = Major security gaps, 1.0 = Comprehensive security design"
    ),
    QualityDimension(
        name="scalability_design",
        weight=0.15,
        description="Ability to handle growth and peak loads",
        scale_description="0.0 = No scalability consideration, 1.0 = Excellent scalability design"
    ),
    QualityDimension(
        name="completeness",
        weight=0.10,
        description="Coverage of all requirements and system aspects",
        scale_description="0.0 = Many missing components, 1.0 = All requirements addressed"
    ),
    QualityDimension(
        name="clarity",
        weight=0.10,
        description="Clarity of explanations and documentation",
        scale_description="0.0 = Unclear explanations, 1.0 = Crystal clear documentation"
    )
]

# Test scenarios for research
TEST_SCENARIOS = {
    "simple": [
        {
            "id": "simple_web_app",
            "input": """
            Design a simple web application system with the following requirements:
            - Expected users: 10,000 monthly active users
            - Basic CRUD operations with user authentication
            - Geographic scope: United States only
            - Budget constraint: $2,000/month
            - Compliance: Basic data privacy (no specific regulations)
            """,
            "expected_components": ["web_server", "database", "authentication", "cdn"]
        }
    ],
    "medium": [
        {
            "id": "ecommerce_platform",
            "input": """
            Design an e-commerce platform with these requirements:
            - Expected users: 100,000 monthly active users
            - Peak traffic: 5x normal during sales events
            - Features: product catalog, shopping cart, payment processing, order management
            - Geographic scope: North America and Europe
            - Budget constraint: $15,000/month
            - Compliance: PCI DSS for payment processing
            - Availability requirement: 99.9% uptime
            """,
            "expected_components": ["microservices", "database_cluster", "payment_gateway", "cdn", "load_balancer"]
        }
    ],
    "complex": [
        {
            "id": "black_friday_scale",
            "input": """
            Design a system for a major e-commerce platform handling Black Friday traffic:
            - Normal traffic: 1M concurrent users
            - Black Friday peak: 10M concurrent users (10x spike)
            - Global reach: North America, Europe, Asia-Pacific
            - Features: real-time inventory, personalized recommendations, payment processing
            - Budget: $100,000/month (can scale up to $500,000 during peak)
            - Compliance: GDPR, PCI DSS, SOX
            - Availability: 99.99% uptime required
            - Performance: <100ms response time for critical paths
            """,
            "expected_components": ["microservices", "auto_scaling", "global_cdn", "multi_region", "real_time_analytics"]
        }
    ]
}

