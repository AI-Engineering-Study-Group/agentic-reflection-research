from typing import Dict, Any, Optional, List
from google.adk.tools import FunctionTool
from google.adk.tools.base_tool import BaseTool
import httpx
import structlog
from config.settings import settings
import asyncio
import concurrent.futures

logger = structlog.get_logger(__name__)

class CloudPricingTool(BaseTool):
    """
    Base class for cloud provider pricing tools.
    
    Why separate tools for each provider:
    1. Different API structures and authentication
    2. Provider-specific service mappings
    3. Easier to maintain and update
    4. Enables accurate cost comparisons
    """
    
    def __init__(self, provider: str, name: str, description: str):
        self.provider = provider
        super().__init__(name=name, description=description)
    
    async def get_pricing(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get pricing for a list of services."""
        raise NotImplementedError

class GCPPricingTool(CloudPricingTool):
    """
    Google Cloud Platform pricing tool.
    
    Why GCP focus: Your use case specializes in GCP with comparison to others.
    """
    
    def __init__(self):
        super().__init__("gcp", "GCP Pricing Tool", "Get Google Cloud Platform pricing information")
    
    def get_compute_pricing(self, 
                             instance_type: str,
                             region: str,
                             hours_per_month: int) -> Dict[str, Any]:
        """
        Get GCP Compute Engine pricing.
        
        Args:
            instance_type: e.g., "e2-standard-4"
            region: e.g., "us-central1"
            hours_per_month: Operating hours (default: 730 for full month)
        """
        try:
            # In real implementation, call GCP Pricing API
            # For now, return mock data with realistic structure
            
            pricing_data = self._fetch_gcp_pricing(instance_type, region)
            
            monthly_cost = pricing_data["hourly_rate"] * hours_per_month
            
            result = {
                "provider": "gcp",
                "service": "compute_engine",
                "instance_type": instance_type,
                "region": region,
                "hourly_rate": pricing_data["hourly_rate"],
                "monthly_cost": monthly_cost,
                "currency": "USD",
                "last_updated": pricing_data["last_updated"]
            }
            
            logger.info(
                "GCP compute pricing retrieved",
                instance_type=instance_type,
                region=region,
                monthly_cost=monthly_cost
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to get GCP pricing",
                instance_type=instance_type,
                region=region,
                error=str(e)
            )
            raise
    
    def _fetch_gcp_pricing(self, instance_type: str, region: str) -> Dict[str, Any]:
        """
        Fetch pricing from static pricing database.
        
        Using static pricing data ensures:
        1. Consistent results across research experiments
        2. No API rate limits or failures
        3. Reliable cost comparisons
        4. Based on real GCP pricing as of January 2025
        """
        # Comprehensive GCP pricing data (January 2025)
        gcp_pricing = {
            # Compute Engine - General Purpose
            "e2-standard-2": {"hourly_rate": 0.067, "vcpus": 2, "memory": 8},
            "e2-standard-4": {"hourly_rate": 0.134, "vcpus": 4, "memory": 16},
            "e2-standard-8": {"hourly_rate": 0.268, "vcpus": 8, "memory": 32},
            "e2-standard-16": {"hourly_rate": 0.536, "vcpus": 16, "memory": 64},
            
            # Compute Engine - High Memory
            "n1-highmem-2": {"hourly_rate": 0.118, "vcpus": 2, "memory": 13},
            "n1-highmem-4": {"hourly_rate": 0.237, "vcpus": 4, "memory": 26},
            "n1-highmem-8": {"hourly_rate": 0.474, "vcpus": 8, "memory": 52},
            
            # Compute Engine - High CPU
            "n1-highcpu-16": {"hourly_rate": 0.568, "vcpus": 16, "memory": 14.4},
            "n1-highcpu-32": {"hourly_rate": 1.136, "vcpus": 32, "memory": 28.8},
            
            # Compute Engine - Standard
            "n1-standard-1": {"hourly_rate": 0.047, "vcpus": 1, "memory": 3.75},
            "n1-standard-2": {"hourly_rate": 0.095, "vcpus": 2, "memory": 7.5},
            "n1-standard-4": {"hourly_rate": 0.190, "vcpus": 4, "memory": 15},
            "n1-standard-8": {"hourly_rate": 0.380, "vcpus": 8, "memory": 30},
            
            # Compute Optimized
            "c2-standard-4": {"hourly_rate": 0.212, "vcpus": 4, "memory": 16},
            "c2-standard-8": {"hourly_rate": 0.424, "vcpus": 8, "memory": 32},
            "c2-standard-16": {"hourly_rate": 0.848, "vcpus": 16, "memory": 64},
        }
        
        # Regional pricing adjustments (simplified)
        region_multipliers = {
            "us-central1": 1.0,
            "us-east1": 1.0,
            "us-west1": 1.05,
            "europe-west1": 1.08,
            "asia-east1": 1.12,
        }
        
        base_pricing = gcp_pricing.get(instance_type, {"hourly_rate": 0.100, "vcpus": 2, "memory": 8})
        multiplier = region_multipliers.get(region, 1.0)
        
        return {
            "hourly_rate": base_pricing["hourly_rate"] * multiplier,
            "vcpus": base_pricing.get("vcpus", 2),
            "memory_gb": base_pricing.get("memory", 8),
            "last_updated": "2025-01-01",
            "region_multiplier": multiplier
        }

class AWSPricingTool(CloudPricingTool):
    """AWS pricing tool for cost comparison."""
    
    def __init__(self):
        super().__init__("aws", "AWS Pricing Tool", "Get Amazon Web Services pricing information")
    
    def get_compute_pricing(self,
                             instance_type: str,
                             region: str,
                             hours_per_month: int) -> Dict[str, Any]:
        """Get AWS EC2 pricing."""
        try:
            pricing_data = self._fetch_aws_pricing(instance_type, region)
            monthly_cost = pricing_data["hourly_rate"] * hours_per_month
            
            return {
                "provider": "aws",
                "service": "ec2",
                "instance_type": instance_type,
                "region": region,
                "hourly_rate": pricing_data["hourly_rate"],
                "monthly_cost": monthly_cost,
                "currency": "USD",
                "last_updated": pricing_data["last_updated"]
            }
            
        except Exception as e:
            logger.error("Failed to get AWS pricing", error=str(e))
            raise
    
    def _fetch_aws_pricing(self, instance_type: str, region: str) -> Dict[str, Any]:
        """Fetch AWS pricing data."""
        # Mock implementation
        mock_pricing = {
            "m5.xlarge": {"hourly_rate": 0.192, "last_updated": "2024-01-01"},
            "c5.xlarge": {"hourly_rate": 0.170, "last_updated": "2024-01-01"},
            "t3.xlarge": {"hourly_rate": 0.166, "last_updated": "2024-01-01"}
        }
        
        return mock_pricing.get(instance_type, {"hourly_rate": 0.150, "last_updated": "2024-01-01"})

# TEMPORARY: Ultra-simple test functions
def test_ultra_simple_cloud(instance_type: str, region: str, hours_per_month: int) -> Dict[str, Any]:
    """Ultra simple cloud pricing test function."""
    return {
        "provider": "test_cloud",
        "instance_type": instance_type,
        "region": region,
        "monthly_cost": hours_per_month * 0.150,
        "currency": "USD",
        "test": "ultra_simple"
    }

# Create function tools for ADK integration with enhanced descriptions
def get_gcp_compute_pricing(instance_type: str, region: str, hours_per_month: int, **kwargs) -> Dict[str, Any]:
    """
    Get GCP Compute Engine pricing for system design.
    
    Args:
        instance_type: GCP instance type (e.g., 'e2-standard-4', 'n1-standard-2', 'c2-standard-8')
        region: GCP region (e.g., 'us-central1', 'us-east1', 'europe-west1', 'asia-east1') 
        hours_per_month: Operating hours per month (typically 730 for full month)
    
    Returns:
        Dict containing pricing information including hourly and monthly costs, vCPUs, and memory
    
    Example usage:
        get_gcp_compute_pricing("e2-standard-4", "us-central1", 730)
        # Returns: {"provider": "gcp", "instance_type": "e2-standard-4", "monthly_cost": 97.82, ...}
    """
    # Input validation
    if not instance_type or not isinstance(instance_type, str):
        raise ValueError("instance_type must be a non-empty string")
    if not region or not isinstance(region, str):
        raise ValueError("region must be a non-empty string")
    if not isinstance(hours_per_month, int) or hours_per_month <= 0:
        raise ValueError("hours_per_month must be a positive integer")
    
    try:
        tool = GCPPricingTool()
        return tool.get_compute_pricing(instance_type, region, hours_per_month)
    except Exception as e:
        logger.error(f"Failed to get GCP pricing: {str(e)}")
        return {
            "error": f"Failed to get GCP pricing: {str(e)}",
            "provider": "gcp",
            "instance_type": instance_type,
            "region": region
        }

def get_amazon_compute_pricing(instance_type: str, region: str, hours_per_month: int, **kwargs) -> Dict[str, Any]:
    """
    Get AWS EC2 pricing for cost comparison.
    
    Args:
        instance_type: AWS instance type (e.g., 'm5.xlarge', 'c5.xlarge', 't3.xlarge')
        region: AWS region (e.g., 'us-east-1', 'us-west-2', 'eu-west-1')
        hours_per_month: Operating hours per month (typically 730 for full month)
    
    Returns:
        Dict containing pricing information including hourly and monthly costs
    
    Example usage:
        get_aws_compute_pricing("m5.xlarge", "us-east-1", 730)
        # Returns: {"provider": "aws", "instance_type": "m5.xlarge", "monthly_cost": 140.16, ...}
    """
    # Input validation
    if not instance_type or not isinstance(instance_type, str):
        raise ValueError("instance_type must be a non-empty string")
    if not region or not isinstance(region, str):
        raise ValueError("region must be a non-empty string")
    if not isinstance(hours_per_month, int) or hours_per_month <= 0:
        raise ValueError("hours_per_month must be a positive integer")
    
    try:
        tool = AWSPricingTool()
        return tool.get_compute_pricing(instance_type, region, hours_per_month)
    except Exception as e:
        logger.error(f"Failed to get AWS pricing: {str(e)}")
        return {
            "error": f"Failed to get AWS pricing: {str(e)}",
            "provider": "aws",
            "instance_type": instance_type,
            "region": region
        }

# Additional tools for comprehensive system design
def analyze_architecture_security(architecture: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze system architecture for security vulnerabilities and best practices.
    
    Args:
        architecture: System architecture specification
    
    Returns:
        Dict containing security analysis and recommendations
    """
    # Mock implementation - replace with actual security analysis
    return {
        "security_score": 0.8,
        "vulnerabilities": [
            "Database not in private subnet",
            "Missing WAF configuration"
        ],
        "recommendations": [
            "Move database to private subnet",
            "Configure Web Application Firewall",
            "Enable encryption at rest"
        ],
        "compliance_status": {
            "pci_dss": "partial",
            "gdpr": "compliant",
            "sox": "not_applicable"
        }
    }

def calculate_scaling_requirements(base_load: int, peak_multiplier: float) -> Dict[str, Any]:
    """
    Calculate scaling requirements for handling peak loads.
    
    Args:
        base_load: Base concurrent users
        peak_multiplier: Peak load multiplier (e.g., 10 for 10x traffic)
    
    Returns:
        Dict containing scaling recommendations
    """
    peak_load = base_load * peak_multiplier
    
    return {
        "base_load": base_load,
        "peak_load": peak_load,
        "scaling_strategy": "auto_scaling_groups" if peak_multiplier > 3 else "manual_scaling",
        "recommended_instances": {
            "base": max(2, base_load // 1000),
            "peak": max(4, peak_load // 1000)
        },
        "scaling_triggers": [
            f"CPU > 70% for 2 minutes",
            f"Memory > 80% for 2 minutes",
            f"Request latency > 200ms"
        ]
    }

def generate_architecture_diagram(architecture: Dict[str, Any]) -> Dict[str, Any]:
    """Generate architecture diagram specification."""
    return {
        "diagram_type": "system_architecture",
        "components": architecture.get("components", []),
        "connections": architecture.get("data_flows", []),
        "layers": ["presentation", "application", "data", "infrastructure"],
        "diagram_url": f"https://example.com/diagrams/{architecture.get('id', 'default')}.png"
    }

def estimate_total_costs(services: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate total system costs across providers."""
    # Mock implementation - replace with actual cost calculation
    gcp_total = sum(service.get("monthly_cost", 0) for service in services if service.get("provider") == "gcp")
    aws_total = gcp_total * 1.15  # AWS typically 15% more expensive
    azure_total = gcp_total * 1.10  # Azure typically 10% more expensive
    
    return {
        "monthly_costs": {
            "gcp": gcp_total,
            "aws": aws_total,
            "azure": azure_total
        },
        "annual_costs": {
            "gcp": gcp_total * 12,
            "aws": aws_total * 12,
            "azure": azure_total * 12
        },
        "cost_breakdown": services,
        "recommendations": [
            "Consider reserved instances for 20-30% savings",
            "Use preemptible instances for non-critical workloads",
            "Implement auto-scaling to optimize costs"
        ]
    }


# Research Tool Implementation
def research_current_information(query: str, **kwargs) -> Dict[str, Any]:
    """
    Research current information using Google Search for system design decisions.
    
    This tool enables access to real-time information about:
    - Latest cloud service features and pricing updates
    - Recent security advisories and compliance requirements  
    - Performance benchmarks and optimization techniques
    - Industry trends and emerging technologies
    
    Args:
        query: The research question or topic to investigate
        **kwargs: Additional parameters including agent_model from tool context
    
    Returns:
        Dict containing research results with sources and recommendations
    """
    from ..agents import SystemDesignResearcher
    import asyncio
    
    # Extract model from kwargs (passed via tool context)
    agent_model = kwargs.get('model', kwargs.get('agent_model', 'gemini-2.5-flash-lite'))
    
    logger.info("Research tool called", query=query, model=agent_model)
    
    try:
        # Create researcher agent using the same model as the calling agent
        researcher = SystemDesignResearcher(agent_model)
        
        # Execute research using the agent with Google Search
        # Since we're already in an async context, we need to use a thread pool
        
        def run_research_sync():
            # Create a new event loop for the research
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(researcher.run({'input': query}))
                return result
            finally:
                loop.close()
        
        # Run the research in a thread to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_research_sync)
            research_result = future.result(timeout=30)  # 30 second timeout
        
        # Extract and format the response
        response_text = research_result.get('response', '')
        
        # Return structured research results
        return {
            "research_query": query,
            "findings": response_text,
            "source": "Google Search via SystemDesignResearcher",
            "grounded": True,
            "timestamp": "current",
            "recommendations": [
                "Review the findings for latest information",
                "Cross-reference with your current architecture",
                "Consider implementing suggested best practices"
            ]
        }
        
    except concurrent.futures.TimeoutError:
        logger.error("Research tool timed out", query=query)
        return {
            "research_query": query,
            "findings": "Research timed out after 30 seconds. Please try a more specific query.",
            "source": "Timeout",
            "grounded": False,
            "error": "Timeout"
        }
    except Exception as e:
        logger.error("Research tool failed", query=query, error=str(e))
        return {
            "research_query": query,
            "findings": f"Research failed: {str(e)}",
            "source": "Error",
            "grounded": False,
            "error": str(e)
        }

