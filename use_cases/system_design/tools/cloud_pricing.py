from typing import Dict, Any, Optional, List
from google.adk.tools import FunctionTool
from google.adk.tools.base_tool import BaseTool
import httpx
import structlog
from config.settings import settings
import asyncio

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
    
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__()
    
    async def get_pricing(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get pricing for a list of services."""
        raise NotImplementedError

class GCPPricingTool(CloudPricingTool):
    """
    Google Cloud Platform pricing tool.
    
    Why GCP focus: Your use case specializes in GCP with comparison to others.
    """
    
    def __init__(self):
        super().__init__("gcp")
    
    async def get_compute_pricing(self, 
                                instance_type: str,
                                region: str,
                                hours_per_month: int = 730,
                                **kwargs) -> Dict[str, Any]:
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
            
            pricing_data = await self._fetch_gcp_pricing(instance_type, region)
            
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
    
    async def _fetch_gcp_pricing(self, instance_type: str, region: str) -> Dict[str, Any]:
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
        super().__init__("aws")
    
    async def get_compute_pricing(self,
                                instance_type: str,
                                region: str,
                                hours_per_month: int = 730,
                                **kwargs) -> Dict[str, Any]:
        """Get AWS EC2 pricing."""
        try:
            pricing_data = await self._fetch_aws_pricing(instance_type, region)
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
    
    async def _fetch_aws_pricing(self, instance_type: str, region: str) -> Dict[str, Any]:
        """Fetch AWS pricing data."""
        # Mock implementation
        mock_pricing = {
            "m5.xlarge": {"hourly_rate": 0.192, "last_updated": "2024-01-01"},
            "c5.xlarge": {"hourly_rate": 0.170, "last_updated": "2024-01-01"},
            "t3.xlarge": {"hourly_rate": 0.166, "last_updated": "2024-01-01"}
        }
        
        return mock_pricing.get(instance_type, {"hourly_rate": 0.150, "last_updated": "2024-01-01"})

# Create function tools for ADK integration
def get_gcp_compute_pricing(instance_type: str, region: str, hours_per_month: int = 730, **kwargs) -> Dict[str, Any]:
    """
    Get GCP Compute Engine pricing for system design.
    
    Args:
        instance_type: GCP instance type (e.g., 'e2-standard-4')
        region: GCP region (e.g., 'us-central1') 
        hours_per_month: Operating hours per month (default: 730)
    
    Returns:
        Dict containing pricing information including hourly and monthly costs
    """
    tool = GCPPricingTool()
    return asyncio.run(tool.get_compute_pricing(instance_type, region, hours_per_month))

def get_aws_compute_pricing(instance_type: str, region: str, hours_per_month: int = 730, **kwargs) -> Dict[str, Any]:
    """
    Get AWS EC2 pricing for cost comparison.
    
    Args:
        instance_type: AWS instance type (e.g., 'm5.xlarge')
        region: AWS region (e.g., 'us-east-1')
        hours_per_month: Operating hours per month (default: 730)
    
    Returns:
        Dict containing pricing information including hourly and monthly costs
    """
    tool = AWSPricingTool()
    return asyncio.run(tool.get_compute_pricing(instance_type, region, hours_per_month))

# Additional tools for comprehensive system design
def analyze_architecture_security(architecture: Dict[str, Any], **kwargs) -> Dict[str, Any]:
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

def calculate_scaling_requirements(base_load: int, peak_multiplier: float, **kwargs) -> Dict[str, Any]:
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

def generate_architecture_diagram(architecture: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Generate architecture diagram specification."""
    return {
        "diagram_type": "system_architecture",
        "components": architecture.get("components", []),
        "connections": architecture.get("data_flows", []),
        "layers": ["presentation", "application", "data", "infrastructure"],
        "diagram_url": f"https://example.com/diagrams/{architecture.get('id', 'default')}.png"
    }

def estimate_total_costs(services: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
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

