from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path

class Settings(BaseSettings):
    """Global framework configuration"""
    
    # Google ADK
    google_api_key: Optional[str] = None
    google_project_id: Optional[str] = None
    default_model: str = "gemini-2.5-flash-lite"
    pro_model: str = "gemini-2.5-pro"
    
    # Research Configuration
    enable_research_mode: bool = False
    max_reflection_iterations: int = 5
    cost_tracking_enabled: bool = True
    experiment_output_dir: Path = Path("research/data/experiments")
    enable_expert_evaluation: bool = True
    
    # Database
    database_url: Optional[str] = None
    research_postgres_db: str = "research_db"
    research_postgres_user: str = "research_user"
    research_postgres_password: str = "research_pass"
    
    # Session management (using in-memory for simplicity)
    use_in_memory_sessions: bool = True
    
    # Cloud Providers (System Design use case)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    azure_subscription_id: Optional[str] = None
    hetzner_api_token: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Available models for research
    available_models: List[str] = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

