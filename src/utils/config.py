"""
Configuration management for WarmStart.
Handles loading from YAML files, environment variables, and domain overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    
    # Vector DB
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp"
    use_chromadb: bool = True
    
    # Database
    database_url: str = "sqlite:///./warmstart.db"
    
    # Environment
    environment: str = "development"
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Cost limits
    max_cost_per_experiment: float = 100.0
    max_cost_per_generation: float = 5.0
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 90000
    
    # Monitoring
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    
    # MLflow
    mlflow_tracking_uri: str = "./experiments/mlruns"
    mlflow_experiment_name: str = "warmstart"
    
    # Safety
    enable_content_filter: bool = True
    max_prompt_length: int = 8000


class Config:
    """
    Configuration loader that merges default.yaml, domain configs, and env vars.
    """
    
    def __init__(self, config_dir: Optional[Path] = None, domain: Optional[str] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"
        self.domain = domain
        
        # Load settings from environment
        self.settings = Settings()
        
        # Load default config
        self.config = self._load_yaml(self.config_dir / "default.yaml")
        
        # Apply domain override if specified
        if domain:
            self._apply_domain_override(domain)
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not path.exists():
            return {}
        
        with open(path, 'r') as f:
            content = f.read()
            # Replace environment variable placeholders
            content = self._substitute_env_vars(content)
            return yaml.safe_load(content) or {}
    
    def _substitute_env_vars(self, content: str) -> str:
        """Replace ${VAR_NAME} placeholders with environment variables."""
        import re
        
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return re.sub(r'\$\{([^}]+)\}', replacer, content)
    
    def _apply_domain_override(self, domain: str):
        """Apply domain-specific configuration overrides."""
        domain_config_path = self.config_dir / "domains" / f"{domain}.yaml"
        
        if domain_config_path.exists():
            domain_config = self._load_yaml(domain_config_path)
            self.config = self._deep_merge(self.config, domain_config)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        Example: config.get('llm.tier1.model')
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_llm_config(self, tier: str) -> Dict[str, Any]:
        """Get LLM configuration for a specific tier."""
        llm_config = self.config.get('llm', {}).get(tier, {})
        
        # Add API key based on provider
        provider = llm_config.get('provider')
        if provider == 'openai':
            llm_config['api_key'] = self.settings.openai_api_key
        elif provider == 'anthropic':
            llm_config['api_key'] = self.settings.anthropic_api_key
        elif provider == 'gemini':
            llm_config['api_key'] = self.settings.google_api_key
        elif provider == 'groq':
            llm_config['api_key'] = self.settings.groq_api_key
        
        return llm_config
    
    def get_evolution_config(self) -> Dict[str, Any]:
        """Get genetic algorithm configuration."""
        return self.config.get('evolution', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        rag_config = self.config.get('rag', {})
        
        # Add credentials
        if rag_config.get('vector_db') == 'pinecone':
            rag_config['api_key'] = self.settings.pinecone_api_key
            rag_config['environment'] = self.settings.pinecone_environment
        
        return rag_config
    
    def get_cost_control_config(self) -> Dict[str, Any]:
        """Get cost control configuration."""
        return self.config.get('cost_control', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.config.get('monitoring', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self.config


# Global config instance
_config: Optional[Config] = None


def get_config(domain: Optional[str] = None, reload: bool = False) -> Config:
    """
    Get or create global configuration instance.
    
    Args:
        domain: Domain name for domain-specific config
        reload: Force reload of configuration
    
    Returns:
        Config instance
    """
    global _config
    
    if _config is None or reload or (domain and _config.domain != domain):
        _config = Config(domain=domain)
    
    return _config


def set_config(config: Config):
    """Set global configuration instance."""
    global _config
    _config = config
