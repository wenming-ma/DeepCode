"""
DeepCode Configuration Settings

Central configuration module for the DeepCode framework, providing settings for
LLM providers, API keys, model configurations, and operational parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Path Configuration
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_DIR = CONFIG_DIR / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOGS_DIR = PROJECT_ROOT / "logs"


# =============================================================================
# Environment Variable Names
# =============================================================================

class EnvVars:
    """Environment variable names for configuration."""
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    BRAVE_API_KEY = "BRAVE_API_KEY"
    GITHUB_TOKEN = "GITHUB_TOKEN"
    LOG_LEVEL = "DEEPCODE_LOG_LEVEL"
    DEBUG_MODE = "DEEPCODE_DEBUG"
    MODEL_PROVIDER = "DEEPCODE_MODEL_PROVIDER"
    MODEL_NAME = "DEEPCODE_MODEL_NAME"


# =============================================================================
# Model Configuration
# =============================================================================

class ModelProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ModelConfig(BaseModel):
    """Configuration for a specific LLM model."""
    provider: ModelProvider = ModelProvider.ANTHROPIC
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=200000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0.0)
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str, info) -> str:
        """Validate model name based on provider."""
        return v


# Default model configurations for different tasks
DEFAULT_MODELS = {
    "blueprint": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514",
        temperature=0.3,
        max_tokens=8192,
    ),
    "code_generation": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514",
        temperature=0.2,
        max_tokens=16384,
    ),
    "verification": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514",
        temperature=0.1,
        max_tokens=8192,
    ),
    "analysis": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514",
        temperature=0.5,
        max_tokens=8192,
    ),
}


# =============================================================================
# API Configuration
# =============================================================================

class APIConfig(BaseModel):
    """API configuration settings."""
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None
    github_token: Optional[str] = None
    
    # API endpoints (for custom deployments)
    anthropic_base_url: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    def model_post_init(self, __context: Any) -> None:
        """Load API keys from environment if not provided."""
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv(EnvVars.ANTHROPIC_API_KEY)
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv(EnvVars.OPENAI_API_KEY)
        if self.brave_api_key is None:
            self.brave_api_key = os.getenv(EnvVars.BRAVE_API_KEY)
        if self.github_token is None:
            self.github_token = os.getenv(EnvVars.GITHUB_TOKEN)
    
    def get_api_key(self, provider: ModelProvider) -> Optional[str]:
        """Get API key for a specific provider."""
        if provider == ModelProvider.ANTHROPIC:
            return self.anthropic_api_key
        elif provider == ModelProvider.OPENAI:
            return self.openai_api_key
        return None
    
    def has_api_key(self, provider: ModelProvider) -> bool:
        """Check if API key is available for provider."""
        return self.get_api_key(provider) is not None


# =============================================================================
# Phase Configuration
# =============================================================================

class Phase1Config(BaseModel):
    """Configuration for Phase 1: Blueprint Generation."""
    # Document parsing
    max_document_pages: int = Field(default=100, ge=1)
    extract_figures: bool = True
    extract_tables: bool = True
    extract_equations: bool = True
    
    # Content segmentation
    min_chunk_size: int = Field(default=100, ge=10)
    max_chunk_size: int = Field(default=5000, ge=100)
    overlap_size: int = Field(default=50, ge=0)
    
    # Agent settings
    concept_keywords: List[str] = Field(default_factory=lambda: [
        "introduction", "method", "overview", "architecture", 
        "framework", "approach", "contribution", "background"
    ])
    algorithm_keywords: List[str] = Field(default_factory=lambda: [
        "algorithm", "hyperparameter", "equation", "training",
        "loss", "optimization", "network", "layer", "model"
    ])
    
    # Web search for reference implementations
    enable_web_search: bool = True
    max_search_results: int = Field(default=5, ge=1, le=20)


class Phase2Config(BaseModel):
    """Configuration for Phase 2: Code Generation."""
    # CodeMem settings
    enable_code_memory: bool = True
    max_memory_entries: int = Field(default=50, ge=1)
    memory_context_limit: int = Field(default=8000, ge=100)
    
    # CodeRAG settings
    enable_code_rag: bool = True
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    rag_complexity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Generation settings
    max_file_size: int = Field(default=50000, ge=1000)  # characters
    max_files_per_batch: int = Field(default=5, ge=1)
    
    # Reference repositories
    reference_repos: List[str] = Field(default_factory=list)


class Phase3Config(BaseModel):
    """Configuration for Phase 3: Verification."""
    # Static analysis
    enable_static_analysis: bool = True
    check_imports: bool = True
    check_types: bool = True
    check_style: bool = True
    max_complexity: int = Field(default=15, ge=1)
    
    # Sandbox execution
    enable_sandbox: bool = True
    sandbox_timeout: int = Field(default=300, ge=10)  # seconds
    max_iterations: int = Field(default=10, ge=1)
    
    # LSP modifications
    enable_lsp_fixes: bool = True
    max_fix_attempts: int = Field(default=5, ge=1)
    
    # Success criteria
    require_no_errors: bool = True
    require_tests_pass: bool = False  # Optional for initial runs


# =============================================================================
# Logging Configuration
# =============================================================================

class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_file: Optional[Path] = None
    log_dir: Path = LOGS_DIR
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    json_format: bool = False
    colorize: bool = True
    console_output: bool = True
    max_file_size: int = Field(default=10 * 1024 * 1024, ge=1024)  # 10MB
    backup_count: int = Field(default=5, ge=0)
    
    def model_post_init(self, __context: Any) -> None:
        """Set default log file path if not provided."""
        if self.log_file is None:
            self.log_file = self.log_dir / "deepcode.log"
        
        # Override from environment
        env_level = os.getenv(EnvVars.LOG_LEVEL)
        if env_level:
            try:
                self.level = LogLevel(env_level.upper())
            except ValueError:
                pass


# =============================================================================
# Output Configuration
# =============================================================================

class OutputConfig(BaseModel):
    """Configuration for output generation."""
    output_dir: Path = OUTPUT_DIR
    create_git_repo: bool = True
    generate_readme: bool = True
    generate_requirements: bool = True
    generate_tests: bool = True
    
    # File formatting
    use_black: bool = True
    use_isort: bool = True
    line_length: int = Field(default=88, ge=40, le=200)
    
    # Export options
    export_blueprint: bool = True
    export_memory: bool = True
    export_rag_index: bool = True


# =============================================================================
# Main Settings Class
# =============================================================================

class DeepCodeSettings(BaseModel):
    """
    Main configuration class for DeepCode framework.
    
    Aggregates all configuration sections and provides methods for
    loading from files and environment variables.
    """
    # Core settings
    debug_mode: bool = Field(default=False)
    verbose: bool = Field(default=False)
    
    # Sub-configurations
    api: APIConfig = Field(default_factory=APIConfig)
    models: Dict[str, ModelConfig] = Field(default_factory=lambda: DEFAULT_MODELS.copy())
    phase1: Phase1Config = Field(default_factory=Phase1Config)
    phase2: Phase2Config = Field(default_factory=Phase2Config)
    phase3: Phase3Config = Field(default_factory=Phase3Config)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # Paths
    project_root: Path = PROJECT_ROOT
    config_dir: Path = CONFIG_DIR
    prompts_dir: Path = PROMPTS_DIR
    cache_dir: Path = CACHE_DIR
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization setup."""
        # Check for debug mode from environment
        debug_env = os.getenv(EnvVars.DEBUG_MODE, "").lower()
        if debug_env in ("1", "true", "yes"):
            self.debug_mode = True
            self.logging.level = LogLevel.DEBUG
        
        # Create necessary directories
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [self.cache_dir, self.output.output_dir, self.logging.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self, task: str) -> ModelConfig:
        """Get model configuration for a specific task."""
        return self.models.get(task, self.models.get("analysis", ModelConfig()))
    
    def get_prompt_path(self, agent_name: str) -> Path:
        """Get path to prompt template for an agent."""
        return self.prompts_dir / f"{agent_name}.yaml"
    
    def has_required_api_keys(self) -> bool:
        """Check if at least one LLM API key is configured."""
        return (
            self.api.has_api_key(ModelProvider.ANTHROPIC) or
            self.api.has_api_key(ModelProvider.OPENAI)
        )
    
    def get_default_provider(self) -> ModelProvider:
        """Get the default model provider based on available API keys."""
        # Check environment override
        env_provider = os.getenv(EnvVars.MODEL_PROVIDER, "").lower()
        if env_provider == "openai" and self.api.has_api_key(ModelProvider.OPENAI):
            return ModelProvider.OPENAI
        
        # Default to Anthropic if available
        if self.api.has_api_key(ModelProvider.ANTHROPIC):
            return ModelProvider.ANTHROPIC
        elif self.api.has_api_key(ModelProvider.OPENAI):
            return ModelProvider.OPENAI
        
        # Fallback to Anthropic (will fail later if no key)
        return ModelProvider.ANTHROPIC
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DeepCodeSettings":
        """Load settings from a YAML file."""
        import yaml
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Settings file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "DeepCodeSettings":
        """Create settings from environment variables."""
        return cls()
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save settings to a YAML file."""
        import yaml
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict, handling Path objects
        data = self.model_dump()
        
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            return obj
        
        data = convert_paths(data)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()


# =============================================================================
# Global Settings Instance
# =============================================================================

# Singleton settings instance
_settings: Optional[DeepCodeSettings] = None


def get_settings() -> DeepCodeSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = DeepCodeSettings()
    return _settings


def configure(settings: Optional[DeepCodeSettings] = None, **kwargs: Any) -> DeepCodeSettings:
    """
    Configure global settings.
    
    Args:
        settings: Optional pre-configured settings instance
        **kwargs: Override specific settings
    
    Returns:
        The configured settings instance
    """
    global _settings
    
    if settings is not None:
        _settings = settings
    elif kwargs:
        _settings = DeepCodeSettings(**kwargs)
    else:
        _settings = DeepCodeSettings()
    
    return _settings


def reset_settings() -> None:
    """Reset settings to default."""
    global _settings
    _settings = None


# =============================================================================
# Convenience Functions
# =============================================================================

def get_api_key(provider: ModelProvider) -> Optional[str]:
    """Get API key for a provider from settings."""
    return get_settings().api.get_api_key(provider)


def get_model_config(task: str = "analysis") -> ModelConfig:
    """Get model configuration for a task."""
    return get_settings().get_model_config(task)


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return get_settings().debug_mode


def get_prompt_path(agent_name: str) -> Path:
    """Get path to prompt template for an agent."""
    return get_settings().get_prompt_path(agent_name)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Path constants
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "PROMPTS_DIR",
    "OUTPUT_DIR",
    "CACHE_DIR",
    "LOGS_DIR",
    
    # Environment variables
    "EnvVars",
    
    # Enums
    "ModelProvider",
    "LogLevel",
    
    # Configuration classes
    "ModelConfig",
    "APIConfig",
    "Phase1Config",
    "Phase2Config",
    "Phase3Config",
    "LoggingConfig",
    "OutputConfig",
    "DeepCodeSettings",
    
    # Default configurations
    "DEFAULT_MODELS",
    
    # Functions
    "get_settings",
    "configure",
    "reset_settings",
    "get_api_key",
    "get_model_config",
    "is_debug_mode",
    "get_prompt_path",
]
