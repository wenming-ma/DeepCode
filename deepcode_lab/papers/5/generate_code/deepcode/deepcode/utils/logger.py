"""
Logger utility module for DeepCode framework.

Provides structured logging with configurable levels, formatters, and handlers
for tracking agent execution, LLM calls, and pipeline progress.
"""

import logging
import sys
import time
import functools
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    def to_logging_level(self) -> int:
        """Convert to standard logging level."""
        return getattr(logging, self.value)


class LogConfig(BaseModel):
    """Configuration for logging setup."""
    
    level: LogLevel = Field(default=LogLevel.INFO, description="Minimum log level")
    log_file: Optional[str] = Field(default=None, description="Path to log file")
    log_dir: Optional[str] = Field(default=None, description="Directory for log files")
    format_string: str = Field(
        default="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        description="Log message format"
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for log messages"
    )
    json_format: bool = Field(
        default=False,
        description="Whether to output logs in JSON format"
    )
    include_caller: bool = Field(
        default=True,
        description="Include caller information in logs"
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum log file size before rotation"
    )
    backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )
    console_output: bool = Field(
        default=True,
        description="Whether to output logs to console"
    )
    colorize: bool = Field(
        default=True,
        description="Whether to colorize console output"
    )


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)


class ColorFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def __init__(self, fmt: str, datefmt: str = None):
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        # Save original levelname
        original_levelname = record.levelname
        
        # Add color to levelname
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return result


class Logger:
    """
    Main logger class for DeepCode framework.
    
    Provides structured logging with support for:
    - Multiple output handlers (console, file)
    - JSON and text formatting
    - Colored console output
    - Execution timing
    - Agent call tracking
    """
    
    _instances: Dict[str, "Logger"] = {}
    _config: Optional[LogConfig] = None
    _initialized: bool = False
    
    def __init__(self, name: str, config: Optional[LogConfig] = None):
        """
        Initialize logger.
        
        Args:
            name: Logger name (typically module name)
            config: Optional logging configuration
        """
        self.name = name
        self._logger = logging.getLogger(name)
        
        # Use provided config or global config
        self._config = config or Logger._config or LogConfig()
        
        # Configure if not already done
        if not Logger._initialized or config is not None:
            self._configure()
    
    def _configure(self) -> None:
        """Configure the logger with handlers and formatters."""
        config = self._config
        
        # Set level
        self._logger.setLevel(config.level.to_logging_level())
        
        # Remove existing handlers to avoid duplicates
        self._logger.handlers.clear()
        
        # Create formatters
        if config.json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                config.format_string,
                datefmt=config.date_format
            )
        
        # Console handler
        if config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(config.level.to_logging_level())
            
            if config.colorize and not config.json_format:
                console_formatter = ColorFormatter(
                    config.format_string,
                    datefmt=config.date_format
                )
                console_handler.setFormatter(console_formatter)
            else:
                console_handler.setFormatter(formatter)
            
            self._logger.addHandler(console_handler)
        
        # File handler
        if config.log_file or config.log_dir:
            log_path = self._get_log_path(config)
            
            # Create directory if needed
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_path,
                    maxBytes=config.max_file_size,
                    backupCount=config.backup_count
                )
            except ImportError:
                file_handler = logging.FileHandler(log_path)
            
            file_handler.setLevel(config.level.to_logging_level())
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self._logger.propagate = False
    
    def _get_log_path(self, config: LogConfig) -> Path:
        """Get the log file path."""
        if config.log_file:
            return Path(config.log_file)
        
        if config.log_dir:
            log_dir = Path(config.log_dir)
            timestamp = datetime.now().strftime("%Y%m%d")
            return log_dir / f"deepcode_{timestamp}.log"
        
        return Path("logs") / "deepcode.log"
    
    def _log(
        self,
        level: int,
        message: str,
        *args,
        extra_data: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs
    ) -> None:
        """Internal logging method."""
        # Create extra dict for additional data
        extra = {}
        if extra_data:
            extra["extra_data"] = extra_data
        
        self._logger.log(level, message, *args, exc_info=exc_info, extra=extra, **kwargs)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, *args, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, *args, exc_info: bool = False, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, *args, exc_info=exc_info, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self._log(logging.ERROR, message, *args, exc_info=True, **kwargs)
    
    def log_agent_start(
        self,
        agent_name: str,
        agent_role: str,
        input_summary: Optional[str] = None
    ) -> None:
        """Log agent execution start."""
        self.info(
            f"Agent started: {agent_name} ({agent_role})",
            extra_data={
                "event": "agent_start",
                "agent_name": agent_name,
                "agent_role": agent_role,
                "input_summary": input_summary
            }
        )
    
    def log_agent_complete(
        self,
        agent_name: str,
        agent_role: str,
        duration_ms: float,
        success: bool,
        output_summary: Optional[str] = None
    ) -> None:
        """Log agent execution completion."""
        level = logging.INFO if success else logging.ERROR
        status = "completed" if success else "failed"
        
        self._log(
            level,
            f"Agent {status}: {agent_name} ({agent_role}) in {duration_ms:.2f}ms",
            extra_data={
                "event": "agent_complete",
                "agent_name": agent_name,
                "agent_role": agent_role,
                "duration_ms": duration_ms,
                "success": success,
                "output_summary": output_summary
            }
        )
    
    def log_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool
    ) -> None:
        """Log LLM API call."""
        level = logging.DEBUG if success else logging.WARNING
        status = "success" if success else "failed"
        
        self._log(
            level,
            f"LLM call {status}: {provider}/{model} - {input_tokens}+{output_tokens} tokens in {latency_ms:.2f}ms",
            extra_data={
                "event": "llm_call",
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "success": success
            }
        )
    
    def log_phase_start(self, phase_name: str, phase_number: int) -> None:
        """Log pipeline phase start."""
        self.info(
            f"Phase {phase_number} started: {phase_name}",
            extra_data={
                "event": "phase_start",
                "phase_name": phase_name,
                "phase_number": phase_number
            }
        )
    
    def log_phase_complete(
        self,
        phase_name: str,
        phase_number: int,
        duration_ms: float,
        success: bool
    ) -> None:
        """Log pipeline phase completion."""
        level = logging.INFO if success else logging.ERROR
        status = "completed" if success else "failed"
        
        self._log(
            level,
            f"Phase {phase_number} {status}: {phase_name} in {duration_ms:.2f}ms",
            extra_data={
                "event": "phase_complete",
                "phase_name": phase_name,
                "phase_number": phase_number,
                "duration_ms": duration_ms,
                "success": success
            }
        )
    
    def log_file_generated(
        self,
        file_path: str,
        line_count: int,
        generation_time_ms: float
    ) -> None:
        """Log file generation."""
        self.info(
            f"Generated file: {file_path} ({line_count} lines) in {generation_time_ms:.2f}ms",
            extra_data={
                "event": "file_generated",
                "file_path": file_path,
                "line_count": line_count,
                "generation_time_ms": generation_time_ms
            }
        )
    
    def log_verification_result(
        self,
        iteration: int,
        errors_found: int,
        errors_fixed: int,
        success: bool
    ) -> None:
        """Log verification iteration result."""
        level = logging.INFO if success else logging.WARNING
        
        self._log(
            level,
            f"Verification iteration {iteration}: {errors_found} errors found, {errors_fixed} fixed",
            extra_data={
                "event": "verification_result",
                "iteration": iteration,
                "errors_found": errors_found,
                "errors_fixed": errors_fixed,
                "success": success
            }
        )


# Type variable for decorator return type preservation
F = TypeVar("F", bound=Callable[..., Any])


def get_logger(name: str, config: Optional[LogConfig] = None) -> Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (typically __name__ of the module)
        config: Optional logging configuration
    
    Returns:
        Logger instance
    """
    if name not in Logger._instances:
        Logger._instances[name] = Logger(name, config)
    return Logger._instances[name]


def setup_logging(config: LogConfig) -> None:
    """
    Initialize global logging configuration.
    
    Args:
        config: Logging configuration to apply globally
    """
    Logger._config = config
    Logger._initialized = True
    
    # Reconfigure existing loggers
    for logger in Logger._instances.values():
        logger._config = config
        logger._configure()


def log_execution_time(
    logger: Optional[Logger] = None,
    level: LogLevel = LogLevel.DEBUG
) -> Callable[[F], F]:
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger to use (creates one if not provided)
        level: Log level for timing messages
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger._log(
                    level.to_logging_level(),
                    f"{func.__name__} completed in {elapsed_ms:.2f}ms"
                )
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"{func.__name__} failed after {elapsed_ms:.2f}ms: {e}",
                    exc_info=True
                )
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger._log(
                    level.to_logging_level(),
                    f"{func.__name__} completed in {elapsed_ms:.2f}ms"
                )
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"{func.__name__} failed after {elapsed_ms:.2f}ms: {e}",
                    exc_info=True
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


def log_agent_call(
    agent_name: Optional[str] = None,
    agent_role: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to log agent execution.
    
    Args:
        agent_name: Name of the agent (defaults to function name)
        agent_role: Role of the agent
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            name = agent_name or func.__name__
            role = agent_role or "unknown"
            
            logger.log_agent_start(name, role)
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.log_agent_complete(name, role, elapsed_ms, success=True)
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.log_agent_complete(name, role, elapsed_ms, success=False)
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            name = agent_name or func.__name__
            role = agent_role or "unknown"
            
            logger.log_agent_start(name, role)
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.log_agent_complete(name, role, elapsed_ms, success=True)
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.log_agent_complete(name, role, elapsed_ms, success=False)
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


# Convenience exports
__all__ = [
    # Classes
    "Logger",
    "LogLevel",
    "LogConfig",
    "JSONFormatter",
    "ColorFormatter",
    # Functions
    "get_logger",
    "setup_logging",
    "log_execution_time",
    "log_agent_call",
]
