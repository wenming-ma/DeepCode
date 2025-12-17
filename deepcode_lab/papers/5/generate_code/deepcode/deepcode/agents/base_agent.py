"""
Base Agent Module for DeepCode Framework.

Provides the abstract base class for all agents in the system, including
prompt template handling, response parsing, and common agent functionality.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
import json
import re
import yaml
from pathlib import Path

from pydantic import BaseModel, Field


class AgentRole(Enum):
    """Enumeration of agent roles in the DeepCode pipeline."""
    DOCUMENT_PARSER = "document_parser"
    CONTENT_SEGMENTER = "content_segmenter"
    CONCEPT_ANALYZER = "concept_analyzer"
    ALGORITHM_EXTRACTOR = "algorithm_extractor"
    PLANNING = "planning"
    CODE_GENERATOR = "code_generator"
    STATIC_ANALYZER = "static_analyzer"
    MODIFICATION = "modification"
    SANDBOX_EXECUTOR = "sandbox_executor"
    ORCHESTRATOR = "orchestrator"


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class AgentConfig(BaseModel):
    """Configuration for an agent instance."""
    name: str = Field(..., description="Agent name")
    role: AgentRole = Field(..., description="Agent role in pipeline")
    model: str = Field(default="claude-3-5-sonnet-20241022", description="LLM model to use")
    temperature: float = Field(default=0.0, description="LLM temperature")
    max_tokens: int = Field(default=8192, description="Maximum tokens in response")
    timeout: float = Field(default=300.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    prompt_template_path: Optional[str] = Field(default=None, description="Path to prompt template")
    
    class Config:
        use_enum_values = True


class AgentMessage(BaseModel):
    """Message structure for agent communication."""
    role: str = Field(..., description="Message role: system, user, assistant")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentResponse(BaseModel):
    """Structured response from an agent."""
    success: bool = Field(..., description="Whether execution succeeded")
    content: Any = Field(default=None, description="Response content")
    raw_response: str = Field(default="", description="Raw LLM response")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    tokens_used: int = Field(default=0, description="Tokens consumed")
    
    @classmethod
    def success_response(cls, content: Any, raw_response: str = "", **metadata) -> "AgentResponse":
        """Create a successful response."""
        return cls(
            success=True,
            content=content,
            raw_response=raw_response,
            metadata=metadata
        )
    
    @classmethod
    def error_response(cls, error: str, **metadata) -> "AgentResponse":
        """Create an error response."""
        return cls(
            success=False,
            error=error,
            metadata=metadata
        )


class PromptTemplate(BaseModel):
    """Template for agent prompts with variable substitution."""
    system_prompt: str = Field(default="", description="System prompt template")
    user_prompt: str = Field(default="", description="User prompt template")
    few_shot_examples: List[Dict[str, str]] = Field(default_factory=list, description="Few-shot examples")
    output_format: str = Field(default="text", description="Expected output format: text, json, yaml")
    variables: List[str] = Field(default_factory=list, description="Required template variables")
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PromptTemplate":
        """Load prompt template from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def render(self, **kwargs) -> Dict[str, str]:
        """Render the template with provided variables."""
        # Check for missing required variables
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required template variables: {missing}")
        
        # Render system prompt
        system = self.system_prompt
        for key, value in kwargs.items():
            system = system.replace(f"{{{{{key}}}}}", str(value))
        
        # Render user prompt
        user = self.user_prompt
        for key, value in kwargs.items():
            user = user.replace(f"{{{{{key}}}}}", str(value))
        
        return {
            "system": system,
            "user": user
        }
    
    def get_messages(self, **kwargs) -> List[AgentMessage]:
        """Get formatted messages for LLM call."""
        rendered = self.render(**kwargs)
        messages = []
        
        if rendered["system"]:
            messages.append(AgentMessage(role="system", content=rendered["system"]))
        
        # Add few-shot examples
        for example in self.few_shot_examples:
            if "user" in example:
                messages.append(AgentMessage(role="user", content=example["user"]))
            if "assistant" in example:
                messages.append(AgentMessage(role="assistant", content=example["assistant"]))
        
        if rendered["user"]:
            messages.append(AgentMessage(role="user", content=rendered["user"]))
        
        return messages


# Type variable for generic response parsing
T = TypeVar('T', bound=BaseModel)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the DeepCode framework.
    
    Provides common functionality for:
    - LLM interaction via configurable interface
    - Prompt template management
    - Response parsing (text, JSON, YAML)
    - Error handling and retries
    - State management
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_interface: Optional[Any] = None,
        prompt_template: Optional[PromptTemplate] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration
            llm_interface: LLM interface for API calls (injected dependency)
            prompt_template: Optional prompt template (can be loaded from config path)
        """
        self.config = config
        self.llm = llm_interface
        self.state = AgentState.IDLE
        self.execution_history: List[Dict[str, Any]] = []
        
        # Load prompt template
        if prompt_template:
            self.prompt_template = prompt_template
        elif config.prompt_template_path:
            self.prompt_template = PromptTemplate.from_yaml(config.prompt_template_path)
        else:
            self.prompt_template = PromptTemplate()
    
    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name
    
    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return AgentRole(self.config.role)
    
    def set_llm_interface(self, llm_interface: Any) -> None:
        """Set or update the LLM interface."""
        self.llm = llm_interface
    
    def set_prompt_template(self, template: PromptTemplate) -> None:
        """Set or update the prompt template."""
        self.prompt_template = template
    
    @abstractmethod
    async def execute(self, **kwargs) -> AgentResponse:
        """
        Execute the agent's main task.
        
        This method must be implemented by all concrete agents.
        
        Args:
            **kwargs: Task-specific arguments
            
        Returns:
            AgentResponse with execution results
        """
        pass
    
    async def call_llm(
        self,
        messages: List[AgentMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call the LLM with the given messages.
        
        Args:
            messages: List of messages to send
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Raw response string from LLM
            
        Raises:
            RuntimeError: If LLM interface is not set
        """
        if self.llm is None:
            raise RuntimeError(f"LLM interface not set for agent {self.name}")
        
        # Convert messages to format expected by LLM interface
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.llm.generate(
            messages=formatted_messages,
            model=self.config.model,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            timeout=self.config.timeout
        )
        
        return response
    
    async def call_with_template(self, **kwargs) -> str:
        """
        Call LLM using the configured prompt template.
        
        Args:
            **kwargs: Variables to substitute in template
            
        Returns:
            Raw response string from LLM
        """
        messages = self.prompt_template.get_messages(**kwargs)
        return await self.call_llm(messages)
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.
        
        Handles common issues like markdown code blocks and trailing content.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed JSON as dictionary
            
        Raises:
            ValueError: If JSON parsing fails
        """
        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # Try parsing the entire response
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {e}")
    
    def parse_yaml_response(self, response: str) -> Dict[str, Any]:
        """
        Parse YAML from LLM response.
        
        Handles common issues like markdown code blocks.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed YAML as dictionary
            
        Raises:
            ValueError: If YAML parsing fails
        """
        # Try to extract YAML from markdown code blocks
        yaml_patterns = [
            r'```yaml\s*([\s\S]*?)\s*```',
            r'```yml\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```'
        ]
        
        for pattern in yaml_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    return yaml.safe_load(match.group(1))
                except yaml.YAMLError:
                    continue
        
        # Try parsing the entire response
        try:
            return yaml.safe_load(response)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML from response: {e}")
    
    def parse_structured_response(
        self,
        response: str,
        model_class: Type[T]
    ) -> T:
        """
        Parse response into a Pydantic model.
        
        Args:
            response: Raw LLM response string
            model_class: Pydantic model class to parse into
            
        Returns:
            Instance of the model class
            
        Raises:
            ValueError: If parsing or validation fails
        """
        # Determine format based on template or try both
        data = None
        
        if self.prompt_template.output_format == "json":
            data = self.parse_json_response(response)
        elif self.prompt_template.output_format == "yaml":
            data = self.parse_yaml_response(response)
        else:
            # Try JSON first, then YAML
            try:
                data = self.parse_json_response(response)
            except ValueError:
                try:
                    data = self.parse_yaml_response(response)
                except ValueError:
                    raise ValueError("Failed to parse response as JSON or YAML")
        
        try:
            return model_class(**data)
        except Exception as e:
            raise ValueError(f"Failed to validate response against {model_class.__name__}: {e}")
    
    def extract_code_blocks(self, response: str, language: Optional[str] = None) -> List[str]:
        """
        Extract code blocks from LLM response.
        
        Args:
            response: Raw LLM response string
            language: Optional language filter (e.g., 'python')
            
        Returns:
            List of extracted code strings
        """
        if language:
            pattern = rf'```{language}\s*([\s\S]*?)\s*```'
        else:
            pattern = r'```(?:\w+)?\s*([\s\S]*?)\s*```'
        
        matches = re.findall(pattern, response)
        return [m.strip() for m in matches if m.strip()]
    
    def extract_single_code_block(self, response: str, language: str = "python") -> str:
        """
        Extract a single code block, preferring the specified language.
        
        Args:
            response: Raw LLM response string
            language: Preferred language
            
        Returns:
            Extracted code string
            
        Raises:
            ValueError: If no code block found
        """
        # Try language-specific first
        blocks = self.extract_code_blocks(response, language)
        if blocks:
            return blocks[0]
        
        # Fall back to any code block
        blocks = self.extract_code_blocks(response)
        if blocks:
            return blocks[0]
        
        raise ValueError(f"No code block found in response")
    
    def log_execution(
        self,
        action: str,
        inputs: Dict[str, Any],
        outputs: Any,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Log an execution step for debugging and analysis.
        
        Args:
            action: Description of the action
            inputs: Input parameters
            outputs: Output results
            success: Whether the action succeeded
            error: Error message if failed
        """
        from datetime import datetime
        
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "inputs": inputs,
            "outputs": outputs,
            "success": success,
            "error": error
        })
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution history."""
        total = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e["success"])
        
        return {
            "agent_name": self.name,
            "agent_role": self.config.role,
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "history": self.execution_history
        }
    
    def reset_state(self) -> None:
        """Reset agent state to idle."""
        self.state = AgentState.IDLE
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history = []


class SyncBaseAgent(BaseAgent):
    """
    Synchronous version of BaseAgent for simpler use cases.
    
    Wraps async methods for synchronous execution.
    """
    
    def execute_sync(self, **kwargs) -> AgentResponse:
        """
        Synchronous wrapper for execute().
        
        Args:
            **kwargs: Task-specific arguments
            
        Returns:
            AgentResponse with execution results
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.execute(**kwargs))
    
    def call_llm_sync(
        self,
        messages: List[AgentMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Synchronous wrapper for call_llm().
        
        Args:
            messages: List of messages to send
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Raw response string from LLM
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.call_llm(messages, temperature, max_tokens)
        )


# Export all public classes and types
__all__ = [
    "AgentRole",
    "AgentState",
    "AgentConfig",
    "AgentMessage",
    "AgentResponse",
    "PromptTemplate",
    "BaseAgent",
    "SyncBaseAgent",
]
