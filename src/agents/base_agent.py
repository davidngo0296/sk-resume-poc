"""
Base agent class for the video script generation multi-agent system.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from semantic_kernel import Kernel
    from semantic_kernel.agents import ChatCompletionAgent


class BaseVideoAgent(ABC):
    """
    Base class for all video script generation agents.
    Provides common initialization and utility methods.
    """
    
    def __init__(self, name: str, description: str, instructions: str):
        self.name = name
        self.description = description
        self.instructions = instructions
        self.kernel = self._create_kernel()
        self.agent = None  # Will be set by child classes if needed
    
    def _create_kernel(self) -> "Kernel":
        """Create and configure the Semantic Kernel instance."""
        try:
            from semantic_kernel import Kernel
            from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
        except ImportError as e:
            print(f"Error importing Semantic Kernel: {e}")
            print("Please ensure you have the latest version of semantic-kernel installed:")
            print("pip install semantic-kernel --upgrade")
            raise
        
        kernel = Kernel()
        
        # Check if we have valid Azure OpenAI configuration (not placeholder values)
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        
        # Use Azure OpenAI only if we have valid (non-placeholder) configuration
        if (azure_endpoint and azure_api_key and 
            not azure_endpoint.startswith("your_") and 
            not azure_api_key.startswith("your_") and
            azure_endpoint.startswith("https://")):
            
            print("Using Azure OpenAI configuration")
            chat_completion = AzureChatCompletion(
                deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
                api_key=azure_api_key,
                endpoint=azure_endpoint,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            )
        else:
            # Use OpenAI
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key or openai_api_key.startswith("your_"):
                raise ValueError(
                    "No valid API configuration found. Please set either:\n"
                    "1. Valid Azure OpenAI configuration (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY)\n"
                    "2. Valid OpenAI configuration (OPENAI_API_KEY)\n"
                    "Check your api_keys_config.env file."
                )
            
            print("Using OpenAI configuration")
            chat_completion = OpenAIChatCompletion(
                ai_model_id=os.getenv("CHAT_MODEL_ID", "gpt-4o-mini"),
                api_key=openai_api_key,
                org_id=os.getenv("OPENAI_ORG_ID")
            )
        
        kernel.add_service(chat_completion)
        return kernel
    
    def _create_agent(self) -> Optional["ChatCompletionAgent"]:
        """Create the chat completion agent with specific instructions."""
        try:
            from semantic_kernel.agents import ChatCompletionAgent
            
            return ChatCompletionAgent(
                service_id="default",
                kernel=self.kernel,
                name=self.name,
                instructions=self.instructions,
                description=self.description
            )
        except ImportError:
            # If ChatCompletionAgent is not available in this version, return None
            print(f"Warning: ChatCompletionAgent not available, using direct kernel service for {self.name}")
            return None
    
    @abstractmethod
    def get_specialized_instructions(self) -> str:
        """Return agent-specific instructions for their role."""
        pass
    
    def get_agent(self) -> Optional["ChatCompletionAgent"]:
        """Return the configured agent instance if available."""
        if self.agent is None:
            self.agent = self._create_agent()
        return self.agent
    
    def get_kernel(self) -> "Kernel":
        """Return the configured kernel instance."""
        return self.kernel 