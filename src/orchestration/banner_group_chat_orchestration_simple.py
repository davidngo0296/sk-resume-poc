"""
Banner Group Chat Orchestration using Semantic Kernel's GroupChatOrchestration pattern.
Uses custom GroupChatManager for banner design workflow coordination following Microsoft's official sample.
"""

import sys
from typing import Dict, Any, List, cast
import asyncio

# Core Semantic Kernel imports following official sample
from semantic_kernel.agents import Agent, ChatCompletionAgent, GroupChatOrchestration
from semantic_kernel.agents.orchestration.group_chat import BooleanResult, GroupChatManager, MessageResult, StringResult
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.prompt_template import KernelPromptTemplate, PromptTemplateConfig

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.graphic_designer_agent import GraphicDesignerAgent
from src.agents.graphic_evaluator_agent import GraphicEvaluatorAgent
from src.agents.manager_agent import ManagerAgent

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


class BannerDesignGroupChatManager(GroupChatManager):
    """
    Custom Group Chat Manager for banner design workflow.
    
    This manager controls the flow between designer, evaluator, and manager agents,
    using LLM-based decisions for termination and agent selection.
    """

    # Pydantic fields - define as class attributes
    service: ChatCompletionClientBase
    topic: str

    termination_prompt: str = (
        "You are managing a banner design workflow on the topic: '{{$topic}}'. "
        "You need to determine if the banner design task has been completed successfully. "
        "The workflow should end when the banner design is complete, has been evaluated, and approved by the manager. "
        "If the banner design task is complete and satisfactory, respond with True. Otherwise, respond with False."
    )

    selection_prompt: str = (
        "You are managing a banner design workflow on the topic: '{{$topic}}'. "
        "You need to select the next participant to work on the banner design. "
        "Here are the available participants: "
        "{{$participants}}\n"
        "Select who should work next based on the current state of the design process. "
        "Consider: Designer creates, Evaluator reviews, Manager coordinates and approves. "
        "Please respond with only the name of the participant you would like to select."
    )

    result_filter_prompt: str = (
        "You are managing a banner design workflow on the topic: '{{$topic}}'. "
        "The banner design task has been completed. "
        "Please provide the final banner design description and a summary of the design process."
    )

    async def _render_prompt(self, prompt: str, arguments: KernelArguments) -> str:
        """Helper to render a prompt with arguments."""
        prompt_template_config = PromptTemplateConfig(template=prompt)
        prompt_template = KernelPromptTemplate(prompt_template_config=prompt_template_config)
        return await prompt_template.render(Kernel(), arguments=arguments)

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """Determine if user input is needed - generally not required for automated banner design workflow."""
        return BooleanResult(
            result=False,
            reason="Banner design workflow operates autonomously between agents.",
        )

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """Determine if the banner design workflow should end."""
        should_terminate = await super().should_terminate(chat_history)
        if should_terminate.result:
            return should_terminate

        # Create a copy of chat history for termination check
        termination_history = ChatHistory()
        termination_history.messages = chat_history.messages.copy()
        
        termination_history.messages.insert(
            0,
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.termination_prompt,
                    KernelArguments(topic=self.topic),
                ),
            ),
        )
        termination_history.add_message(
            ChatMessageContent(role=AuthorRole.USER, content="Determine if the banner design task is complete."),
        )

        try:
            response = await self.service.get_chat_message_content(
                termination_history,
                settings=PromptExecutionSettings(response_format=BooleanResult),
            )

            if response and response.content:
                termination_with_reason = BooleanResult.model_validate_json(response.content)

                print("=" * 50)
                print(f"🎨 Banner Design Workflow Status:")
                print(f"Should terminate: {termination_with_reason.result}")
                print(f"Reason: {termination_with_reason.reason}")
                print("=" * 50)

                return termination_with_reason
            else:
                raise ValueError("Empty response from service")
                
        except Exception as e:
            print(f"⚠️ Error in termination check: {e}")
            # Default to continue if there's an error
            return BooleanResult(
                result=False,
                reason=f"Error in termination check: {e}. Continuing workflow."
            )

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringResult:
        """Select the next agent to work on the banner design."""
        # Create a copy of chat history for agent selection
        selection_history = ChatHistory()
        selection_history.messages = chat_history.messages.copy()
        
        selection_history.messages.insert(
            0,
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.selection_prompt,
                    KernelArguments(
                        topic=self.topic,
                        participants="\n".join([f"{k}: {v}" for k, v in participant_descriptions.items()]),
                    ),
                ),
            ),
        )
        selection_history.add_message(
            ChatMessageContent(role=AuthorRole.USER, content="Select the next participant to work on the banner design."),
        )

        try:
            response = await self.service.get_chat_message_content(
                selection_history,
                settings=PromptExecutionSettings(response_format=StringResult),
            )

            if response and response.content:
                participant_name_with_reason = StringResult.model_validate_json(response.content)

                print("=" * 50)
                print(f"🎯 Next Agent Selection:")
                print(f"Selected: {participant_name_with_reason.result}")
                print(f"Reason: {participant_name_with_reason.reason}")
                print("=" * 50)

                if participant_name_with_reason.result in participant_descriptions:
                    return participant_name_with_reason

                raise RuntimeError(f"Unknown participant selected: {participant_name_with_reason.result}")
            else:
                raise ValueError("Empty response from service")
                
        except Exception as e:
            print(f"⚠️ Error in agent selection: {e}")
            # Fallback to first available agent
            first_agent = list(participant_descriptions.keys())[0]
            return StringResult(
                result=first_agent,
                reason=f"Error in selection, defaulting to {first_agent}: {e}"
            )

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> MessageResult:
        """Filter and summarize the banner design results."""
        if not chat_history.messages:
            raise RuntimeError("No messages in the chat history.")

        # Create a copy of chat history for result filtering
        filter_history = ChatHistory()
        filter_history.messages = chat_history.messages.copy()
        
        filter_history.messages.insert(
            0,
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.result_filter_prompt,
                    KernelArguments(topic=self.topic),
                ),
            ),
        )
        filter_history.add_message(
            ChatMessageContent(role=AuthorRole.USER, content="Please provide the final banner design and summary."),
        )

        try:
            response = await self.service.get_chat_message_content(
                filter_history,
                settings=PromptExecutionSettings(response_format=StringResult),
            )
            
            if response and response.content:
                string_with_reason = StringResult.model_validate_json(response.content)

                return MessageResult(
                    result=ChatMessageContent(role=AuthorRole.ASSISTANT, content=string_with_reason.result),
                    reason=string_with_reason.reason,
                )
            else:
                raise ValueError("Empty response from service")
                
        except Exception as e:
            print(f"⚠️ Error in result filtering: {e}")
            # Fallback to simple summary
            return MessageResult(
                result=ChatMessageContent(
                    role=AuthorRole.ASSISTANT, 
                    content="Banner design workflow completed. Please review the conversation above for the final design."
                ),
                reason=f"Error in filtering, using fallback summary: {e}"
            )


def banner_agent_response_callback(messages) -> None:
    """Callback function to handle banner design agent responses."""
    if isinstance(messages, list):
        for message in messages:
            if hasattr(message, 'name') and hasattr(message, 'content'):
                print(f"\n🎨 **{message.name}**")
                print(f"🖼️ {message.content}\n")
                print("-" * 80)
    elif hasattr(messages, 'name') and hasattr(messages, 'content'):
        print(f"\n🎨 **{messages.name}**")
        print(f"🖼️ {messages.content}\n")
        print("-" * 80)


class BannerGroupChatOrchestration:
    """
    Banner design workflow using GroupChatOrchestration with custom manager.
    Updated to follow Microsoft's official sample patterns.
    """
    
    def __init__(self):
        """Initialize the banner workflow with all required agents."""
        
        # Create service for the manager (reuse the same configuration)
        self.service = self._create_chat_service()
        
        # Initialize specialized agents following official pattern
        self.designer_agent = GraphicDesignerAgent()
        self.evaluator_agent = GraphicEvaluatorAgent()
        self.manager_agent = ManagerAgent()
        
        # Get ChatCompletionAgent instances
        designer_agent = self.designer_agent.get_agent()
        evaluator_agent = self.evaluator_agent.get_agent()
        manager_agent = self.manager_agent.get_agent()
        
        # Validate agents
        if not designer_agent or not evaluator_agent or not manager_agent:
            raise RuntimeError("Failed to create required ChatCompletionAgent instances")
        
        # Cast to Agent type as required by GroupChatOrchestration
        self.agents: List[Agent] = cast(List[Agent], [manager_agent, designer_agent, evaluator_agent])
        
        print(f"✅ BannerGroupChatOrchestration initialized with {len(self.agents)} agents")
        agent_names = [getattr(agent, 'name', 'Unknown') for agent in self.agents]
        print(f"📋 Agent order: {agent_names}")
    
    def _create_chat_service(self) -> ChatCompletionClientBase:
        """Create chat completion service for the manager."""
        # Check if we have valid Azure OpenAI configuration
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        
        if (azure_endpoint and azure_api_key and 
            not azure_endpoint.startswith("your_") and 
            not azure_api_key.startswith("your_") and
            azure_endpoint.startswith("https://")):
            
            return AzureChatCompletion(
                deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
                api_key=azure_api_key,
                endpoint=azure_endpoint,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            )
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key or openai_api_key.startswith("your_"):
                raise ValueError("No valid API configuration found for GroupChatManager")
            
            return OpenAIChatCompletion(
                ai_model_id=os.getenv("CHAT_MODEL_ID", "gpt-4o-mini"),
                api_key=openai_api_key,
                org_id=os.getenv("OPENAI_ORG_ID")
            )

    async def start_conversation(self, user_request: str) -> str:
        """
        Start a banner design conversation with the user request.
        Updated to follow official Microsoft sample pattern.
        
        Args:
            user_request: The user's banner design request
            
        Returns:
            The conversation result as a string
        """
        
        try:
            print(f"🎨 Starting banner design workflow...")
            print(f"📝 User request: {user_request}")
            
            # Generate conversation ID
            from src.serialization.conversation_serializer import WorkflowSerializer, WorkflowStatus
            serializer = WorkflowSerializer()
            conversation_id = serializer.generate_conversation_id("banner_design", user_request)
            
            # Create custom group chat manager following official sample
            group_chat_manager = BannerDesignGroupChatManager(
                topic=user_request,
                service=self.service,
                max_rounds=15,  # Allow more rounds for design iteration
            )
            
            # Create the GroupChatOrchestration following official pattern
            group_chat_orchestration = GroupChatOrchestration(
                members=self.agents,
                manager=group_chat_manager,
                agent_response_callback=banner_agent_response_callback,
            )
            
            # Create and start the runtime (following official sample)
            runtime = InProcessRuntime()
            runtime.start()
            
            try:
                # Invoke the orchestration with timeout (following official pattern)
                print("🚀 Starting banner design orchestration...")
                orchestration_result = await asyncio.wait_for(
                    group_chat_orchestration.invoke(
                        task=f"Please start working on this banner design request: {user_request}",
                        runtime=runtime,
                    ),
                    timeout=180.0  # 3 minute timeout for full design conversation
                )
                
                # Get the final result (following official pattern)
                print("⏳ Waiting for final design results...")
                final_output = await asyncio.wait_for(
                    orchestration_result.get(),
                    timeout=30.0  # 30 second timeout for result retrieval
                )
                
                print(f"✅ Banner design workflow completed successfully")
                
                # Create minimal chat history for serialization
                chat_history = ChatHistory()
                chat_history.add_message(ChatMessageContent(
                    role=AuthorRole.USER,
                    content=user_request
                ))
                
                # Handle different result types - final_output could be a single message or list
                if isinstance(final_output, list):
                    # Extract content from list of messages
                    if final_output and hasattr(final_output[0], 'content'):
                        result_content = str(final_output[0].content)
                    else:
                        result_content = str(final_output)
                elif hasattr(final_output, 'content'):
                    result_content = str(final_output.content)
                else:
                    result_content = str(final_output)
                
                # Add result to chat history
                chat_history.add_message(ChatMessageContent(
                    role=AuthorRole.ASSISTANT,
                    content=result_content
                ))
                
                # Create comprehensive conversation snapshot
                snapshot = serializer.create_conversation_snapshot(
                    conversation_id=conversation_id,
                    workflow_name="Banner Design Workflow",
                    workflow_type="banner_design",
                    user_request=user_request,
                    chat_history=chat_history,
                    group_chat_manager=group_chat_manager,
                    agents=self.agents,
                    checkpoint_reason="Workflow completed successfully",
                    status=WorkflowStatus.COMPLETED,
                    current_agent=None,
                    resume_instructions=None
                )
                
                # Save the snapshot
                serializer.save_conversation_snapshot(snapshot)
                
                # Create comprehensive result file
                result_file = serializer.create_result_file(snapshot, result_content)
                print(f"📄 Result file created: {result_file}")
                
                return result_content
                
            except asyncio.TimeoutError:
                # Create snapshot for timeout case
                chat_history = ChatHistory()
                chat_history.add_message(ChatMessageContent(
                    role=AuthorRole.USER,
                    content=user_request
                ))
                
                snapshot = serializer.create_conversation_snapshot(
                    conversation_id=conversation_id,
                    workflow_name="Banner Design Workflow",
                    workflow_type="banner_design",
                    user_request=user_request,
                    chat_history=chat_history,
                    group_chat_manager=group_chat_manager,
                    agents=self.agents,
                    checkpoint_reason="Workflow timed out",
                    status=WorkflowStatus.FAILED,
                    current_agent=None,
                    resume_instructions="Workflow timed out - check API configuration"
                )
                
                # Save the snapshot
                serializer.save_conversation_snapshot(snapshot)
                
                error_msg = (
                    "❌ Banner design workflow timed out. This usually means:\n"
                    "1. API calls are failing due to invalid API keys\n"
                    "2. Network connectivity issues\n"
                    "3. The design conversation is taking too long to reach a conclusion\n\n"
                    "Please check your API configuration in api_keys_config.env"
                )
                print(error_msg)
                
                # Create result file even for timeout
                result_file = serializer.create_result_file(snapshot, error_msg)
                print(f"📄 Result file created: {result_file}")
                
                return error_msg
                
            finally:
                # Always stop the runtime when done (following official pattern)
                try:
                    await asyncio.wait_for(
                        runtime.stop_when_idle(), 
                        timeout=10.0
                    )
                    print("🛑 Runtime stopped successfully")
                except asyncio.TimeoutError:
                    print("⚠️ Warning: Runtime cleanup timed out")
                except Exception as cleanup_e:
                    print(f"⚠️ Warning: Runtime cleanup error: {cleanup_e}")
            
        except Exception as e:
            error_msg = f"❌ Error in banner design workflow: {e}"
            print(error_msg)
            return error_msg
    
    def get_agents_info(self) -> Dict[str, Any]:
        """Return information about the workflow agents."""
        agent_names = []
        for agent in self.agents:
            if agent is not None:
                agent_names.append(getattr(agent, 'name', 'Unknown'))
        
        return {
            "total_agents": len(self.agents),
            "agent_names": agent_names,
            "workflow_type": "banner_design",
            "orchestration_type": "GroupChatOrchestration",
            "manager_type": "BannerDesignGroupChatManager",
            "follows_official_pattern": True
        } 