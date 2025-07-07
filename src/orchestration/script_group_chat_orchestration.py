"""
Script Group Chat Orchestration using Semantic Kernel's GroupChatOrchestration pattern.
Uses custom GroupChatManager for script writing workflow coordination following Microsoft's official sample.
"""

import sys
from typing import Dict, Any, List, cast, ClassVar
import asyncio

# Core Semantic Kernel imports following official sample
from semantic_kernel.agents import Agent, ChatCompletionAgent, GroupChatOrchestration
from semantic_kernel.agents.orchestration.group_chat import BooleanResult, GroupChatManager, MessageResult, StringResult
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.script_writer_agent import ScriptWriterAgent
from src.agents.manager_agent import ManagerAgent
from src.agents.audio_selector_agent import AudioSelectorAgent

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


async def human_response_function(chat_history: ChatHistory) -> ChatMessageContent:
    """Function to get user input for script approval following official MS pattern."""
    print("\n" + "="*60)
    print("🎯 USER INPUT REQUIRED")
    print("📝 Please review the script above and provide your feedback:")
    print("💡 Type 'approve' to accept the script")
    print("💡 Type 'reject' to request a rewrite")
    print("💡 Type 'pause' to save and resume later")
    print("💡 Or provide specific feedback for improvements")
    print("="*60)
    
    user_input = input("Your response: ").strip()
    
    print(f"✅ User response received: {user_input}")
    print("-" * 60)
    
    # Store user input in conversation history
    global _conversation_history
    _conversation_history.append({
        'name': 'User',
        'role': 'user',
        'content': user_input
    })
    
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)


# Global flag to track pause requests
_pause_requested = False

def set_pause_requested():
    """Set the global pause flag."""
    global _pause_requested
    _pause_requested = True

def is_pause_requested():
    """Check if pause was requested."""
    global _pause_requested
    return _pause_requested

def clear_pause_flag():
    """Clear the pause flag."""
    global _pause_requested
    _pause_requested = False


class WorkflowPauseException(Exception):
    """Exception raised when user requests to pause the workflow."""
    pass


class ScriptWritingGroupChatManager(GroupChatManager):
    """
    Script Writing Group Chat Manager using prompt engineering approach.
    
    Follows Microsoft's pattern with LLM-driven decision making via prompts.
    """
    
    # Pydantic model fields - required for proper validation
    topic: str
    service: ChatCompletionClientBase
    
    # Prompt templates for LLM decision making (using ClassVar to avoid Pydantic field validation)
    termination_prompt: ClassVar[str] = (
        "You are managing a script writing workflow for the topic: '{{$topic}}'. "
        "The workflow completes when the full sequence is done: ScriptWriter creates script → User approves → AudioSelector provides audio → COMPLETE\n"
        "\n"
        "TERMINATION CRITERIA (respond True if ANY of these are met):\n"
        "1) Workflow status contains 'READY TO TERMINATE'\n"
        "2) Workflow status contains 'WORKFLOW COMPLETE - AudioSelector provided audio recommendations'\n"
        "3) Workflow status contains 'Audio recommendations provided'\n"
        "4) Conversation has 25+ messages (safety limit)\n"
        "\n"
        "DO NOT TERMINATE if:\n"
        "- Status contains 'WORKFLOW NEEDS AUDIO' or 'WORKFLOW WAITING'\n"
        "- Status contains 'WORKFLOW STARTING' or 'NEEDS REWRITE'\n"
        "- Only ScriptWriter has provided a script without AudioSelector\n"
        "\n"
        "Current workflow status: {{$workflow_status}}\n"
        "Recent messages: {{$recent_messages}}\n"
        "\n"
        "If workflow status shows READY TO TERMINATE or WORKFLOW COMPLETE, respond True to end the workflow."
    )
    
    agent_selection_prompt: ClassVar[str] = (
        "You are managing a script writing workflow for topic: '{{$topic}}'. "
        "Workflow steps: ScriptWriter creates script → User approves → AudioSelector suggests audio → Complete\n"
        "\n"
        "Participants: {{$participants}}\n"
        "Recent conversation: {{$recent_messages}}\n"
        "Workflow status: {{$workflow_status}}\n"
        "\n"
        "Who should speak next? Rules (in priority order):\n"
        "1. If workflow status contains 'User APPROVED script' AND ('NEEDS AUDIO' OR 'Script exists'): AudioSelector\n"
        "2. If workflow status contains 'User REJECTED script' or 'NEEDS REWRITE': ScriptWriter\n"
        "3. If workflow status contains 'STARTING' AND 'no script exists': ScriptWriter\n"
        "4. If workflow status contains 'WAITING' and script exists: wait for user input\n"
        "5. Manager coordinates when needed\n"
        "\n"
        "CRITICAL: For resume scenarios, if user approved a script (even from resume context), go to AudioSelector.\n"
        "IMPORTANT: Respond with ONLY the agent name (ScriptWriter, AudioSelector, or Manager). Do not include explanations or reasoning."
    )
    
    user_input_prompt: ClassVar[str] = (
        "You are managing a script writing workflow. "
        "Check if the ScriptWriter just created a complete video script that needs user approval.\n"
        "\n"
        "Recent messages: {{$recent_messages}}\n"
        "\n"
        "Look for these indicators that a COMPLETE script was just created by ScriptWriter:\n"
        "- Contains 'Title:', 'Scene', 'Duration:', 'Narrator:' or 'Narration:'\n"
        "- Has multiple scenes with visual descriptions\n"
        "- Contains actual dialogue or narrator text\n"
        "- Is substantial content (1000+ characters)\n"
        "- Was just posted by ScriptWriter agent\n"
        "\n"
        "IMPORTANT: Only return True if ScriptWriter JUST created a complete script in the most recent message.\n"
        "Return False if: no script exists, script is incomplete, or AudioSelector already responded.\n"
        "\n"
        "Should we request user input to approve/reject this script?"
    )
    
    result_filter_prompt: ClassVar[str] = (
        "You are concluding a script writing workflow for '{{$topic}}'. "
        "Extract and format the final approved script and audio recommendations.\n"
        "\n"
        "Conversation history: {{$conversation}}\n"
        "\n"
        "Provide a professional summary with the final script and audio suggestions."
    )
    
    async def _render_prompt(self, prompt: str, arguments) -> str:
        """Helper to render a prompt with arguments."""
        from semantic_kernel.kernel import Kernel
        from semantic_kernel.prompt_template import KernelPromptTemplate, PromptTemplateConfig
        from semantic_kernel.functions import KernelArguments
        
        prompt_template_config = PromptTemplateConfig(template=prompt)
        prompt_template = KernelPromptTemplate(prompt_template_config=prompt_template_config)
        return await prompt_template.render(Kernel(), arguments=arguments)
    
    def _get_recent_messages_summary(self, chat_history: ChatHistory, max_messages: int = 5) -> str:
        """Get a summary of recent messages for prompt context."""
        recent_messages = chat_history.messages[-max_messages:] if len(chat_history.messages) > max_messages else chat_history.messages
        
        summary = []
        for msg in recent_messages:
            sender = getattr(msg, 'name', str(msg.role))
            content = str(msg.content)
            # Truncate long messages for prompt efficiency
            if len(content) > 200:
                content = content[:200] + "..."
            summary.append(f"{sender}: {content}")
        
        return "\n".join(summary) if summary else "No messages yet"
    
    def _analyze_workflow_status(self, chat_history: ChatHistory) -> str:
        """Analyze current workflow status for prompt context."""
        status_info = []
        
        # Check for user responses (with special handling for resume context)
        user_messages = [msg for msg in chat_history.messages if hasattr(msg, 'role') and msg.role == AuthorRole.USER]
        user_approval_status = "No user input"
        
        # DEBUG: Print all user messages to see what's happening
        print(f"🔍 DEBUG: All user messages in history:")
        for i, msg in enumerate(user_messages):
            print(f"  {i}: '{str(msg.content).strip()}'")
        
        if user_messages:
            last_user = str(user_messages[-1].content).lower().strip()
            print(f"🔍 DEBUG: Last user message: '{last_user}'")
            
            # PRIORITY CHECK: Look for direct approve/reject messages first
            # Check all recent user messages for direct approve/reject (not just the last one)
            direct_approval_found = False
            for msg in reversed(user_messages[-3:]):  # Check last 3 user messages
                content = str(msg.content).lower().strip()
                if content == 'approve':
                    user_approval_status = "User APPROVED script"
                    status_info.append("User APPROVED script")
                    print(f"🔍 DEBUG: Direct user approval detected: '{content}'")
                    direct_approval_found = True
                    break
                elif content == 'reject':
                    user_approval_status = "User REJECTED script"
                    status_info.append("User REJECTED script - needs rewrite")
                    print(f"🔍 DEBUG: Direct user rejection detected: '{content}'")
                    direct_approval_found = True
                    break
            
            # Only process other message types if no direct approve/reject was found
            if not direct_approval_found:
                # Special handling for resume context messages
                if 'resuming script writing workflow' in last_user:
                    # Extract the actual user action from resume context
                    if 'user rejected the script: "reject"' in last_user:
                        user_approval_status = "User REJECTED script"
                        status_info.append("User REJECTED script - needs rewrite")
                        print(f"🔍 DEBUG: User rejection detected in resume context!")
                    elif 'user approved the script: "approve"' in last_user:
                        user_approval_status = "User APPROVED script"
                        status_info.append("User APPROVED script")
                        print(f"🔍 DEBUG: User approval detected in resume context!")
                    elif 'user provided feedback:' in last_user:
                        user_approval_status = "User provided feedback"
                        status_info.append("User provided feedback - needs revision")
                        print(f"🔍 DEBUG: User provided feedback detected in resume context!")
                    else:
                        user_approval_status = "User provided other input"
                        status_info.append("User provided other input")
                        print(f"🔍 DEBUG: Other user input detected in resume context!")
                # Standard user input detection for direct messages
                elif last_user:
                    user_approval_status = "User provided feedback"
                    status_info.append("User provided feedback - needs revision")
                    print(f"🔍 DEBUG: User provided other feedback!")
        
        # Check for script creation
        script_exists = False
        script_messages = [msg for msg in chat_history.messages if hasattr(msg, 'name') and msg.name == "ScriptWriter"]
        if script_messages:
            for msg in script_messages:
                content = str(msg.content).lower()
                if any(indicator in content for indicator in ["title:", "scene", "narrator", "duration:"]):
                    if len(content) > 500:
                        script_exists = True
                        status_info.append("Complete script created by ScriptWriter")
                        break
        
        # SPECIAL CASE: If we're in resume context and user approved/rejected a script,
        # we can infer that a script exists even if we don't see it in current chat history
        if not script_exists and user_approval_status in ["User APPROVED script", "User REJECTED script"]:
            last_user_message = ""
            for msg in reversed(chat_history.messages):
                if msg.role == AuthorRole.USER:
                    last_user_message = str(msg.content).lower()
                    break
            
            if 'resuming script writing workflow' in last_user_message and 'script preview:' in last_user_message:
                script_exists = True
                status_info.append("Script exists from resumed conversation context")
                print(f"🔍 DEBUG: Inferred script exists from resume context with user approval/rejection")
        
        # Check for audio suggestions
        audio_provided = False
        audio_messages = [msg for msg in chat_history.messages if hasattr(msg, 'name') and msg.name == "AudioSelector"]
        if audio_messages:
            for msg in audio_messages:
                content = str(msg.content).lower()
                if any(word in content for word in ["audio", "music", "sound", "track"]):
                    audio_provided = True
                    status_info.append("Audio recommendations provided")
                    break
        
        # Add summary of workflow completion status
        if audio_provided:
            status_info.append("WORKFLOW COMPLETE - AudioSelector provided audio recommendations")
            status_info.append("READY TO TERMINATE - Full workflow sequence completed")
        elif script_exists and user_approval_status == "User APPROVED script" and not audio_provided:
            status_info.append("WORKFLOW NEEDS AUDIO - User approved script, need audio recommendations")
        elif script_exists and user_approval_status == "User REJECTED script":
            status_info.append("WORKFLOW NEEDS REWRITE - User rejected script")
        elif script_exists and user_approval_status == "No user input":
            status_info.append("WORKFLOW WAITING - Script created, awaiting user approval")
        elif not script_exists:
            status_info.append("WORKFLOW STARTING - No script exists yet")
        
        return "; ".join(status_info) if status_info else "Workflow starting"

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """Use LLM to determine if user input is needed for script approval."""
        
        print("🔍 should_request_user_input() called!")
        print(f"🔍 Messages in history: {len(chat_history.messages)}")
        
        if len(chat_history.messages) == 0:
            return BooleanResult(result=False, reason="No messages yet.")
        
        # Use LLM to determine if user input is needed
        from semantic_kernel.functions import KernelArguments
        from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
        
        recent_messages = self._get_recent_messages_summary(chat_history)
        
        # Create a temporary chat history for the LLM decision
        temp_history = ChatHistory()
        temp_history.add_message(ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content=await self._render_prompt(
                self.user_input_prompt,
                KernelArguments(recent_messages=recent_messages)
            )
        ))
        temp_history.add_message(ChatMessageContent(
            role=AuthorRole.USER,
            content="Should we request user input now?"
        ))
        
        try:
            # SAFETY CHECK: Direct detection of complete scripts for user approval
            last_message = chat_history.messages[-1] if chat_history.messages else None
            if (last_message and 
                hasattr(last_message, 'name') and 
                last_message.name == "ScriptWriter"):
                
                content = str(last_message.content)
                script_indicators = ["title:", "scene", "duration:", "narrator", "narration"]
                
                # Check if this looks like a complete script
                has_indicators = sum(1 for indicator in script_indicators if indicator.lower() in content.lower())
                is_substantial = len(content) > 1000
                
                print(f"🔍 Script detection: {has_indicators} indicators, {len(content)} chars")
                
                if has_indicators >= 3 and is_substantial:
                    print("\n" + "="*80)
                    print("📝 ✅ COMPLETE SCRIPT DETECTED BY SAFETY CHECK!")
                    print("🎯 WORKFLOW PAUSED: Requesting your approval...")
                    print("💡 The script is ready for your review!")
                    print("="*80)
                    
                    # Get user input
                    user_response = await human_response_function(chat_history)
                    chat_history.add_message(user_response)
                    
                    # Check if user requested pause
                    if user_response.content.lower().strip() == 'pause':
                        print("⏸️ Workflow pause requested - saving current state...")
                        set_pause_requested()
                        return BooleanResult(
                            result=False,
                            reason="User requested workflow pause."
                        )
                    
                    print(f"✅ User response added to chat: {user_response.content}")
                    
                    return BooleanResult(
                        result=False,  # Return False since we've already handled the user input
                        reason="User input collected via safety check."
                    )
            
            # Fall back to LLM decision
            response = await self.service.get_chat_message_content(
                temp_history,
                settings=PromptExecutionSettings(response_format=BooleanResult),
            )
            
            if not response or not response.content:
                return BooleanResult(result=False, reason="No response from LLM.")
            
            decision = BooleanResult.model_validate_json(response.content)
            
            print(f"🤖 LLM decision: {decision.result} - {decision.reason}")
            
            # If LLM says we need user input, get it
            if decision.result:
                print("\n" + "="*80)
                print("📝 ✅ COMPLETE SCRIPT DETECTED!")
                print("🎯 WORKFLOW PAUSED: Requesting your approval...")
                print("💡 The script is ready for your review!")
                print("="*80)
                
                # Get user input
                user_response = await human_response_function(chat_history)
                chat_history.add_message(user_response)
                
                # Check if user requested pause
                if user_response.content.lower().strip() == 'pause':
                    print("⏸️ Workflow pause requested - saving current state...")
                    set_pause_requested()
                    return BooleanResult(
                        result=False,
                        reason="User requested workflow pause."
                    )
                
                print(f"✅ User response added to chat: {user_response.content}")
                
                return BooleanResult(
                    result=False,  # Return False since we've already handled the user input
                    reason="User input collected and added to chat history."
                )
            
            return decision
            
        except Exception as e:
            print(f"⚠️ Error in LLM decision making: {e}")
            # Fallback to simple logic if LLM fails
            return BooleanResult(result=False, reason="LLM decision failed, continuing workflow.")

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """Use LLM to determine if workflow should terminate."""
        
        # Check if pause was requested - terminate immediately if so
        if is_pause_requested():
            print("🔍 DEBUG: Pause detected - terminating workflow for pause handling")
            return BooleanResult(
                result=True,
                reason="User requested workflow pause - terminating for pause handling."
            )
        
        # Safety check - terminate if too many messages to prevent infinite loops
        if len(chat_history.messages) > 25:
            return BooleanResult(
                result=True,
                reason="Maximum conversation length reached - terminating for safety."
            )
        
        # Use LLM to determine termination
        from semantic_kernel.functions import KernelArguments
        from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
        
        workflow_status = self._analyze_workflow_status(chat_history)
        recent_messages = self._get_recent_messages_summary(chat_history, max_messages=10)
        
        # Create a temporary chat history for the LLM decision
        temp_history = ChatHistory()
        temp_history.add_message(ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content=await self._render_prompt(
                self.termination_prompt,
                KernelArguments(
                    topic=self.topic,
                    workflow_status=workflow_status,
                    recent_messages=recent_messages
                )
            )
        ))
        temp_history.add_message(ChatMessageContent(
            role=AuthorRole.USER,
            content="Should the workflow terminate now?"
        ))
        
        try:
            response = await self.service.get_chat_message_content(
                temp_history,
                settings=PromptExecutionSettings(response_format=BooleanResult),
            )
            
            if not response or not response.content:
                return BooleanResult(result=False, reason="No response from LLM.")
            
            decision = BooleanResult.model_validate_json(response.content)
            
            print(f"🤖 Termination decision: {decision.result} - {decision.reason}")
            
            # Debug: Show what workflow status was analyzed
            current_status = self._analyze_workflow_status(chat_history)
            print(f"🔍 Current workflow status: {current_status}")
            
            # Override termination decision based on clear workflow status
            if "WORKFLOW COMPLETE" in current_status:
                print("🔍 DEBUG: Overriding termination decision - workflow is complete!")
                return BooleanResult(result=True, reason="Workflow complete - user approved script and audio provided")
            elif "WORKFLOW NEEDS REWRITE" in current_status or "WORKFLOW NEEDS AUDIO" in current_status:
                print("🔍 DEBUG: Overriding termination decision - workflow needs to continue!")
                return BooleanResult(result=False, reason="Workflow needs to continue - waiting for completion")
            
            return decision
            
        except Exception as e:
            print(f"⚠️ Error in LLM termination decision: {e}")
            # Fallback to simple logic if LLM fails
            return BooleanResult(result=False, reason="LLM decision failed, continuing workflow.")

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringResult:
        """Use LLM to select the next agent to speak."""
        
        # Check if pause was requested - return special result to trigger termination
        if is_pause_requested():
            print("🔍 DEBUG: Pause detected in agent selection - triggering workflow termination")
            return StringResult(
                result="PAUSE_REQUESTED",
                reason="User requested workflow pause during agent selection."
            )
        
        print(f"🔍 Selecting next agent from: {list(participant_descriptions.keys())}")
        
        # Use LLM to select next agent
        from semantic_kernel.functions import KernelArguments
        from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
        
        workflow_status = self._analyze_workflow_status(chat_history)
        recent_messages = self._get_recent_messages_summary(chat_history)
        participants_info = "\n".join([f"{k}: {v}" for k, v in participant_descriptions.items()])
        
        print(f"🔍 DEBUG: Workflow status for agent selection: {workflow_status}")
        
        # Create a temporary chat history for the LLM decision
        temp_history = ChatHistory()
        temp_history.add_message(ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content=await self._render_prompt(
                self.agent_selection_prompt,
                KernelArguments(
                    topic=self.topic,
                    participants=participants_info,
                    recent_messages=recent_messages,
                    workflow_status=workflow_status
                )
            )
        ))
        temp_history.add_message(ChatMessageContent(
            role=AuthorRole.USER,
            content="Who should speak next?"
        ))
        
        try:
            response = await self.service.get_chat_message_content(
                temp_history,
                settings=PromptExecutionSettings(response_format=StringResult),
            )
            
            if not response or not response.content:
                return StringResult(result="ScriptWriter", reason="No response from LLM, defaulting to ScriptWriter.")
            
            decision = StringResult.model_validate_json(response.content)
            
            print(f"🤖 Agent selection: {decision.result} - {decision.reason}")
            
            # Extract agent name from decision result - handle cases where LLM returns explanatory text
            selected_agent = decision.result.strip()
            
            # Check if the result contains a valid agent name
            valid_agent = None
            for agent_name in participant_descriptions.keys():
                if agent_name.lower() in selected_agent.lower():
                    valid_agent = agent_name
                    break
            
            if valid_agent:
                print(f"🔍 DEBUG: Selected agent from text: {valid_agent}")
                return StringResult(
                    result=valid_agent,
                    reason=f"Selected {valid_agent} based on workflow analysis."
                )
            else:
                print(f"🔍 DEBUG: Could not extract valid agent from '{selected_agent}', defaulting to ScriptWriter")
                return StringResult(
                    result="ScriptWriter",
                    reason=f"Invalid agent '{selected_agent}' selected by LLM, defaulting to ScriptWriter."
                )
            
        except Exception as e:
            print(f"⚠️ Error in LLM agent selection: {e}")
            # Fallback to simple logic if LLM fails
            return StringResult(result="ScriptWriter", reason="LLM decision failed, defaulting to ScriptWriter.")

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> MessageResult:
        """Use LLM to filter and summarize the final results."""
        
        # CRITICAL: Check for pause before creating any results
        # Check both the pause flag AND the conversation history for pause requests
        last_user_message = None
        for msg in reversed(chat_history.messages):
            if msg.role == AuthorRole.USER and msg.content.strip().lower() == 'pause':
                last_user_message = msg.content.strip().lower()
                break
        
        if is_pause_requested() or last_user_message == 'pause':
            print("🔍 DEBUG: Pause detected in filter_results - returning pause message instead of completion")
            print(f"🔍 DEBUG: Pause flag: {is_pause_requested()}, Last user message: {last_user_message}")
            return MessageResult(
                result=ChatMessageContent(role=AuthorRole.ASSISTANT, content="Workflow paused by user request"),
                reason="User requested workflow pause during result filtering."
            )
        
        if not chat_history.messages:
            raise RuntimeError("No messages in the chat history.")

        # Use LLM to create the final summary
        from semantic_kernel.functions import KernelArguments
        from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
        
        conversation_summary = self._get_recent_messages_summary(chat_history, max_messages=20)
        
        # Create a temporary chat history for the LLM decision
        temp_history = ChatHistory()
        temp_history.add_message(ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content=await self._render_prompt(
                self.result_filter_prompt,
                KernelArguments(
                    topic=self.topic,
                    conversation=conversation_summary
                )
            )
        ))
        temp_history.add_message(ChatMessageContent(
            role=AuthorRole.USER,
            content="Please provide the final summary."
        ))
        
        try:
            response = await self.service.get_chat_message_content(
                temp_history,
                settings=PromptExecutionSettings(response_format=StringResult),
            )
            
            if not response or not response.content:
                return MessageResult(
                    result=ChatMessageContent(role=AuthorRole.ASSISTANT, content="Workflow completed."),
                    reason="No response from LLM for final summary."
                )
            
            summary_result = StringResult.model_validate_json(response.content)
            
            return MessageResult(
                result=ChatMessageContent(role=AuthorRole.ASSISTANT, content=summary_result.result),
                reason=summary_result.reason
            )
            
        except Exception as e:
            print(f"⚠️ Error in LLM result filtering: {e}")
            # Fallback to simple summary
            return MessageResult(
                result=ChatMessageContent(role=AuthorRole.ASSISTANT, content="Script writing workflow completed successfully."),
                reason="LLM result filtering failed, using simple summary."
            )


# Global conversation history storage
_conversation_history: List[Dict[str, str]] = []


def agent_response_callback(messages) -> None:
    """Callback function to handle agent responses with workflow context."""
    global _conversation_history
    
    if isinstance(messages, list):
        for message in messages:
            if hasattr(message, 'name') and hasattr(message, 'content'):
                agent_emoji = "🎬" if message.name == "Manager" else "✍️" if message.name == "ScriptWriter" else "🎵"
                print(f"\n{agent_emoji} **{message.name}**")
                print(f"💬 {message.content}\n")
                print("-" * 80)
                
                # Store in conversation history for result file
                _conversation_history.append({
                    'name': message.name,
                    'role': 'assistant',
                    'content': str(message.content)
                })
                
    elif hasattr(messages, 'name') and hasattr(messages, 'content'):
        agent_emoji = "🎬" if messages.name == "Manager" else "✍️" if messages.name == "ScriptWriter" else "🎵"
        print(f"\n{agent_emoji} **{messages.name}**")
        print(f"💬 {messages.content}\n")
        print("-" * 80)
        
        # Store in conversation history for result file
        _conversation_history.append({
            'name': messages.name,
            'role': 'assistant',
            'content': str(messages.content)
        })


def clear_conversation_history():
    """Clear the conversation history for a new workflow."""
    global _conversation_history
    _conversation_history = []


def get_full_conversation_history():
    """Get the complete conversation history captured by the callback."""
    global _conversation_history
    return _conversation_history.copy()


class ScriptGroupChatOrchestration:
    """
    Script writing workflow using GroupChatOrchestration with custom manager.
    Updated to follow Microsoft's official sample patterns.
    """
    
    def __init__(self):
        """Initialize the script workflow with all required agents."""
        
        # Create service for the manager (reuse the same configuration)
        self.service = self._create_chat_service()
        
        # Initialize specialized agents following official pattern
        self.script_writer_agent = ScriptWriterAgent()
        self.manager_agent = ManagerAgent()
        self.audio_selector_agent = AudioSelectorAgent()
        
        # Get ChatCompletionAgent instances
        script_agent = self.script_writer_agent.get_agent()
        manager_agent = self.manager_agent.get_agent()
        audio_agent = self.audio_selector_agent.get_agent()
        
        # Validate agents
        if not script_agent or not manager_agent or not audio_agent:
            raise RuntimeError("Failed to create required ChatCompletionAgent instances")
        
        # Cast to Agent type as required by GroupChatOrchestration
        # Manager coordinates, ScriptWriter creates, AudioSelector suggests audio
        self.agents: List[Agent] = cast(List[Agent], [manager_agent, script_agent, audio_agent])
        
        print(f"✅ ScriptGroupChatOrchestration initialized with {len(self.agents)} agents")
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

    def _customize_agents_for_workflow(self):
        """Customize agent instructions for focused workflow execution."""
        for agent in self.agents:
            if hasattr(agent, 'name'):
                if agent.name == "Manager":
                    # Make Manager more focused on workflow coordination
                    if hasattr(agent, 'instructions') and agent.instructions:
                        agent.instructions += """

IMPORTANT WORKFLOW INSTRUCTIONS:
- Be concise and focused on workflow coordination only
- Provide clear direction to ScriptWriter to create complete scripts
- Do NOT generate "approve" or "reject" responses - only users do that
- Avoid repetitive or unnecessary messages
- Do not speak multiple times in a row
"""
                elif agent.name == "ScriptWriter":
                    if hasattr(agent, 'instructions') and agent.instructions:
                        agent.instructions += """

CRITICAL WORKFLOW INSTRUCTIONS:
- When given ANY topic, create a COMPLETE video script immediately
- Do NOT ask for clarification - use your best judgment to create a full script
- Always include: Title, Duration, Scene descriptions, Narrator text, Visual descriptions
- Include keywords like 'Scene', 'Narrator', 'Video Script', 'Title:' to help workflow detection
- Be creative and comprehensive - write the full script based on the topic given

REWRITE INSTRUCTIONS (when user rejects):
- If user rejects your script, create a COMPLETELY NEW and DIFFERENT script
- Change the approach, style, tone, and content completely
- Don't just modify the previous script - start fresh with new ideas
- Use different scenes, different narrative structure, different angle
- DO NOT make conversational comments or ask questions - just create the new script immediately

NEVER DO:
- Don't say "Please let me know if you need changes"
- Don't say "Thank you for the feedback"
- Don't ask questions about what the user wants
- Don't make small talk or conversational comments
- Just create complete scripts when requested

ALWAYS DO:
- Create full, complete scripts when given a topic
- Start immediately with "Title:" or script content
- Include proper formatting with Scene/Narrator/Visual elements
"""
                elif agent.name == "AudioSelector":
                    if hasattr(agent, 'instructions') and agent.instructions:
                        agent.instructions += """

WORKFLOW INSTRUCTIONS:
- Provide specific audio recommendations including background music, sound effects, and voiceover guidance
- Be detailed but concise
- Include keywords like 'audio', 'music', 'sound', 'track' for workflow detection
- Give specific recommendations with genres, mood, and timing
"""

    async def start_conversation(self, user_request: str) -> str:
        """
        Start a script writing conversation with the user request.
        Updated to follow official Microsoft sample pattern.
        
        Args:
            user_request: The user's script writing request
            
        Returns:
            The conversation result as a string
        """
        
        try:
            print(f"📝 Starting script writing workflow...")
            print(f"📋 User request: {user_request}")
            
            # Generate conversation ID
            from src.serialization.conversation_serializer import WorkflowSerializer, WorkflowStatus
            serializer = WorkflowSerializer()
            conversation_id = serializer.generate_conversation_id("script_writing", user_request)
            
            # Clear conversation history for new workflow
            clear_conversation_history()
            
            # Customize agents for focused workflow execution
            self._customize_agents_for_workflow()
            
            # Create custom group chat manager with defined workflow and human-in-the-loop
            group_chat_manager = ScriptWritingGroupChatManager(
                topic=user_request,
                service=self.service,
                max_rounds=15,  # Allow more rounds for user interaction and rewrites
            )
            
            # Create the GroupChatOrchestration following official pattern
            group_chat_orchestration = GroupChatOrchestration(
                members=self.agents,
                manager=group_chat_manager,
                agent_response_callback=agent_response_callback,
            )
            
            # Create and start the runtime (following official sample)
            runtime = InProcessRuntime()
            runtime.start()
            
            try:
                # Invoke the orchestration with timeout (following official pattern)
                print("🚀 Starting script writing orchestration...")
                print("📋 Workflow: Manager → ScriptWriter → User Approval → AudioSelector → Final Results")
                print("⏰ Note: You'll be asked to approve the script when it's ready!")
                print("⏰ Workflow will continue until you approve the script or it reaches natural completion")
                print("-" * 80)
                
                # Clear pause flag at start
                clear_pause_flag()
                
                orchestration_result = await asyncio.wait_for(
                    group_chat_orchestration.invoke(
                        task=f"Please start working on this script writing request: {user_request}",
                        runtime=runtime,
                    ),
                    timeout=1800.0  # 30 minute timeout to allow for multiple user interactions and script revisions
                )
                
                # CRITICAL: Check for pause IMMEDIATELY after orchestration completes
                # This must be the first check to prevent any further processing
                print(f"🔍 DEBUG: Pause status after orchestration: {is_pause_requested()}")
                if is_pause_requested():
                    print("⏸️ Workflow pause detected during orchestration")
                    print("🔍 DEBUG: Entering pause handling flow...")
                    
                    # Create comprehensive chat history from captured conversation
                    chat_history = ChatHistory()
                    
                    # Add initial user request
                    chat_history.add_message(ChatMessageContent(
                        role=AuthorRole.USER,
                        content=user_request
                    ))
                    
                    # Add all captured conversation messages
                    full_conversation = get_full_conversation_history()
                    print(f"🔍 DEBUG: Captured conversation has {len(full_conversation)} messages")
                    for i, msg in enumerate(full_conversation):
                        role = AuthorRole.USER if msg['role'] == 'user' else AuthorRole.ASSISTANT
                        message = ChatMessageContent(
                            role=role,
                            content=msg['content']
                        )
                        if 'name' in msg:
                            message.name = msg['name']
                        chat_history.add_message(message)
                        print(f"🔍 DEBUG: Added message {i+1}: {msg['name']} - {msg['content'][:50]}...")
                    
                    # Create snapshot with WAITING_FOR_INPUT status
                    snapshot = serializer.create_conversation_snapshot(
                        conversation_id=conversation_id,
                        workflow_name="Script Writing Workflow",
                        workflow_type="script_writing",
                        user_request=user_request,
                        chat_history=chat_history,
                        group_chat_manager=group_chat_manager,
                        agents=self.agents,
                        checkpoint_reason="User requested workflow pause - waiting for approval/rejection",
                        status=WorkflowStatus.WAITING_FOR_INPUT,
                        current_agent="User",
                        resume_instructions="Workflow paused at user approval step. Resume with 'approve', 'reject', or feedback to continue."
                    )
                    
                    # Save the snapshot
                    serializer.save_conversation_snapshot(snapshot)
                    
                    pause_msg = (
                        "⏸️ Workflow paused successfully!\n"
                        f"📁 Saved as: {conversation_id}\n"
                        "💡 You can resume this workflow later from the main menu.\n"
                        "🔄 When resuming, provide your approval/rejection to continue."
                    )
                    print(pause_msg)
                    
                    # Create result file for paused state (but only for pause, not completion)
                    result_file = serializer.create_result_file(snapshot, pause_msg)
                    print(f"📄 Pause state saved to: {result_file}")
                    
                    # CRITICAL: Ensure we return immediately and don't continue processing
                    print("🛑 Pause handling complete - stopping workflow execution")
                    print(f"🔍 DEBUG: About to return from pause handling with message: {pause_msg[:50]}...")
                    return pause_msg
                
                # SAFETY CHECK: Double-check pause status before processing results
                # Check both pause flag and conversation history for pause requests
                full_conversation = get_full_conversation_history()
                last_user_input = None
                for msg in reversed(full_conversation):
                    if msg.get('role') == 'user' and msg.get('content', '').strip().lower() == 'pause':
                        last_user_input = msg.get('content', '').strip().lower()
                        break
                
                print(f"🔍 DEBUG: Final safety check - pause status: {is_pause_requested()}")
                print(f"🔍 DEBUG: Final safety check - last user input: {last_user_input}")
                if is_pause_requested() or last_user_input == 'pause':
                    print("🛑 SAFETY CHECK: Pause detected - aborting result processing")
                    return "Workflow paused - no results to process"
                
                # Get the final result with more lenient timeout
                print("⏳ Waiting for final results...")
                print("🔍 DEBUG: About to call orchestration_result.get()")
                try:
                    final_output = await asyncio.wait_for(
                        orchestration_result.get(),
                        timeout=60.0  # Increased timeout for result retrieval
                    )
                    
                    # Check if the final output indicates a pause
                    final_output_str = str(final_output)
                    is_paused_output = (
                        "paused by user request" in final_output_str.lower() or
                        "workflow paused" in final_output_str.lower()
                    )
                    
                    if is_paused_output:
                        print(f"🔍 DEBUG: Final output indicates pause: {final_output_str}")
                        # This should be handled as a pause, not completion
                        # Create comprehensive chat history from captured conversation
                        chat_history = ChatHistory()
                        
                        # Add initial user request
                        chat_history.add_message(ChatMessageContent(
                            role=AuthorRole.USER,
                            content=user_request
                        ))
                        
                        # Add all captured conversation messages
                        full_conversation = get_full_conversation_history()
                        for msg in full_conversation:
                            role = AuthorRole.USER if msg['role'] == 'user' else AuthorRole.ASSISTANT
                            message = ChatMessageContent(
                                role=role,
                                content=msg['content']
                            )
                            if 'name' in msg:
                                message.name = msg['name']
                            chat_history.add_message(message)
                        
                        # Create pause snapshot with WAITING_FOR_INPUT status
                        snapshot = serializer.create_conversation_snapshot(
                            conversation_id=conversation_id,
                            workflow_name="Script Writing Workflow",
                            workflow_type="script_writing",
                            user_request=user_request,
                            chat_history=chat_history,
                            group_chat_manager=group_chat_manager,
                            agents=self.agents,
                            checkpoint_reason="User requested workflow pause - waiting for approval/rejection",
                            status=WorkflowStatus.WAITING_FOR_INPUT,
                            current_agent="User",
                            resume_instructions="Workflow paused at user approval step. Resume with 'approve', 'reject', or feedback to continue."
                        )
                        
                        # Save the snapshot
                        serializer.save_conversation_snapshot(snapshot)
                        
                        # Create result file for paused state
                        result_file = serializer.create_result_file(snapshot, final_output_str)
                        print(f"📄 Pause result file created: {result_file}")
                        
                        return final_output_str
                    
                    print(f"✅ Script workflow completed successfully")
                    
                    # Create comprehensive chat history from captured conversation
                    chat_history = ChatHistory()
                    
                    # Add initial user request
                    chat_history.add_message(ChatMessageContent(
                        role=AuthorRole.USER,
                        content=user_request
                    ))
                    
                    # Add all captured conversation messages
                    full_conversation = get_full_conversation_history()
                    for msg in full_conversation:
                        role = AuthorRole.USER if msg['role'] == 'user' else AuthorRole.ASSISTANT
                        message = ChatMessageContent(
                            role=role,
                            content=msg['content']
                        )
                        if 'name' in msg:
                            message.name = msg['name']
                        chat_history.add_message(message)
                    
                    # Add final output if not already included
                    if final_output and str(final_output) not in [msg['content'] for msg in full_conversation]:
                        chat_history.add_message(ChatMessageContent(
                            role=AuthorRole.ASSISTANT,
                            content=str(final_output)
                        ))
                    
                    # Create comprehensive conversation snapshot
                    snapshot = serializer.create_conversation_snapshot(
                        conversation_id=conversation_id,
                        workflow_name="Script Writing Workflow",
                        workflow_type="script_writing",
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
                    
                    # Create comprehensive result file
                    result_file = serializer.create_result_file(snapshot, result_content)
                    print(f"📄 Result file created: {result_file}")
                    
                    return result_content
                        
                except asyncio.TimeoutError:
                    # If result retrieval times out, determine actual workflow status
                    from src.serialization.conversation_serializer import WorkflowStatus
                    
                    print("⚠️ Result retrieval timed out, checking actual workflow status...")
                    print("🔄 Attempting to retrieve conversation history from manager...")
                    
                    # Determine actual workflow status based on manager state
                    actual_status = self._determine_workflow_status(group_chat_manager)
                    current_phase = 'unknown'  # Prompt-based approach is stateless
                    user_approval = None  # Prompt-based approach is stateless
                    
                    print(f"🔍 Actual workflow status: {actual_status}")
                    print(f"🔍 Current phase: {current_phase}")
                    print(f"🔍 User approval: {user_approval}")
                    
                    # Create comprehensive chat history from captured conversation
                    chat_history = ChatHistory()
                    
                    # Add initial user request
                    chat_history.add_message(ChatMessageContent(
                        role=AuthorRole.USER,
                        content=user_request
                    ))
                    
                    # Add all captured conversation messages
                    full_conversation = get_full_conversation_history()
                    print(f"🔍 DEBUG: Captured conversation has {len(full_conversation)} messages")
                    for i, msg in enumerate(full_conversation):
                        role = AuthorRole.USER if msg['role'] == 'user' else AuthorRole.ASSISTANT
                        message = ChatMessageContent(
                            role=role,
                            content=msg['content']
                        )
                        if 'name' in msg:
                            message.name = msg['name']
                        chat_history.add_message(message)
                        print(f"🔍 DEBUG: Added message {i+1}: {msg['name']} - {msg['content'][:50]}...")
                    
                    # Set resume instructions based on status
                    if actual_status == WorkflowStatus.RUNNING:
                        resume_instructions = f"Workflow was interrupted during {current_phase} phase. User approval status: {user_approval}"
                        checkpoint_reason = "Workflow interrupted during execution"
                    elif actual_status == WorkflowStatus.COMPLETED:
                        resume_instructions = "Workflow completed but result retrieval timed out"
                        checkpoint_reason = "Result retrieval timed out after completion"
                    else:
                        resume_instructions = f"Workflow failed or was paused. Phase: {current_phase}, Approval: {user_approval}"
                        checkpoint_reason = "Workflow did not complete successfully"
                    
                    snapshot = serializer.create_conversation_snapshot(
                        conversation_id=conversation_id,
                        workflow_name="Script Writing Workflow",
                        workflow_type="script_writing",
                        user_request=user_request,
                        chat_history=chat_history,
                        group_chat_manager=group_chat_manager,
                        agents=self.agents,
                        checkpoint_reason=checkpoint_reason,
                        status=actual_status,
                        current_agent=None,
                        resume_instructions=resume_instructions
                    )
                    
                    # Save the snapshot
                    serializer.save_conversation_snapshot(snapshot)
                    
                    # Try to get the orchestration result directly
                    # Since the manager has completed, try to access the conversation
                    print("🔄 Creating workflow completion summary...")
                    result_content = self._create_workflow_completion_summary(user_request)
                    
                    # Create result file with full conversation
                    result_file = serializer.create_result_file(snapshot, result_content)
                    print(f"📄 Result file created: {result_file}")
                    
                    return result_content
                
            except asyncio.TimeoutError:
                # Create snapshot for timeout case with proper status determination
                from src.serialization.conversation_serializer import WorkflowStatus
                
                print("⏰ Workflow timed out, determining final status...")
                actual_status = self._determine_workflow_status(group_chat_manager)
                current_phase = 'unknown'  # Prompt-based approach is stateless
                user_approval = None  # Prompt-based approach is stateless
                
                print(f"🔍 Final workflow status: {actual_status}")
                
                chat_history = ChatHistory()
                
                # Add initial user request
                chat_history.add_message(ChatMessageContent(
                    role=AuthorRole.USER,
                    content=user_request
                ))
                
                # Add all captured conversation messages
                full_conversation = get_full_conversation_history()
                for msg in full_conversation:
                    role = AuthorRole.USER if msg['role'] == 'user' else AuthorRole.ASSISTANT
                    message = ChatMessageContent(
                        role=role,
                        content=msg['content']
                    )
                    if 'name' in msg:
                        message.name = msg['name']
                    chat_history.add_message(message)
                
                # Set status as FAILED only if it actually failed, otherwise preserve actual status
                if actual_status == WorkflowStatus.RUNNING and user_approval is False:
                    final_status = WorkflowStatus.PAUSED  # User rejected, can be resumed
                    checkpoint_reason = f"Workflow paused after user rejection in {current_phase} phase"
                    resume_instructions = f"Workflow paused - user rejected script. Can be resumed to continue with revisions."
                elif actual_status == WorkflowStatus.COMPLETED:
                    final_status = WorkflowStatus.COMPLETED
                    checkpoint_reason = "Workflow completed but timed out during result processing"
                    resume_instructions = "Workflow completed successfully but result processing timed out"
                else:
                    final_status = WorkflowStatus.FAILED
                    checkpoint_reason = "Workflow timed out during execution"
                    resume_instructions = "Workflow timed out - check API configuration and network connectivity"
                
                snapshot = serializer.create_conversation_snapshot(
                    conversation_id=conversation_id,
                    workflow_name="Script Writing Workflow",
                    workflow_type="script_writing",
                    user_request=user_request,
                    chat_history=chat_history,
                    group_chat_manager=group_chat_manager,
                    agents=self.agents,
                    checkpoint_reason=checkpoint_reason,
                    status=final_status,
                    current_agent=None,
                    resume_instructions=resume_instructions
                )
                
                # Save the snapshot
                serializer.save_conversation_snapshot(snapshot)
                
                error_msg = (
                    "❌ Script workflow timed out. This usually means:\n"
                    "1. API calls are failing due to invalid API keys\n"
                    "2. Network connectivity issues\n"
                    "3. The conversation is taking too long to reach a conclusion\n\n"
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
            print(f"🔍 DEBUG: Exception caught in script workflow: {e}")
            print(f"🔍 DEBUG: Pause status during exception: {is_pause_requested()}")
            error_msg = f"❌ Error in script workflow: {e}"
            print(error_msg)
            return error_msg
    
    def _determine_workflow_status(self, group_chat_manager):
        """
        Determine the actual workflow status for the new prompt-based approach.
        
        Args:
            group_chat_manager: The workflow's group chat manager
            
        Returns:
            Actual WorkflowStatus (defaults to RUNNING since stateless)
        """
        try:
            from src.serialization.conversation_serializer import WorkflowStatus
            
            # In our new prompt-based approach, the manager is stateless
            # All decisions are made by LLM based on conversation history
            # So we default to RUNNING unless we can determine otherwise
            
            print(f"🔍 Determining status for prompt-based manager")
            
            # Since we're stateless, we can't determine exact status from internal state
            # Default to RUNNING which is appropriate for most cases
            return WorkflowStatus.RUNNING
            
        except Exception as e:
            print(f"⚠️ Error determining workflow status: {e}")
            from src.serialization.conversation_serializer import WorkflowStatus
            return WorkflowStatus.RUNNING
    
    def _create_workflow_completion_summary(self, user_request: str) -> str:
        """
        Create a summary when workflow completes but result retrieval times out.
        This happens when the workflow completes successfully but the result formatting times out.
        """
        return f"""
📝 SCRIPT WORKFLOW STATUS SUMMARY
========================================

📋 Request: {user_request}
🎯 Process: Script → User Approval → Audio Selection

⚠️ Result retrieval timed out during workflow execution.
💡 Check the console output above for the full conversation details
📊 Check saved conversation file for complete workflow state

Note: This may be due to result formatting timeout or incomplete workflow execution.
        """.strip()
    
    async def resume_conversation(self, snapshot, user_input: str) -> str:
        """
        Resume a paused workflow with user input.
        
        Args:
            snapshot: ConversationSnapshot containing the saved workflow state
            user_input: User's approval/rejection/feedback
            
        Returns:
            The workflow result as a string
        """
        try:
            print(f"🔄 Resuming workflow: {snapshot.conversation_id}")
            print(f"📝 User input: {user_input}")
            
            # CRITICAL: Clear pause flag when resuming - this allows workflow to continue
            clear_pause_flag()
            print("🔄 Cleared pause flag for resumed workflow")
            
            # Clear conversation history for resumed workflow
            clear_conversation_history()
            
            # Restore the conversation history from snapshot
            restored_chat_history = self._restore_chat_history_from_snapshot(snapshot)
            
            # Add the user input to conversation history 
            user_message = ChatMessageContent(role=AuthorRole.USER, content=user_input)
            restored_chat_history.add_message(user_message)
            
            # Also add to captured conversation history
            global _conversation_history
            _conversation_history.append({
                'name': 'User',
                'role': 'user',
                'content': user_input
            })
            
            print(f"✅ Restored {len(restored_chat_history.messages)} messages from saved conversation")
            print(f"💬 Added user input: {user_input}")
            
            # Customize agents for workflow execution
            self._customize_agents_for_workflow()
            
            # Create new group chat manager
            group_chat_manager = ScriptWritingGroupChatManager(
                topic=snapshot.metadata.user_request,
                service=self.service,
                max_rounds=15,
            )
            
            # Create the GroupChatOrchestration
            group_chat_orchestration = GroupChatOrchestration(
                members=self.agents,
                manager=group_chat_manager,
                agent_response_callback=agent_response_callback,
            )
            
            # Create and start the runtime
            runtime = InProcessRuntime()
            runtime.start()
            
            try:
                print("🚀 Resuming workflow execution...")
                print("⏰ Workflow will continue automatically from where it was paused")
                print("-" * 80)
                
                # SAFETY CHECK: Confirm pause flag is cleared before continuing
                print(f"🔍 DEBUG: Pause flag before continuing: {is_pause_requested()}")
                
                # Create a comprehensive task message with conversation context
                # This helps the manager understand where we are in the workflow
                conversation_context = self._create_resume_context(snapshot, user_input)
                
                print(f"🔍 DEBUG: Resume context: {conversation_context[:200]}...")
                
                # Continue the workflow - the manager will analyze the conversation and continue appropriately
                # Note: GroupChatOrchestration doesn't accept chat_history parameter directly
                # The manager will analyze the conversation based on the messages it processes
                orchestration_result = await asyncio.wait_for(
                    group_chat_orchestration.invoke(
                        task=conversation_context,
                        runtime=runtime,
                    ),
                    timeout=1800.0
                )
                
                # Get the final result
                print("⏳ Waiting for final results...")
                final_output = await asyncio.wait_for(
                    orchestration_result.get(),
                    timeout=60.0
                )
                
                print(f"✅ Resumed workflow completed successfully")
                
                # Create comprehensive result
                return await self._finalize_resumed_workflow(snapshot, user_input, final_output, restored_chat_history)
                
            except asyncio.TimeoutError:
                return await self._handle_resumed_workflow_timeout(snapshot, user_input, restored_chat_history)
                
            finally:
                try:
                    await asyncio.wait_for(runtime.stop_when_idle(), timeout=10.0)
                    print("🛑 Runtime stopped successfully")
                except:
                    print("⚠️ Warning: Runtime cleanup issue")
                    
        except Exception as e:
            error_msg = f"❌ Error resuming workflow: {e}"
            print(error_msg)
            return error_msg
    
    def _restore_chat_history_from_snapshot(self, snapshot):
        """Restore ChatHistory from conversation snapshot."""
        chat_history = ChatHistory()
        
        for msg_dict in snapshot.chat_history:
            role_str = msg_dict.get('role', 'assistant')
            if role_str == 'user':
                role = AuthorRole.USER
            elif role_str == 'system':
                role = AuthorRole.SYSTEM
            else:
                role = AuthorRole.ASSISTANT
            
            message = ChatMessageContent(
                role=role,
                content=msg_dict.get('content', '')
            )
            
            if 'name' in msg_dict:
                message.name = msg_dict['name']
            
            chat_history.add_message(message)
            
        return chat_history
    
    async def _finalize_resumed_workflow(self, snapshot, user_input, final_output, chat_history):
        """Finalize a resumed workflow with results."""
        from src.serialization.conversation_serializer import WorkflowSerializer, WorkflowStatus
        serializer = WorkflowSerializer()
        
        # Add all captured conversation messages to chat history
        full_conversation = get_full_conversation_history()
        for msg in full_conversation:
            role = AuthorRole.USER if msg['role'] == 'user' else AuthorRole.ASSISTANT
            message = ChatMessageContent(
                role=role,
                content=msg['content']
            )
            if 'name' in msg:
                message.name = msg['name']
            chat_history.add_message(message)
        
        # Create updated snapshot (manager not available after completion, create dummy)
        dummy_manager = ScriptWritingGroupChatManager(
            topic=snapshot.metadata.user_request,
            service=self.service,
            max_rounds=15,
        )
        
        updated_snapshot = serializer.create_conversation_snapshot(
            conversation_id=snapshot.conversation_id,
            workflow_name=snapshot.metadata.workflow_name,
            workflow_type=snapshot.metadata.workflow_type,
            user_request=snapshot.metadata.user_request,
            chat_history=chat_history,
            group_chat_manager=dummy_manager,
            agents=self.agents,
            checkpoint_reason="Workflow resumed and completed successfully",
            status=WorkflowStatus.COMPLETED,
            current_agent=None,
            resume_instructions=None
        )
        
        # Save the updated snapshot
        serializer.save_conversation_snapshot(updated_snapshot)
        
        # Handle different result types
        if isinstance(final_output, list):
            if final_output and hasattr(final_output[0], 'content'):
                result_content = str(final_output[0].content)
            else:
                result_content = str(final_output)
        elif hasattr(final_output, 'content'):
            result_content = str(final_output.content)
        else:
            result_content = str(final_output)
        
        # Create result file
        result_file = serializer.create_result_file(updated_snapshot, result_content)
        print(f"📄 Resumed workflow result saved to: {result_file}")
        
        return result_content
    
    async def _handle_resumed_workflow_timeout(self, snapshot, user_input, chat_history):
        """Handle timeout in resumed workflow."""
        from src.serialization.conversation_serializer import WorkflowSerializer, WorkflowStatus
        serializer = WorkflowSerializer()
        
        print("⚠️ Resumed workflow timed out")
        
        # Create comprehensive chat history from captured conversation
        full_conversation = get_full_conversation_history()
        for msg in full_conversation:
            role = AuthorRole.USER if msg['role'] == 'user' else AuthorRole.ASSISTANT
            message = ChatMessageContent(
                role=role,
                content=msg['content']
            )
            if 'name' in msg:
                message.name = msg['name']
            chat_history.add_message(message)
        
        # Create updated snapshot (manager not available after timeout, create dummy)
        dummy_manager = ScriptWritingGroupChatManager(
            topic=snapshot.metadata.user_request,
            service=self.service,
            max_rounds=15,
        )
        
        updated_snapshot = serializer.create_conversation_snapshot(
            conversation_id=snapshot.conversation_id,
            workflow_name=snapshot.metadata.workflow_name,
            workflow_type=snapshot.metadata.workflow_type,
            user_request=snapshot.metadata.user_request,
            chat_history=chat_history,
            group_chat_manager=dummy_manager,
            agents=self.agents,
            checkpoint_reason="Workflow resumed but timed out during execution",
            status=WorkflowStatus.RUNNING,
            current_agent=None,
            resume_instructions=f"Workflow was resumed with input '{user_input}' but timed out. Can be resumed again."
        )
        
        # Save the updated snapshot
        serializer.save_conversation_snapshot(updated_snapshot)
        
        timeout_msg = (
            "⚠️ Resumed workflow timed out during execution.\n"
            "💡 The workflow state has been saved and can be resumed again.\n"
            "🔧 Check your API configuration and network connectivity."
        )
        
        result_file = serializer.create_result_file(updated_snapshot, timeout_msg)
        print(f"📄 Timeout state saved to: {result_file}")
        
        return timeout_msg
    
    def _create_resume_context(self, snapshot, user_input: str) -> str:
        """
        Create a comprehensive context message for resuming a workflow.
        This helps the manager understand exactly where we are in the workflow.
        """
        try:
            # Analyze the conversation to understand current state
            last_script_message = None
            last_user_message = None
            
            # Find the last script and last user input from conversation
            for msg in reversed(snapshot.chat_history):
                if (not last_script_message and 
                    msg.get('name') == 'ScriptWriter' and 
                    'title' in msg.get('content', '').lower()):
                    last_script_message = msg.get('content', '')
                
                if (not last_user_message and 
                    msg.get('role') == 'user' and 
                    msg.get('content', '').strip().lower() != 'pause'):
                    last_user_message = msg.get('content', '')
            
            # Get script preview
            script_preview = ""
            if last_script_message:
                lines = last_script_message.split('\n')
                title_line = next((line for line in lines if 'title' in line.lower()), '')
                script_preview = f"{title_line[:100]}..." if title_line else last_script_message[:100] + "..."
            
            # Determine workflow phase
            workflow_phase = "script_approval"  # Default for resumed workflows
            next_agent = "AudioSelector"  # Default next step after approval
            
            if user_input.lower().strip() == 'approve':
                action_description = "User APPROVED the script"
                next_action = f"proceed to {next_agent} for audio recommendations"
            elif user_input.lower().strip() == 'reject':
                action_description = "User REJECTED the script"
                next_agent = "ScriptWriter"
                next_action = f"delegate to {next_agent} for script revision"
            else:
                action_description = f"User provided feedback: {user_input}"
                next_action = f"analyze feedback and delegate to appropriate agent"
            
            # Create comprehensive context
            context = f"""RESUMING SCRIPT WRITING WORKFLOW - CONVERSATION CONTEXT:

ORIGINAL REQUEST: {snapshot.metadata.user_request}

WORKFLOW STATE:
- Phase: {workflow_phase} (user approval step)
- Previous work: ScriptWriter created a script about {snapshot.metadata.user_request}
- Script preview: {script_preview}

CURRENT SITUATION:
- {action_description}: "{user_input}"
- Required action: {next_action}

WORKFLOW SEQUENCE:
✅ ScriptWriter created script → ⏸️ User paused → 🔄 RESUMED → {action_description}
➡️ NEXT: {next_action}

CRITICAL CONTEXT FOR MANAGER:
- DO NOT create new scripts - a script about {snapshot.metadata.user_request} already exists
- User input "{user_input}" is responding to the EXISTING script
- If approved: delegate to AudioSelector for music/sound recommendations
- If rejected: delegate to ScriptWriter for revision of existing script
- If feedback: delegate to ScriptWriter for modifications

Manager: Analyze this context and coordinate the next step appropriately."""
            
            return context
            
        except Exception as e:
            # Fallback context if analysis fails
            return f"""RESUMING SCRIPT WRITING WORKFLOW:
Original request: {snapshot.metadata.user_request}
User response: {user_input}
Action needed: Continue workflow based on user input.
Manager: Please analyze the conversation and proceed accordingly."""

    def get_agents_info(self) -> Dict[str, Any]:
        """Return information about the workflow agents."""
        agent_names = []
        for agent in self.agents:
            if agent is not None:
                agent_names.append(getattr(agent, 'name', 'Unknown'))
        
        return {
            "total_agents": len(self.agents),
            "agent_names": agent_names,
            "workflow_type": "script_writing_with_audio",
            "orchestration_type": "GroupChatOrchestration",
            "manager_type": "ScriptWritingGroupChatManager",
            "workflow_sequence": [
                "1. Manager -> ScriptWriter creates script",
                "2. Manager asks user for approval", 
                "3. If rejected: Manager -> ScriptWriter rewrites",
                "4. If approved: Manager -> AudioSelector suggests audio",
                "5. Manager presents final results and terminates"
            ],
            "follows_official_pattern": True
        } 