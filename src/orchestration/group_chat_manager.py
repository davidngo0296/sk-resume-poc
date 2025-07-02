"""
Group Chat Manager for orchestrating multi-agent workflows.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
    from semantic_kernel.agents.agent import Agent
    from semantic_kernel.agents.strategies import (
        KernelFunctionSelectionStrategy,
        KernelFunctionTerminationStrategy,
    )
    from semantic_kernel.contents import ChatMessageContent, AuthorRole
    from semantic_kernel.functions import KernelFunctionFromPrompt

# Import with error handling for runtime
try:
    from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
    from semantic_kernel.agents.agent import Agent
    from semantic_kernel.agents.strategies import (
        KernelFunctionSelectionStrategy,
        KernelFunctionTerminationStrategy,
    )
    from semantic_kernel.contents import ChatMessageContent, AuthorRole
    from semantic_kernel.functions import KernelFunctionFromPrompt
    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError:
    SEMANTIC_KERNEL_AVAILABLE = False
    print("Warning: Semantic Kernel imports not available")


class MultiAgentGroupChatManager:
    """
    Base class for managing multi-agent group chat orchestration.
    Completely generic - no task-specific logic.
    """
    
    def __init__(self, manager_agent, specialist_agents):
        """
        Initialize the group chat manager.
        
        Args:
            manager_agent: The primary coordination agent
            specialist_agents: List of specialist agents that can be delegated to
        """
        self.manager_agent = manager_agent
        self.specialist_agents = specialist_agents
        self.workflow_state = {"conversation_history": []}
        self.workflow_context = {}
        
        # Initialize workflow configuration with defaults
        self.workflow_config = {
            "requires_approval_after": [],
            "auto_proceed_agents": [],
            "workflow_sequence": []
        }
        
        # Create agent list - filter out None values
        self.agents = [agent for agent in [
            self.manager_agent.get_agent(),
            *[spec_agent.get_agent() for spec_agent in self.specialist_agents]
        ] if agent is not None]
        
        # Group chat will be initialized when session starts
        self.group_chat: Optional["AgentGroupChat"] = None
    
    def _create_selection_strategy(self):
        """Create a selection strategy for determining which agent should respond next."""
        if not SEMANTIC_KERNEL_AVAILABLE:
            return None
            
        manager_name = self.manager_agent.name
        
        # Build participant list dynamically from available agents
        participants = [manager_name]
        for spec_agent in self.specialist_agents:
            participants.append(spec_agent.name)
        
        participant_list = "\n".join([f"- {name}" for name in participants])
        
        selection_function = KernelFunctionFromPrompt(
            function_name="selection",
            prompt=f"""
Examine the provided RESPONSE and choose the next participant.
State only the name of the chosen participant without explanation.
Never choose the participant named in the RESPONSE.

Choose only from these participants:
{participant_list}

Workflow Rules:
- If RESPONSE is user input, it is {manager_name}'s turn.
- If RESPONSE is by {manager_name} requesting delegation, choose the appropriate specialist.
- If RESPONSE is by a specialist agent, it is {manager_name}'s turn.
- Follow any workflow sequence if defined.

RESPONSE:
{{{{$lastmessage}}}}
""")
        
        try:
            return KernelFunctionSelectionStrategy(
                function=selection_function,
                kernel=self.manager_agent.kernel,
                result_parser=lambda result: str(result.value[0]).strip() if result.value and result.value[0] is not None else manager_name,
                history_variable_name="lastmessage",
            )
        except Exception as e:
            print(f"Warning: Could not create selection strategy: {e}")
            return None
    
    def _create_termination_strategy(self):
        """Create a termination strategy to know when the workflow is complete."""
        if not SEMANTIC_KERNEL_AVAILABLE:
            return None
            
        termination_keyword = "completed"
        
        # Default termination logic - can be overridden by derived classes
        termination_function = KernelFunctionFromPrompt(
            function_name="termination",
            prompt=f"""
Examine the RESPONSE and determine whether the workflow has been completed.
If the workflow is completed, respond with a single word: {termination_keyword}.

RESPONSE:
{{{{$lastmessage}}}}
""")
        
        try:
            # Use last agent for termination detection
            agent_list: List["Agent"] = [agent for agent in self.agents[-1:]]
            return KernelFunctionTerminationStrategy(
                agents=agent_list,
                function=termination_function,
                kernel=self.manager_agent.kernel,
                result_parser=lambda result: termination_keyword in str(result.value[0]).lower() if result.value and result.value[0] else False,
                history_variable_name="lastmessage",
                maximum_iterations=10,
            )
        except Exception as e:
            print(f"Warning: Could not create termination strategy: {e}")
            return None
    
    def set_workflow_context(self, workflow_type: str, workflow_description: str, agent_roles: Optional[Dict[str, str]] = None, workflow_config: Optional[Dict[str, Any]] = None):
        """
        Set the workflow context to be injected by the calling application.
        
        Args:
            workflow_type: Type of workflow (e.g., "Video Script Creation", "Blog Writing", etc.)
            workflow_description: Description of what this workflow accomplishes
            agent_roles: Dictionary mapping agent names to their roles in this workflow
            workflow_config: Configuration options for workflow behavior including:
                - requires_approval_after: List of agent names that require user approval after their work
                - auto_proceed_agents: List of agent names that can auto-proceed to next agents
                - workflow_sequence: Optional list defining the order of agent involvement
        """
        self.workflow_context = {
            "workflow_type": workflow_type,
            "workflow_description": workflow_description,
            "agent_roles": agent_roles or {}
        }
        
        # Set workflow configuration with defaults
        default_config = {
            "requires_approval_after": [],  # No approval required by default
            "auto_proceed_agents": [],      # No auto-proceed by default
            "workflow_sequence": []         # No enforced sequence by default
        }
        
        self.workflow_config = {**default_config, **(workflow_config or {})}
    
    async def start_session(self) -> None:
        """Initialize the group chat session."""
        if not SEMANTIC_KERNEL_AVAILABLE:
            print("Semantic Kernel not available, using direct agent communication")
            return
            
        if len(self.agents) == 0:
            print("Warning: No agents available for group chat. Using direct agent communication.")
            return
            
        try:
            # Create the AgentGroupChat with selection and termination strategies
            selection_strategy = self._create_selection_strategy()
            termination_strategy = self._create_termination_strategy()
            
            if selection_strategy and termination_strategy:
                # Cast to List[Agent] for type compatibility
                agent_list: List["Agent"] = [agent for agent in self.agents]
                self.group_chat = AgentGroupChat(
                    agents=agent_list,
                    selection_strategy=selection_strategy,
                    termination_strategy=termination_strategy,
                )
                print(f"Group chat initialized with {len(self.agents)} agents")
            else:
                print("Warning: Could not create group chat strategies. Using direct agent communication.")
        except Exception as e:
            print(f"Warning: Could not initialize group chat: {e}. Using direct agent communication.")
    
    async def process_user_message(self, user_message: str) -> str:
        """
        Process user message through the group chat workflow.
        Note: This method is maintained for compatibility but now uses streaming internally.
        """
        # Add user message to workflow history
        self._add_to_conversation_history("user", user_message)
        
        # Collect all streaming responses into a single string
        responses = []
        async for agent_name, response in self._controlled_agent_workflow_streaming(user_message):
            responses.append(f"{agent_name}: {response}")
        
        return "\n\n".join(responses) if responses else "Error: No response from agents"
    
    async def process_user_message_streaming(self, user_message: str):
        """
        Process user message with streaming responses from agents.
        Returns an async generator yielding (agent_name, response) tuples.
        """
        print(f"DEBUG: process_user_message_streaming called with: {user_message}")
        
        # Update conversation history
        self._add_to_conversation_history("user", user_message)
        
        # Delegate to controlled workflow
        try:
            print("DEBUG: About to call _controlled_agent_workflow_streaming")
            async for agent_name, response in self._controlled_agent_workflow_streaming(user_message):
                print(f"DEBUG: Got streaming response from {agent_name}: {response[:50]}...")
                # Update conversation history with agent responses
                self._add_to_conversation_history("assistant", f"{agent_name}: {response}")
                yield agent_name, response
        except Exception as e:
            print(f"Error in streaming workflow: {e}")
            yield "Manager", "Error: Unable to process request"
    
    def _detect_delegation(self, manager_response: str) -> Optional[str]:
        """
        Detect if the manager is delegating to a specialist agent.
        Base implementation with generic delegation detection.
        Can be overridden by derived classes for specific behavior.
        
        Args:
            manager_response: The response from the manager agent
            
        Returns:
            str: Name of the specialist agent to delegate to, or None if no delegation detected
        """
        response_lower = manager_response.lower()
        
        # First check for negative delegation contexts - if found, return None
        negative_contexts = [
            "will not delegate", "will not coordinate", "will not assign", 
            "no delegation", "not assign", "will not pass",
            "cannot delegate", "unable to delegate"
        ]
        
        for negative_context in negative_contexts:
            if negative_context in response_lower:
                print(f"DEBUG: Found negative delegation context: '{negative_context}' - ignoring agent mentions")
                return None
        
        # Find agents mentioned in the manager response
        mentioned_agents = []
        for spec_agent in self.specialist_agents:
            agent_name = spec_agent.name
            if agent_name.lower() in response_lower:
                mentioned_agents.append(agent_name)
        
        # Check for explicit delegation patterns with agent names
        for spec_agent in self.specialist_agents:
            agent_name = spec_agent.name
            agent_name_lower = agent_name.lower()
            
            # Look for active delegation patterns
            delegation_patterns = [
                f"delegate to {agent_name_lower}",
                f"coordinate with {agent_name_lower}",
                f"pass to {agent_name_lower}",
                f"assign to {agent_name_lower}",
                f"ask {agent_name_lower}",
                f"have {agent_name_lower}",
                f"{agent_name_lower} will handle",
                f"{agent_name_lower} should",
                f"notify {agent_name_lower}",
                f"{agent_name_lower}, please"
            ]
            
            for pattern in delegation_patterns:
                if pattern in response_lower:
                    print(f"DEBUG: Found delegation pattern '{pattern}' for agent {agent_name}")
                    return agent_name
        
        # Check for task-based delegation using workflow context
        agent_roles = self.workflow_context.get("agent_roles", {})
        
        # Look for role-based mentions in workflow context
        for agent_name, role_description in agent_roles.items():
            # Extract key terms from role description
            role_keywords = [word for word in role_description.lower().split() if len(word) > 4]
            
            # Check if manager response mentions role-related keywords with delegation context
            for keyword in role_keywords:
                if keyword in response_lower:
                    # Check if this keyword appears with delegation language
                    delegation_context_patterns = [
                        f"delegate.*{keyword}", f"coordinate.*{keyword}", f"assign.*{keyword}",
                        f"{keyword}.*delegate", f"{keyword}.*coordinate", f"{keyword}.*handle"
                    ]
                    
                    import re
                    for pattern in delegation_context_patterns:
                        if re.search(pattern, response_lower):
                            print(f"DEBUG: Found role-based delegation to {agent_name} based on keyword '{keyword}'")
                            return agent_name
        
        # Final fallback - if single agent mentioned, return it
        if len(mentioned_agents) == 1:
            print(f"DEBUG: Using fallback - single agent mentioned: {mentioned_agents[0]}")
            return mentioned_agents[0]
        elif len(mentioned_agents) > 1:
            print(f"DEBUG: Multiple agents mentioned but no clear delegation: {mentioned_agents}")
            return mentioned_agents[0]  # Return first mentioned as last resort
        
        print("DEBUG: No delegation detected in manager response")
        return None
    
    def _get_agent_by_name(self, agent_name: str):
        """
        Get agent wrapper by name.
        
        Args:
            agent_name: Name of the agent to find
            
        Returns:
            Agent wrapper instance or None if not found
        """
        # Check manager agent first
        if self.manager_agent.name == agent_name:
            return self.manager_agent
            
        # Check specialist agents
        for spec_agent in self.specialist_agents:
            if spec_agent.name == agent_name:
                return spec_agent
                
        print(f"DEBUG: Could not find agent with name: {agent_name}")
        print(f"DEBUG: Available agents: {[self.manager_agent.name] + [spec.name for spec in self.specialist_agents]}")
        return None


    
    async def _controlled_agent_workflow_streaming(self, user_message: str):
        """Streaming version of controlled agent workflow."""
        current_phase = self._get_current_phase()
        print(f"DEBUG: _controlled_agent_workflow_streaming called, current_phase: {current_phase}")
        
        if current_phase == "initial":
            print("DEBUG: In initial phase, about to get manager response")
            # Manager responds and determines next steps
            try:
                manager_response = await self._get_agent_response(self.manager_agent, user_message)
                print(f"DEBUG: Got manager response: {manager_response[:100]}...")
                yield "Manager", manager_response
                
                # Check if manager is delegating to any specialist agent
                delegated_agent_name = self._detect_delegation(manager_response)
                print(f"DEBUG: Delegation detected: {delegated_agent_name}")
                
                if delegated_agent_name:
                    specialist_agent = self._get_agent_by_name(delegated_agent_name)
                    print(f"DEBUG: Found specialist agent: {specialist_agent}")
                    
                    if specialist_agent:
                        try:
                            print(f"DEBUG: Invoking {delegated_agent_name} with user message: {user_message}")
                            specialist_response = await self._get_agent_response(specialist_agent, user_message)
                            print(f"DEBUG: Got response from {delegated_agent_name}: {specialist_response[:100]}...")
                            yield delegated_agent_name, specialist_response
                            
                            # Check if this agent requires user approval before proceeding
                            if delegated_agent_name in self.workflow_config.get("requires_approval_after", []):
                                print(f"DEBUG: {delegated_agent_name} requires user approval - workflow will pause for user input")
                                # Workflow will pause here and wait for user approval
                                return
                            
                            # Check if we should auto-proceed to next agent
                            if delegated_agent_name in self.workflow_config.get("auto_proceed_agents", []):
                                print(f"DEBUG: {delegated_agent_name} can auto-proceed - checking for next agent")
                                # Auto-proceed logic can be implemented here in the future
                                
                        except Exception as e:
                            print(f"DEBUG: Error invoking specialist agent: {e}")
                            yield delegated_agent_name, "Error: Agent unavailable"
                    else:
                        print(f"DEBUG: Could not find specialist agent for name: {delegated_agent_name}")
                    
            except Exception as e:
                print(f"DEBUG: Error in manager response: {e}")
                yield "Manager", "Error: Manager agent unavailable"
        
        elif current_phase == "awaiting_approval":
            if any(keyword in user_message.lower() for keyword in ["approve", "yes", "looks good", "accept", "good"]):
                # User approved - let manager coordinate next steps
                try:
                    manager_response = await self._get_agent_response(self.manager_agent, user_message)
                    yield "Manager", manager_response
                    
                    # Check if manager is delegating to any specialist agent for next step
                    delegated_agent_name = self._detect_delegation(manager_response)
                    if delegated_agent_name:
                        specialist_agent = self._get_agent_by_name(delegated_agent_name)
                        if specialist_agent:
                            try:
                                specialist_response = await self._get_agent_response(specialist_agent, user_message)
                                yield delegated_agent_name, specialist_response
                                
                                # Check if this agent also requires approval
                                if delegated_agent_name in self.workflow_config.get("requires_approval_after", []):
                                    print(f"DEBUG: {delegated_agent_name} also requires user approval")
                                    return
                                    
                            except Exception as e:
                                yield delegated_agent_name, "Error: Agent unavailable"
                            
                except Exception as e:
                    yield "Manager", "Error: Manager agent unavailable"
            else:
                # User wants changes - let manager coordinate the revision process
                try:
                    manager_response = await self._get_agent_response(self.manager_agent, user_message)
                    yield "Manager", manager_response
                    
                    # Check if manager is delegating revision to any specialist agent
                    delegated_agent_name = self._detect_delegation(manager_response)
                    if delegated_agent_name:
                        specialist_agent = self._get_agent_by_name(delegated_agent_name)
                        if specialist_agent:
                            try:
                                specialist_response = await self._get_agent_response(specialist_agent, user_message)
                                yield delegated_agent_name, specialist_response
                                
                                # After revision, check if approval is still required
                                if delegated_agent_name in self.workflow_config.get("requires_approval_after", []):
                                    print(f"DEBUG: Revised work from {delegated_agent_name} requires user approval")
                                    return
                                    
                            except Exception as e:
                                yield delegated_agent_name, "Error: Agent unavailable"
                            
                except Exception as e:
                    yield "Manager", "Error: Manager agent unavailable"
        
        else:
            # Default case - let manager handle with its own instructions
            try:
                manager_response = await self._get_agent_response(self.manager_agent, user_message)
                yield "Manager", manager_response
                
                # Check if the current Manager response contains delegation  
                delegated_agent_name = self._detect_delegation(manager_response)
                print(f"DEBUG: Checking current Manager response for delegation: {delegated_agent_name}")
                
                if delegated_agent_name:
                    specialist_agent = self._get_agent_by_name(delegated_agent_name)
                    if specialist_agent:
                        try:
                            print(f"DEBUG: Auto-invoking {delegated_agent_name} based on Manager delegation")
                            specialist_response = await self._get_agent_response(specialist_agent, user_message)
                            yield delegated_agent_name, specialist_response
                            
                            # Check if this agent requires user approval before proceeding
                            if delegated_agent_name in self.workflow_config.get("requires_approval_after", []):
                                print(f"DEBUG: {delegated_agent_name} requires user approval - workflow will pause for user input")
                                return
                            
                            # If we get here, continue the workflow without needing more user input
                            return
                            
                        except Exception as e:
                            print(f"DEBUG: Error invoking specialist agent: {e}")
                            yield delegated_agent_name, "Error: Agent unavailable"
                            return
                        
            except Exception as e:
                yield "Manager", "Error: Manager agent unavailable"


    
    async def _get_agent_response(self, agent, prompt: str) -> str:
        """Get a real LLM response from an individual agent."""
        try:
            from semantic_kernel.contents import ChatHistory
            from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
            from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
            from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
            
            # Create a temporary chat history with the prompt
            chat_history = ChatHistory()
            
            # Get conversation context
            conversation_history = self.workflow_state.get("conversation_history", [])
            original_request = self._extract_original_request()
            previous_work = self._get_latest_specialist_work()
            
            # Prepare context based on agent type
            context_prompt = self._prepare_agent_context(agent, prompt, conversation_history, original_request, previous_work)
            chat_history.add_user_message(context_prompt)
            
            # Get response from the agent's kernel using a more robust method
            settings = OpenAIChatPromptExecutionSettings(
                function_choice_behavior=FunctionChoiceBehavior.Auto(),
                max_tokens=1000,
                temperature=0.7
            )
            
            # Get the chat completion service more reliably
            chat_service = None
            for service in agent.kernel.services.values():
                if hasattr(service, 'get_chat_message_content'):
                    chat_service = service
                    break
            
            if chat_service is None:
                raise Exception("No chat completion service found in agent kernel")
            
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings,
                kernel=agent.kernel
            )
            
            return str(response.content) if response.content else "I apologize, but I couldn't generate a response at this time."
            
        except Exception as e:
            print(f"Error getting agent response: {e}")
            return self._get_error_message(agent)
    
    def _prepare_agent_context(self, agent, prompt: str, conversation_history: List[Dict], original_request: str, previous_work: str) -> str:
        """
        Prepare context for agent response. Can be overridden by derived classes.
        
        Args:
            agent: The agent to prepare context for
            prompt: The user's current prompt
            conversation_history: The conversation history
            original_request: The original user request
            previous_work: Any previous work completed
            
        Returns:
            str: The context prompt for the agent
        """
        if agent == self.manager_agent:
            return self._prepare_manager_context(prompt, conversation_history, original_request, previous_work)
        else:
            return self._prepare_specialist_context(agent, prompt, conversation_history, original_request, previous_work)
    
    def _prepare_manager_context(self, prompt: str, conversation_history: List[Dict], original_request: str, previous_work: str) -> str:
        """Prepare context for the manager agent. Generic implementation."""
        # Build agent list dynamically
        available_agents = []
        for a in self.agents:
            if a != self.manager_agent.get_agent():
                agent_name = getattr(a, 'name', 'Unknown')
                agent_role = self.workflow_context["agent_roles"].get(agent_name, getattr(a, 'description', 'Specialist agent'))
                available_agents.append(f"- {agent_name}: {agent_role}")
        
        agent_list = "\n".join(available_agents) if available_agents else "- No specialist agents available"
        workflow_sequence = self.workflow_config.get("workflow_sequence", [])
        workflow_info = f"Workflow sequence: {' → '.join(workflow_sequence)}" if workflow_sequence else ""
        
        return f"""
You are coordinating a {self.workflow_context.get("workflow_type", "multi-agent")} workflow.

Workflow Description: {self.workflow_context.get("workflow_description", "Coordinate multiple specialist agents")}
{workflow_info}

Available specialist agents:
{agent_list}

CONVERSATION CONTEXT:
- Original user request: {original_request if original_request else "Not specified"}
- Total messages: {len(conversation_history)}

CURRENT USER REQUEST: {prompt}

Your role: Coordinate and delegate work to appropriate specialist agents based on user requests.

DELEGATION RULES:
- Always mention the specific agent name when delegating
- Be explicit: "I'll coordinate with [AgentName]" or "I'll ask [AgentName] to..."
- Analyze the request and delegate to the most appropriate specialist

Based on this request, coordinate with the appropriate specialist agent(s).
"""
    
    def _prepare_specialist_context(self, agent, prompt: str, conversation_history: List[Dict], original_request: str, previous_work: str) -> str:
        """Prepare context for specialist agents. Generic implementation."""
        agent_name = getattr(agent, 'name', 'Unknown')
        agent_role = self.workflow_context["agent_roles"].get(agent_name, "Specialist agent")
        
        return f"""
You are the {agent_name} in a {self.workflow_context.get("workflow_type", "multi-agent")} workflow.

Your role: {agent_role}

CONVERSATION CONTEXT:
- Original user request: {original_request if original_request else "Create content as requested"}
- Total messages in conversation: {len(conversation_history)}
- Previous work done: {"Yes" if previous_work else "No"}

CURRENT USER REQUEST: {prompt}

Instructions:
- Create high-quality work based on the user's request
- Be professional and deliver actual content
- Focus on your specialized area of expertise

Please respond with your work for this {self.workflow_context.get("workflow_type", "workflow")} task.
"""
    
    def _get_error_message(self, agent) -> str:
        """Get appropriate error message for agent failures."""
        agent_name = getattr(agent, 'name', 'Unknown').lower()
        if "manager" in agent_name:
            return "Error: Manager agent unavailable"
        else:
            return f"Error: {getattr(agent, 'name', 'Agent')} unavailable"
    
    def get_workflow_state(self) -> Dict[str, Any]:
        """Return current workflow state for serialization."""
        return self.workflow_state.copy()
    
    def load_workflow_state(self, state: Dict[str, Any]) -> None:
        """Load workflow state from serialized data."""
        self.workflow_state.update(state)
    
    async def stop_session(self) -> None:
        """Stop the group chat session."""
        if self.group_chat:
            try:
                await self.group_chat.reset()
            except Exception as e:
                print(f"Warning: Could not reset group chat: {e}")
    
    def pause_conversation(self) -> Dict[str, Any]:
        """
        Pause the conversation and return serializable state.
        """
        return self.get_workflow_state()
    
    def reset_workflow_state(self) -> None:
        """
        Reset the workflow state to start a fresh conversation.
        """
        self.workflow_state = {"conversation_history": []}
    
    async def resume_conversation(self, serialized_state: Dict[str, Any]) -> str:
        """
        Resume a paused conversation from serialized state.
        """
        # Load the workflow state directly (no more double nesting)
        self.load_workflow_state(serialized_state)
        
        # Restart the session
        await self.start_session()
        
        return f"Conversation resumed! We're currently in the {self._get_current_phase()} phase. How can I help you continue?"
    
    def _add_to_conversation_history(self, role: str, content: str):
        """Add message to conversation history."""
        if "conversation_history" not in self.workflow_state:
            self.workflow_state["conversation_history"] = []
        
        self.workflow_state["conversation_history"].append({
            "role": role,
            "content": content
        })
    
    def _get_current_phase(self) -> str:
        """
        Determine the current phase based on conversation history analysis.
        
        Returns:
            str: Current workflow phase
        """
        history = self.workflow_state['conversation_history']
        
        # Initial phase: no conversation history or only user messages
        if not history:
            return "initial"
        
        # Check if we only have user messages (no agent responses yet)
        agent_messages = [msg for msg in history if msg.get('role') != 'user']
        if not agent_messages:
            return "initial"
        
        # Check if the last agent response was from an agent that requires approval
        if agent_messages:
            last_agent_message = agent_messages[-1]
            # Extract agent name from the message content (format: "AgentName: content")
            content = last_agent_message.get('content', '')
            last_agent_name = content.split(':')[0].strip() if ':' in content else ''
            
            # Check if this agent requires approval and if we haven't received user approval yet
            if last_agent_name in self.workflow_config.get("requires_approval_after", []):
                # Check if there's a user message after this agent's response
                last_agent_index = None
                for i, msg in enumerate(history):
                    if msg == last_agent_message:
                        last_agent_index = i
                        break
                
                if last_agent_index is not None:
                    # Check if there are any user messages after this agent response
                    user_messages_after = [msg for msg in history[last_agent_index + 1:] if msg.get('role') == 'user']
                    if not user_messages_after:
                        return "awaiting_approval"
        
        # If multiple exchanges exist, we're in working phase
        if len(agent_messages) >= 1:
            return "working"
        
        return "initial"
    
    def _get_latest_specialist_work(self) -> str:
        """Extract the latest work product from specialist agents."""
        history = self.workflow_state.get("conversation_history", [])
        
        # Look through history in reverse to find the most recent substantial work from non-manager agents
        manager_name = self.manager_agent.name.lower()
        for msg in reversed(history):
            content = msg.get("content", "")
            # Look for responses from non-manager agents with substantial content
            if manager_name not in content.lower() and ":" in content and len(content) > 100:
                return content
        
        return ""
    
    def _extract_original_request(self) -> str:
        """Extract the original user request from conversation history."""
        history = self.workflow_state.get("conversation_history", [])
        
        # Find the first user message in the conversation
        for msg in history:
            if msg.get("role") == "user":
                return msg.get("content", "")
        
        return "" 