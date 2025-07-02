"""
Script-specific Group Chat Manager for video script creation workflows.
Extends the base MultiAgentGroupChatManager with video script workflow intelligence.
"""

from typing import Optional, List, Dict, Any
from .group_chat_manager import MultiAgentGroupChatManager


class ScriptGroupChatManager(MultiAgentGroupChatManager):
    """
    Script-specific group chat manager that extends the base class with 
    video script workflow intelligence and delegation patterns.
    """
    
    def __init__(self, manager_agent, specialist_agents):
        """Initialize with video script workflow configuration."""
        super().__init__(manager_agent, specialist_agents)
        
        # Set up video script workflow context automatically
        self._setup_video_script_workflow()
    
    def _setup_video_script_workflow(self):
        """Configure the workflow for video script creation."""
        self.set_workflow_context(
            workflow_type="Video Script Creation",
            workflow_description="Create professional video scripts with audio recommendations",
            agent_roles={
                "ScriptWriter": "Creates engaging and professional video scripts based on user requirements and feedback",
                "AudioSelector": "Provides audio recommendations and music suggestions that complement video scripts"
            },
            workflow_config={
                "requires_approval_after": ["ScriptWriter"],  # Require approval after script creation
                "auto_proceed_agents": [],  # No auto-proceed agents for this workflow
                "workflow_sequence": ["ScriptWriter", "AudioSelector"]  # Define the typical flow
            }
        )
    
    def get_workflow_display_info(self) -> Dict[str, Any]:
        """Get display information for the UI."""
        return {
            "title": "🎬 Video Script Multi-Agent Team POC",
            "subtitle": "Powered by Semantic Kernel Group Chat Orchestration",
            "welcome_title": "Welcome",
            "new_session_action": "Start new video script creation",
            "new_session_prompt": "Describe the video script you want to create",
            "session_starting_message": "Starting new video script creation session...",
            "goodbye_message": "Thank you for using Video Script Multi-Agent Team! 👋"
        }
    
    def get_agent_colors(self) -> Dict[str, str]:
        """Get color mapping for agents in the UI."""
        return {
            "Manager": "cyan",
            "ScriptWriter": "green", 
            "AudioSelector": "magenta"
        }
    
    def is_workflow_complete(self, conversation_history: List[Dict]) -> bool:
        """
        Check if the video script workflow is complete.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            bool: True if workflow appears complete
        """
        if not conversation_history or len(conversation_history) < 3:
            return False
        
        # Look for AudioSelector providing actual recommendations (not just Manager mentioning them)
        for msg in reversed(conversation_history[-3:]):  # Check last 3 messages
            content = msg.get("content", "")
            role = msg.get("role", "")
            # Check if AudioSelector provided substantial work
            if ("audioselector" in role.lower() or "AudioSelector:" in content) and len(content) > 200:
                if any(keyword in content.lower() for keyword in ["music", "track", "audio", "sound", "recommendation"]):
                    return True
        
        return False
    
    def _detect_delegation(self, manager_response: str) -> Optional[str]:
        """
        Enhanced delegation detection with script workflow sequence awareness.
        
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
        
        # PRIORITY 1: Check workflow sequence context first (approval/revision scenarios)
        conversation_history = self.workflow_state.get("conversation_history", [])
        workflow_sequence = self.workflow_config.get("workflow_sequence", [])
        
        # Get recent user messages to understand context
        user_messages = [msg for msg in reversed(conversation_history[-3:]) 
                        if msg.get("role") == "user"]
        last_user_message = user_messages[0].get("content", "").lower() if user_messages else ""
        
        # Find agents mentioned in the manager response
        mentioned_agents = []
        for spec_agent in self.specialist_agents:
            agent_name = spec_agent.name
            if agent_name.lower() in response_lower:
                mentioned_agents.append(agent_name)
        
        # If we have workflow context and multiple agents mentioned, use smart routing
        if mentioned_agents and workflow_sequence and last_user_message:
            # Find the most recent agent that provided substantial work
            last_working_agent = None
            for msg in reversed(conversation_history[-5:]):  # Check last 5 messages
                content = msg.get("content", "")
                if len(content) > 100:  # Substantial work
                    for agent_name in mentioned_agents:
                        if agent_name.lower() in content.lower() and ":" in content:
                            last_working_agent = agent_name
                            break
                    if last_working_agent:
                        break
            
            if last_working_agent and last_working_agent in workflow_sequence:
                try:
                    current_index = workflow_sequence.index(last_working_agent)
                    
                    # Check user feedback type
                    approval_keywords = ["approve", "yes", "looks good", "accept", "proceed", "next"]
                    revision_keywords = ["reject", "no", "revise", "change", "shorter", "longer", "different"]
                    
                    if any(keyword in last_user_message for keyword in approval_keywords):
                        # Move to next agent in sequence
                        if current_index + 1 < len(workflow_sequence):
                            next_agent = workflow_sequence[current_index + 1]
                            if next_agent in mentioned_agents:
                                print(f"DEBUG: WORKFLOW PRIORITY - Approval detected, moving to next agent: {next_agent}")
                                return next_agent
                    
                    elif any(keyword in last_user_message for keyword in revision_keywords):
                        # Stay with current agent for revision
                        if last_working_agent in mentioned_agents:
                            print(f"DEBUG: WORKFLOW PRIORITY - Revision detected, staying with current agent: {last_working_agent}")
                            return last_working_agent
                
                except (ValueError, IndexError):
                    print(f"DEBUG: Error in workflow sequence logic")
        
        # Fall back to base class delegation detection
        return super()._detect_delegation(manager_response)
    
    def _prepare_manager_context(self, prompt: str, conversation_history: List[Dict], original_request: str, previous_work: str) -> str:
        """Prepare script-specific context for the manager agent."""
        # Build agent list dynamically
        available_agents = []
        for a in self.agents:
            if a != self.manager_agent.get_agent():
                agent_name = getattr(a, 'name', 'Unknown')
                agent_role = self.workflow_context["agent_roles"].get(agent_name, getattr(a, 'description', 'Specialist agent'))
                available_agents.append(f"- {agent_name}: {agent_role}")
        
        agent_list = "\n".join(available_agents) if available_agents else "- No specialist agents available"
        
        # Analyze conversation context for workflow state
        recent_work_summary = self._get_recent_work_summary()
        is_user_feedback = len(conversation_history) > 1 and any(keyword in prompt.lower() for keyword in 
            ["reject", "approve", "yes", "no", "revise", "change", "good", "bad", "shorter", "longer", "different"])
        
        workflow_sequence = self.workflow_config.get("workflow_sequence", [])
        workflow_info = f"Workflow sequence: {' → '.join(workflow_sequence)}" if workflow_sequence else ""
        
        return f"""
You are coordinating a {self.workflow_context["workflow_type"]} workflow.

Workflow Description: {self.workflow_context["workflow_description"]}
{workflow_info}

Available specialist agents:
{agent_list}

CONVERSATION CONTEXT:
- Original user request: {original_request if original_request else "Not specified"}
- Total messages: {len(conversation_history)}
- Recent work completed: {recent_work_summary}

CURRENT SITUATION:
- User input: "{prompt}"
- Is this user feedback on completed work? {"YES" if is_user_feedback else "NO"}

{"🚨 CRITICAL - USER FEEDBACK DETECTED:" if is_user_feedback else "📝 NEW REQUEST:"}
{'''
This is user feedback on previously completed work. You MUST delegate to appropriate agents:
- "reject", "no", "revise", "change" = delegate to the SAME agent that just completed work for revision
- "approve", "yes", "good" = delegate to the NEXT agent in the workflow sequence
- Never say "no further action required" when user gives feedback on completed work
- Always delegate when user provides feedback - this is workflow progression!''' if is_user_feedback else '''
This appears to be a new request. Analyze and delegate to appropriate specialist agents.'''}

Your role: Coordinate and delegate work to appropriate specialist agents based on user requests and feedback.

DELEGATION RULES:
- Always mention the specific agent name when delegating
- For feedback: Always delegate to continue the workflow
- For new requests: Delegate to appropriate starting agent
- Be explicit: "I'll coordinate with [AgentName]" or "I'll ask [AgentName] to..."

Based on this {'feedback' if is_user_feedback else 'request'}, coordinate with the appropriate specialist agent(s).
"""
    
    def _prepare_specialist_context(self, agent, prompt: str, conversation_history: List[Dict], original_request: str, previous_work: str) -> str:
        """Prepare script-specific context for specialist agents."""
        agent_name = getattr(agent, 'name', 'Unknown')
        agent_role = self.workflow_context["agent_roles"].get(agent_name, "Specialist agent")
        
        # Check if this is a revision/rejection scenario
        is_revision = any(keyword in prompt.lower() for keyword in ["reject", "revise", "change", "redo", "again", "different", "shorter", "longer"])
        
        return f"""
You are the {agent_name} in a {self.workflow_context["workflow_type"]} workflow.

Your role: {agent_role}

CONVERSATION CONTEXT:
- Original user request: {original_request if original_request else "Create content as requested"}
- Total messages in conversation: {len(conversation_history)}
- Previous work done: {"Yes" if previous_work else "No"}

{"REVISION REQUEST DETECTED: The user is asking for changes to previous work." if is_revision else ""}

{f"PREVIOUS WORK TO REVISE: {previous_work[:500]}..." if is_revision and previous_work else ""}

CURRENT USER REQUEST: {prompt}

Instructions:
{"- This is a revision request - create NEW content based on the original topic" if is_revision else "- Create content based on the user's request"}
{"- Do NOT ask for more details - be proactive and create the content" if is_revision else ""}
{"- Use the original request context to understand what to create" if is_revision else ""}
- Be professional and create high-quality work
- Focus on delivering actual content, not questions

Please respond with your work for this {self.workflow_context["workflow_type"]} task.
"""
    
    def _get_recent_work_summary(self) -> str:
        """Extract a summary of recent work completed from conversation history."""
        history = self.workflow_state.get("conversation_history", [])
        
        # Look through history in reverse to find the most recent substantial work from specialist agents
        for msg in reversed(history[-10:]):  # Check last 10 messages
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            # Look for responses from specialist agents (not manager) with substantial content
            if len(content) > 200:  # Substantial work
                # Check if this is from a specialist agent
                for spec_agent in self.specialist_agents:
                    agent_name = spec_agent.name
                    if agent_name.lower() in content.lower() and ":" in content:
                        # Extract what type of work was done
                        if "script" in content.lower():
                            return f"{agent_name} completed a script (ready for review)"
                        elif "audio" in content.lower() or "music" in content.lower():
                            return f"{agent_name} provided audio recommendations"
                        else:
                            return f"{agent_name} completed work (ready for review)"
        
        # Fallback - check if any agents have been mentioned in recent responses
        for msg in reversed(history[-5:]):
            content = msg.get("content", "")
            for spec_agent in self.specialist_agents:
                agent_name = spec_agent.name
                if agent_name in content and len(content) > 50:
                    return f"Recent activity involving {agent_name}"
        
        return "No recent specialist work found" 