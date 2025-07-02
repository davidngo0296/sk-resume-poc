"""
Banner-specific Group Chat Manager for banner design workflows.
Extends the base MultiAgentGroupChatManager with banner design workflow intelligence.
"""

from typing import Optional, List, Dict, Any
from .group_chat_manager import MultiAgentGroupChatManager
import os


class BannerGroupChatManager(MultiAgentGroupChatManager):
    """
    Banner-specific group chat manager that extends the base class with 
    banner design workflow intelligence and delegation patterns.
    """
    
    def __init__(self, manager_agent, specialist_agents):
        """Initialize with banner design workflow configuration."""
        super().__init__(manager_agent, specialist_agents)
        
        # Set up banner design workflow context automatically
        self._setup_banner_design_workflow()
    
    def _setup_banner_design_workflow(self):
        """Configure the workflow for banner design."""
        self.set_workflow_context(
            workflow_type="Social Banner Design",
            workflow_description="Create engaging social media banners with AI-powered visual content evaluation",
            agent_roles={
                "GraphicDesigner": "Creates banner designs by selecting appropriate images from available resources",
                "GraphicEvaluator": "Evaluates banner designs using OpenAI Vision API to analyze actual image content for cat detection"
            },
            workflow_config={
                "requires_approval_after": [],  # Fully automated workflow - no approval required
                "auto_proceed_agents": ["GraphicDesigner", "GraphicEvaluator"],  # Both agents auto-proceed
                "workflow_sequence": ["GraphicDesigner", "GraphicEvaluator"],  # Define the typical flow
                "auto_start": True  # Automatically start without asking for banner description
            }
        )
    
    def get_workflow_display_info(self) -> Dict[str, Any]:
        """Get display information for the UI."""
        return {
            "title": "🎨 Social Banner Design Multi-Agent Team",
            "subtitle": "Powered by Semantic Kernel Group Chat Orchestration",
            "welcome_title": "Welcome to Banner Design Studio",
            "new_session_action": "Start new banner design project",
            "new_session_prompt": "Describe the banner you want to create (purpose, style, audience, etc.)",
            "session_starting_message": "Starting new banner design session...",
            "goodbye_message": "Thank you for using Banner Design Multi-Agent Team! 🎨"
        }
    
    def get_agent_colors(self) -> Dict[str, str]:
        """Get color mapping for agents in the UI."""
        return {
            "Manager": "cyan",
            "GraphicDesigner": "green", 
            "GraphicEvaluator": "magenta"
        }
    
    def is_workflow_complete(self, conversation_history: List[Dict]) -> bool:
        """
        Check if the banner design workflow is complete.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            bool: True if workflow appears complete
        """
        if not conversation_history or len(conversation_history) < 3:
            return False
        
        # Look for GraphicEvaluator providing approval
        for msg in reversed(conversation_history[-5:]):  # Check last 5 messages
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            # Check if GraphicEvaluator approved a banner
            if ("graphicevaluator" in role.lower() or "GraphicEvaluator:" in content):
                if "APPROVED" in content and "✅" in content:
                    return True
        
        return False
    
    def _detect_delegation(self, manager_response: str) -> Optional[str]:
        """
        Use AI to intelligently detect which agent should be delegated to based on context.
        This replaces hardcoded keyword matching with intelligent prompt-based analysis.
        """
        try:
            import openai
            
            # Get OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                # Fall back to hardcoded logic if no API key
                return self._detect_delegation_fallback(manager_response)
            
            # Get conversation context
            conversation_history = self.workflow_state.get("conversation_history", [])
            workflow_sequence = self.workflow_config.get("workflow_sequence", [])
            
            # Build available agents list
            available_agents = [agent.name for agent in self.specialist_agents]
            
            # Get recent conversation context (last 5 messages)
            recent_context = ""
            for msg in conversation_history[-5:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]  # Truncate for context
                recent_context += f"{role}: {content}\n"
            
            # Create AI prompt for delegation detection
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster model for workflow decisions
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a workflow coordination AI for a banner design process. Analyze the manager's response and determine which specialist agent (if any) should handle the next task.

WORKFLOW CONTEXT:
- Available agents: {', '.join(available_agents)}
- Workflow sequence: {' → '.join(workflow_sequence)}
- Recent conversation context:
{recent_context}

MANAGER'S RESPONSE:
"{manager_response}"

BANNER WORKFLOW RULES:
1. If manager mentions coordinating with GraphicDesigner → return "GraphicDesigner"
2. If manager mentions coordinating with GraphicEvaluator → return "GraphicEvaluator"  
3. If GraphicDesigner just provided a banner selection → return "GraphicEvaluator"
4. If GraphicEvaluator rejected a banner → return "GraphicDesigner"
5. If GraphicEvaluator approved a banner → return "NONE" (workflow complete)
6. If manager says no delegation needed → return "NONE"
7. For initial banner requests → return "GraphicDesigner"

RESPONSE FORMAT:
Return only the agent name ("GraphicDesigner", "GraphicEvaluator") or "NONE" if no delegation is needed.
Do not include any explanation or additional text."""
                    }
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            if result:
                result = result.strip()
            else:
                result = ""
            
            # Validate result
            if result == "NONE":
                return None
            elif result in available_agents:
                print(f"DEBUG: AI Delegation Detection → {result}")
                return result
            else:
                print(f"DEBUG: AI returned invalid agent '{result}', falling back to hardcoded logic")
                return self._detect_delegation_fallback(manager_response)
                
        except Exception as e:
            print(f"DEBUG: AI delegation detection failed: {e}, falling back to hardcoded logic")
            return self._detect_delegation_fallback(manager_response)
    
    def _detect_delegation_fallback(self, manager_response: str) -> Optional[str]:
        """
        Fallback hardcoded delegation detection logic.
        """
        response_lower = manager_response.lower()
        
        # First check for negative delegation contexts
        negative_contexts = [
            "will not delegate", "will not coordinate", "will not assign", 
            "no delegation", "not assign", "will not pass",
            "cannot delegate", "unable to delegate"
        ]
        
        for negative_context in negative_contexts:
            if negative_context in response_lower:
                return None
        
        # Find agents mentioned in the manager response
        mentioned_agents = []
        for spec_agent in self.specialist_agents:
            agent_name = spec_agent.name
            if agent_name.lower() in response_lower:
                mentioned_agents.append(agent_name)
        
        conversation_history = self.workflow_state.get("conversation_history", [])
        workflow_sequence = self.workflow_config.get("workflow_sequence", [])
        
        # Banner-specific workflow logic
        if mentioned_agents and workflow_sequence:
            # Find the most recent agent that provided substantial work
            last_working_agent = None
            last_evaluator_decision = None
            
            for msg in reversed(conversation_history[-5:]):  # Check last 5 messages
                content = msg.get("content", "")
                role = msg.get("role", "")
                
                # Check for GraphicEvaluator decisions
                if ("graphicevaluator" in role.lower() or "GraphicEvaluator:" in content):
                    if "REJECTED" in content and "❌" in content:
                        last_evaluator_decision = "rejected"
                        last_working_agent = "GraphicDesigner"  # Need to try again
                        break
                    elif "APPROVED" in content and "✅" in content:
                        last_evaluator_decision = "approved"
                        break
                
                # Check for GraphicDesigner work
                if len(content) > 100:  # Substantial work
                    for agent_name in mentioned_agents:
                        if agent_name.lower() in content.lower() and ":" in content:
                            last_working_agent = agent_name
                            break
            
            # Handle workflow sequence based on context
            if last_evaluator_decision == "rejected" and "GraphicDesigner" in mentioned_agents:
                print(f"DEBUG: FALLBACK BANNER WORKFLOW - Evaluator rejected, delegating to GraphicDesigner for new selection")
                return "GraphicDesigner"
            
            elif last_working_agent == "GraphicDesigner" and "GraphicEvaluator" in mentioned_agents:
                print(f"DEBUG: FALLBACK BANNER WORKFLOW - Designer provided banner, delegating to GraphicEvaluator for evaluation")
                return "GraphicEvaluator"
            
            # Check for initial delegation (user describing banner requirements)
            if not last_working_agent and "GraphicDesigner" in mentioned_agents:
                print(f"DEBUG: FALLBACK BANNER WORKFLOW - Initial request, delegating to GraphicDesigner")
                return "GraphicDesigner"
        
        # Fall back to base class delegation detection
        return super()._detect_delegation(manager_response)
    
    def _detect_user_feedback_type(self, user_message: str) -> str:
        """
        Use AI to detect the type of user feedback instead of hardcoded keywords.
        
        Returns:
            str: "approval", "revision", or "neutral"
        """
        try:
            import openai
            
            # Get OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return self._detect_user_feedback_type_fallback(user_message)
            
            # Create AI prompt for feedback detection
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this user message and determine what type of feedback they're providing about completed work.

USER MESSAGE:
"{user_message}"

INSTRUCTIONS:
Respond with exactly one word:
- "approval" - if user approves, likes, or wants to proceed (e.g., "yes", "good", "approve", "looks great", "proceed", "next")
- "revision" - if user wants changes or improvements (e.g., "no", "change", "revise", "different", "try again")
- "neutral" - if it's not clear feedback or a new request

Consider the overall intent and tone, not just specific keywords.

Respond with only one word: approval, revision, or neutral."""
                    }
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            if result:
                result = result.strip().lower()
                if result in ["approval", "revision", "neutral"]:
                    return result
            
            return self._detect_user_feedback_type_fallback(user_message)
                
        except Exception as e:
            print(f"DEBUG: AI feedback detection failed: {e}, falling back")
            return self._detect_user_feedback_type_fallback(user_message)
    
    def _detect_user_feedback_type_fallback(self, user_message: str) -> str:
        """Fallback hardcoded feedback type detection."""
        user_message_lower = user_message.lower()
        
        # Check user feedback type with banner-specific keywords
        approval_keywords = ["approve", "yes", "looks good", "accept", "proceed", "next"]
        revision_keywords = ["reject", "no", "revise", "change", "different", "try again"]
        
        if any(keyword in user_message_lower for keyword in approval_keywords):
            return "approval"
        elif any(keyword in user_message_lower for keyword in revision_keywords):
            return "revision"
        else:
            return "neutral"
    
    def _prepare_manager_context(self, prompt: str, conversation_history: List[Dict], original_request: str, previous_work: str) -> str:
        """Prepare banner-specific context for the manager agent."""
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
        is_user_feedback = len(conversation_history) > 1 and self._detect_user_feedback_type(prompt) != "neutral"
        
        workflow_sequence = self.workflow_config.get("workflow_sequence", [])
        workflow_info = f"Workflow sequence: {' → '.join(workflow_sequence)}" if workflow_sequence else ""
        
        # Banner-specific context
        banner_status = self._get_banner_evaluation_status()
        
        return f"""
You are coordinating a {self.workflow_context["workflow_type"]} workflow.

Workflow Description: {self.workflow_context["workflow_description"]}
{workflow_info}

Available specialist agents:
{agent_list}

BANNER DESIGN WORKFLOW CONTEXT:
- The GraphicDesigner selects banner images from available resources (mock_banner folder)  
- The GraphicEvaluator uses OpenAI Vision API to analyze actual image content and detect cats
- The evaluation is based on AI-generated text descriptions of the visual content, not filenames
- You coordinate between them and present final approved designs to the user

CONVERSATION CONTEXT:
- Original user request: {original_request if original_request else "Not specified"}
- Total messages: {len(conversation_history)}
- Recent work completed: {recent_work_summary}
- Banner evaluation status: {banner_status}

CURRENT SITUATION:
- User input: "{prompt}"

DELEGATION RULES FOR BANNER DESIGN:
1. INITIAL REQUEST: When user describes banner needs → delegate to GraphicDesigner
2. AFTER DESIGNER SELECTS: When GraphicDesigner presents a banner → delegate to GraphicEvaluator  
3. AFTER REJECTION: When GraphicEvaluator rejects → delegate to GraphicDesigner for new selection
4. AFTER APPROVAL: When GraphicEvaluator approves → present final result to user
5. USER REQUESTS CHANGES: Delegate to GraphicDesigner for new selection

Remember: Your role is to coordinate the workflow and communicate with the user. Always delegate creative work to specialists and ensure proper evaluation before presenting final results.
"""
    
    def _prepare_specialist_context(self, agent, prompt: str, conversation_history: List[Dict], original_request: str, previous_work: str) -> str:
        """Prepare banner-specific context for specialist agents."""
        agent_name = getattr(agent, 'name', 'Unknown')
        
        # Get recent evaluation results for context
        recent_evaluation = self._get_recent_evaluation_result()
        
        base_context = f"""
You are working on a {self.workflow_context["workflow_type"]} project.

Project Description: {self.workflow_context["workflow_description"]}
Your Role: {self.workflow_context["agent_roles"].get(agent_name, "Specialist agent")}

Original User Request: {original_request if original_request else "Not specified"}
Current Task: {prompt}

Recent Work Summary: {previous_work}
"""
        
        # Add agent-specific context
        if agent_name == "GraphicDesigner":
            if recent_evaluation and "rejected" in recent_evaluation.lower():
                base_context += f"""

IMPORTANT: The previous banner selection was rejected by the evaluator.
Evaluation feedback: {recent_evaluation}
Please select a different banner that better meets the evaluation criteria.
Remember to select an actual image file from the mock_banner directory.
"""
            else:
                base_context += """

Instructions: Select an appropriate banner image based on the user's requirements.
Use your select_random_banner() method to choose an actual image file.
Present your selection with professional explanation of your choice.
"""
        
        elif agent_name == "GraphicEvaluator":
            base_context += """

Instructions: Evaluate the banner design presented by the GraphicDesigner.
Use OpenAI Vision API to analyze the actual image content and generate detailed descriptions.
Analyze the AI-generated text descriptions to detect cats or feline animals.
Provide clear feedback with specific reasoning including confidence levels and AI insights.
Focus on the AI's visual analysis results, completely ignoring filenames.
"""
        
        return base_context
    
    def _get_recent_work_summary(self) -> str:
        """Get an AI-generated summary of recent work completed in the conversation."""
        conversation_history = self.workflow_state.get("conversation_history", [])
        if not conversation_history:
            return "No previous work"
        
        try:
            import openai
            
            # Get OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return self._get_recent_work_summary_fallback()
            
            # Get recent conversation context (last 5 messages)
            recent_context = ""
            for msg in conversation_history[-5:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:300]  # Truncate for context
                recent_context += f"{role}: {content}\n"
            
            # Create AI prompt for work summary
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze the recent conversation and provide a brief summary of work completed in the banner design workflow.

RECENT CONVERSATION:
{recent_context}

INSTRUCTIONS:
- Focus on what GraphicDesigner and GraphicEvaluator have accomplished
- Note if banners were selected, evaluated, approved, or rejected
- Keep summary under 50 words
- Use format: "Agent did action → Agent did action" 
- If no work done yet, say "Initial request received"

EXAMPLES:
- "GraphicDesigner selected a banner → GraphicEvaluator approved the banner"
- "GraphicDesigner selected a banner → GraphicEvaluator rejected the banner"
- "Initial request received"

Provide only the summary, no additional text."""
                    }
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            if result:
                return result.strip()
            else:
                return self._get_recent_work_summary_fallback()
                
        except Exception as e:
            print(f"DEBUG: AI work summary failed: {e}, falling back")
            return self._get_recent_work_summary_fallback()
    
    def _get_recent_work_summary_fallback(self) -> str:
        """Fallback hardcoded work summary analysis."""
        conversation_history = self.workflow_state.get("conversation_history", [])
        if not conversation_history:
            return "No previous work"
        
        # Look for recent agent activities
        recent_activities = []
        for msg in reversed(conversation_history[-3:]):
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            if "GraphicDesigner:" in content and len(content) > 50:
                if any(word in content.lower() for word in ["selected", "banner", "image", ".jpg", ".png"]):
                    recent_activities.append("GraphicDesigner selected a banner")
            
            elif "GraphicEvaluator:" in content and len(content) > 50:
                if "APPROVED" in content:
                    recent_activities.append("GraphicEvaluator approved the banner")
                elif "REJECTED" in content:
                    recent_activities.append("GraphicEvaluator rejected the banner")
        
        return " → ".join(reversed(recent_activities)) if recent_activities else "Initial request received"
    
    def _get_banner_evaluation_status(self) -> str:
        """Get AI-analyzed current status of banner evaluation."""
        conversation_history = self.workflow_state.get("conversation_history", [])
        
        try:
            import openai
            
            # Get OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return self._get_banner_evaluation_status_fallback()
            
            # Get recent evaluator messages
            evaluator_context = ""
            for msg in reversed(conversation_history[-5:]):
                content = msg.get("content", "")
                if "GraphicEvaluator:" in content:
                    evaluator_context += f"{content[:300]}\n"
            
            if not evaluator_context:
                return "No evaluation yet"
            
            # Create AI prompt for evaluation status
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze the GraphicEvaluator's recent responses and determine the current evaluation status.

EVALUATOR RESPONSES:
{evaluator_context}

INSTRUCTIONS:
Respond with exactly one of these options:
- "Last banner was APPROVED" (if evaluator approved a banner)
- "Last banner was REJECTED" (if evaluator rejected a banner)
- "No evaluation yet" (if no clear evaluation found)

Look for approval indicators like ✅, "APPROVED", positive feedback.
Look for rejection indicators like ❌, "REJECTED", negative feedback.

Provide only the status, no additional text."""
                    }
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            if result:
                return result.strip()
            else:
                return self._get_banner_evaluation_status_fallback()
                
        except Exception as e:
            print(f"DEBUG: AI evaluation status failed: {e}, falling back")
            return self._get_banner_evaluation_status_fallback()
    
    def _get_banner_evaluation_status_fallback(self) -> str:
        """Fallback hardcoded evaluation status analysis."""
        conversation_history = self.workflow_state.get("conversation_history", [])
        
        for msg in reversed(conversation_history[-3:]):
            content = msg.get("content", "")
            if "GraphicEvaluator:" in content:
                if "APPROVED" in content and "✅" in content:
                    return "Last banner was APPROVED"
                elif "REJECTED" in content and "❌" in content:
                    return "Last banner was REJECTED"
        
        return "No evaluation yet"
    
    def _get_recent_evaluation_result(self) -> str:
        """Get the most recent evaluation result for context."""
        conversation_history = self.workflow_state.get("conversation_history", [])
        
        for msg in reversed(conversation_history[-3:]):
            content = msg.get("content", "")
            if "GraphicEvaluator:" in content and ("APPROVED" in content or "REJECTED" in content):
                return content
        
        return ""
    
    async def _controlled_agent_workflow_streaming(self, user_message: str):
        """
        Override base workflow to implement banner-specific auto-proceed behavior.
        This implements the fully automated banner workflow without user approval steps.
        """
        print(f"DEBUG: Banner workflow _controlled_agent_workflow_streaming called with: {user_message}")
        
        current_phase = self._get_current_phase()
        print(f"DEBUG: Banner workflow current_phase: {current_phase}")
        
        # For banner workflow, we want it to auto-start without asking for description
        if current_phase == "initial":
            # Auto-start the banner workflow
            print("DEBUG: Banner workflow auto-starting without user input")
            
            # Manager introduces the workflow and delegates to GraphicDesigner
            manager_intro = "I'll coordinate the creation of a social media banner. Let me start by having the GraphicDesigner select an appropriate image from our available options."
            yield "Manager", manager_intro
            
            # Automatically invoke GraphicDesigner
            graphic_designer = self._get_agent_by_name("GraphicDesigner")
            if graphic_designer:
                try:
                    print("DEBUG: Auto-invoking GraphicDesigner to select banner image")
                    # Directly call the banner selection function to get real results
                    selected_banner = graphic_designer.select_random_banner()
                    designer_response = f"I have selected the banner image: **{selected_banner}**\n\n**Professional Explanation:**\nThis image was selected using our systematic banner selection process to ensure optimal visual appeal and engagement potential for social media platforms. The chosen banner will now undergo comprehensive AI-powered evaluation to verify it meets our content requirements."
                    yield "GraphicDesigner", designer_response
                    
                    # Automatically proceed to GraphicEvaluator
                    print("DEBUG: Auto-proceeding to GraphicEvaluator for image evaluation")
                    graphic_evaluator = self._get_agent_by_name("GraphicEvaluator")
                    if graphic_evaluator:
                        # Extract banner filename from the selected banner string
                        banner_filename = selected_banner.replace("Selected banner image: ", "").strip()
                        print(f"DEBUG: Evaluating banner file: {banner_filename}")
                        
                        # Directly call the evaluation function to get real results
                        evaluation_result = await graphic_evaluator.evaluate_banner_visual_content(banner_filename)
                        yield "GraphicEvaluator", evaluation_result
                        
                        # Check evaluation result and auto-proceed accordingly
                        if "APPROVED" in evaluation_result and "✅" in evaluation_result:
                            print("DEBUG: Banner approved - presenting final result")
                            final_manager_response = "Excellent! The GraphicEvaluator has approved the selected banner. The banner design process is now complete and ready for use."
                            yield "Manager", final_manager_response
                        elif "REJECTED" in evaluation_result and "❌" in evaluation_result:
                            print("DEBUG: Banner rejected - attempting new selection")
                            retry_manager_response = "The GraphicEvaluator has rejected the current selection. Let me coordinate with the GraphicDesigner to select a different banner image."
                            yield "Manager", retry_manager_response
                            
                            # Try again with GraphicDesigner
                            retry_selected_banner = graphic_designer.select_random_banner()
                            retry_designer_response = f"I have selected a different banner image: **{retry_selected_banner}**\n\n**Revised Selection:**\nBased on the evaluation feedback, I have chosen a new banner to better meet the content requirements. This alternative selection will be evaluated to ensure it aligns with our cat-themed criteria."
                            yield "GraphicDesigner", retry_designer_response
                            
                            # Evaluate the new selection
                            retry_banner_filename = retry_selected_banner.replace("Selected banner image: ", "").strip()
                            print(f"DEBUG: Evaluating retry banner file: {retry_banner_filename}")
                            retry_evaluation_result = await graphic_evaluator.evaluate_banner_visual_content(retry_banner_filename)
                            yield "GraphicEvaluator", retry_evaluation_result
                            
                            # Final result after retry
                            if "APPROVED" in retry_evaluation_result and "✅" in retry_evaluation_result:
                                final_approved_response = "Perfect! The GraphicEvaluator has approved the second banner selection. The banner design workflow is now complete."
                                yield "Manager", final_approved_response
                            else:
                                final_partial_response = "The workflow has completed two attempts. Please review the evaluation feedback and consider manual selection if needed."
                                yield "Manager", final_partial_response
                        
                except Exception as e:
                    print(f"DEBUG: Error in banner auto-workflow: {e}")
                    yield "Manager", "Error: Unable to complete the automated banner workflow"
            
        elif current_phase == "working":
            # Handle user feedback during the workflow (like "go on", "continue", etc.)
            if any(keyword in user_message.lower() for keyword in ["go on", "continue", "proceed", "next"]):
                print("DEBUG: User requested to continue banner workflow")
                
                # Check current workflow state and proceed accordingly
                conversation_history = self.workflow_state.get("conversation_history", [])
                last_designer_work = None
                last_evaluator_work = None
                
                # Find the most recent work from each agent
                for msg in reversed(conversation_history[-10:]):
                    content = msg.get("content", "")
                    if "GraphicDesigner:" in content and not last_designer_work:
                        last_designer_work = content
                    elif "GraphicEvaluator:" in content and not last_evaluator_work:
                        last_evaluator_work = content
                
                if last_designer_work and not last_evaluator_work:
                    # Designer has worked, need evaluator
                    print("DEBUG: Designer has provided work, proceeding to evaluator")
                    manager_response = "Now let me coordinate with the GraphicEvaluator to assess the selected banner."
                    yield "Manager", manager_response
                    
                    graphic_evaluator = self._get_agent_by_name("GraphicEvaluator")
                    if graphic_evaluator:
                        evaluator_response = await self._get_agent_response(graphic_evaluator, "Evaluate the banner image selected by the GraphicDesigner")
                        yield "GraphicEvaluator", evaluator_response
                        
                        # Auto-proceed based on evaluation
                        if "APPROVED" in evaluator_response and "✅" in evaluator_response:
                            final_response = "The banner has been approved! The design workflow is complete."
                            yield "Manager", final_response
                        elif "REJECTED" in evaluator_response and "❌" in evaluator_response:
                            reject_response = "The evaluator has provided feedback. Let me coordinate a revision with the GraphicDesigner."
                            yield "Manager", reject_response
                
                elif last_evaluator_work:
                    # Evaluator has provided feedback, determine next steps
                    if "REJECTED" in last_evaluator_work and "❌" in last_evaluator_work:
                        print("DEBUG: Previous evaluation rejected, trying new selection")
                        manager_response = "Based on the evaluation feedback, I'll coordinate with the GraphicDesigner to select a different banner."
                        yield "Manager", manager_response
                        
                        graphic_designer = self._get_agent_by_name("GraphicDesigner")
                        if graphic_designer:
                            designer_response = await self._get_agent_response(graphic_designer, "Select a different banner image based on the evaluation feedback")
                            yield "GraphicDesigner", designer_response
                    else:
                        manager_response = "The banner workflow appears to be complete. The selected banner has been processed."
                        yield "Manager", manager_response
            else:
                # Fall back to base class behavior for other inputs
                async for agent_name, response in super()._controlled_agent_workflow_streaming(user_message):
                    yield agent_name, response
        else:
            # Fall back to base class behavior for other phases
            async for agent_name, response in super()._controlled_agent_workflow_streaming(user_message):
                yield agent_name, response 