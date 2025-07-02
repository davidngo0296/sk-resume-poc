"""
Demo script for Video Script Multi-Agent Team POC
This demonstrates the concept without requiring full Semantic Kernel dependencies.
"""

import asyncio
import os
from typing import Dict, Any
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Load environment variables
load_dotenv("api_keys_config.env")

class MockVideoAgent:
    """Mock agent for demonstration purposes."""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process a message and return a response based on agent role."""
        
        if self.name == "Manager":
            if "script" in message.lower() and "video" in message.lower():
                return f"""Hello! I'm the Manager agent. I understand you want to create a video script.

I'll coordinate with my team to help you create an engaging script. Let me delegate this to our Script Writer agent who specializes in creative video content.

Script Writer, please create a video script based on this request: "{message}" """
            
            elif "approve" in message.lower() or "yes" in message.lower():
                return f"""Perfect! I can see you're happy with the script. Now let me bring in our Audio Selector agent to suggest appropriate music and sound effects for your video.

Audio Selector, please provide audio recommendations for the approved script."""
            
            elif "pause" in message.lower():
                return "I understand you need to pause our conversation. I'll save our progress and you can resume anytime. All our work will be preserved."
            
            else:
                return f"As the Manager, I'm here to coordinate our video script creation process. How can I help you today?"
        
        elif self.name == "ScriptWriter":
            return f"""As the Script Writer, I've created this video script for you:

**VIDEO SCRIPT**

[SCENE 1 - OPENING]
🎬 *Fade in from black*

NARRATOR (Voice-over): "{message.replace('create', 'Welcome to').replace('script', 'story').replace('video', 'journey')}"

[Visual: Engaging opening sequence with dynamic text overlay]

[SCENE 2 - MAIN CONTENT]  
🎥 *Cut to main presentation*

NARRATOR: "In today's fast-paced world, we're going to explore something truly exciting..."

[Visual: Relevant B-roll footage supporting the narrative]

[SCENE 3 - CALL TO ACTION]
🎯 *Strong closing sequence*

NARRATOR: "Don't forget to like, subscribe, and share your thoughts in the comments below!"

[Visual: Subscribe button animation and social media icons]

**END OF SCRIPT**

Manager, please present this script to the user for approval."""
        
        elif self.name == "AudioSelector":
            return f"""As the Audio Selector, here are my audio recommendations for your video:

**🎵 AUDIO RECOMMENDATIONS**

**Background Music:**
- Genre: Upbeat Electronic/Corporate
- Tempo: 120-128 BPM  
- Style: Motivational and energetic
- Recommended Track: "Inspiring Success" (Royalty-free)
- Source: Epidemic Sound or YouTube Audio Library

**Sound Effects:**
- Intro: Swoosh sound for text animations (0:00-0:03)
- Transition: Subtle "ding" for scene changes (0:15, 0:30)
- Outro: Success chime for call-to-action (0:45-0:50)

**Audio Mixing Suggestions:**
- Background music volume: 20-30% during narration
- Fade in music over 2 seconds at start
- Duck music volume during voice-over
- Fade out over 3 seconds at end

**Licensing Information:**
- All recommended tracks are royalty-free
- Attribution may be required (check source terms)
- Commercial use approved for most sources

**Alternative Options:**
- Classical/Orchestral for serious content
- Jazz for sophisticated/educational videos  
- Ambient for calm, informative content

Your video script is now complete with full audio recommendations! 🎉"""
        
        return f"I'm {self.name} ({self.role}), ready to help!"

class MockVideoScriptTeam:
    """Mock multi-agent team for demonstration."""
    
    def __init__(self):
        self.console = Console()
        self.agents = {
            "Manager": MockVideoAgent("Manager", "Workflow Coordinator"),
            "ScriptWriter": MockVideoAgent("ScriptWriter", "Creative Writer"),
            "AudioSelector": MockVideoAgent("AudioSelector", "Audio Specialist")
        }
        # Use new generic workflow state structure - no task-specific fields
        self.workflow_state = {
            "conversation_history": []
        }
        
        # Add workflow context like the real application
        self.workflow_context = {
            "workflow_type": "Video Script Creation",
            "workflow_description": "Create professional video scripts with audio recommendations",
            "agent_roles": {
                "ScriptWriter": "Creates engaging and professional video scripts based on user requirements and feedback",
                "AudioSelector": "Provides audio recommendations and music suggestions that complement video scripts"
            },
            "workflow_config": {
                "requires_approval_after": ["ScriptWriter"],
                "auto_proceed_agents": [],
                "workflow_sequence": ["ScriptWriter", "AudioSelector"]
            }
        }
    
    def display_banner(self):
        """Display the demo banner."""
        banner = Text("🎬 Video Script Multi-Agent Team - DEMO MODE", style="bold blue")
        subtitle = Text("Simulating Semantic Kernel Group Chat Orchestration", style="italic")
        
        self.console.print(Panel.fit(
            f"{banner}\n{subtitle}",
            title="Demo",
            border_style="blue"
        ))
        
        self.console.print(Panel(
            "This demo simulates the multi-agent workflow without requiring OpenAI API keys.\n"
            "In the full implementation, each agent would be powered by Semantic Kernel and LLMs.",
            title="Note",
            border_style="yellow"
        ))
    
    def _get_current_phase(self) -> str:
        """Extract current workflow phase from conversation history - matching real implementation."""
        history = self.workflow_state.get("conversation_history", [])
        
        if not history:
            return "initial"
        
        # Get the last few messages to determine current state
        recent_messages = history[-5:] if len(history) >= 5 else history
        recent_content = " ".join([msg.get("content", "") for msg in recent_messages]).lower()
        
        # Check for approval patterns
        if any(word in recent_content for word in ["approve", "approved", "looks good", "accept"]):
            return "post_approval"
        
        # Check for revision/feedback patterns  
        if any(word in recent_content for word in ["reject", "revise", "change", "modify", "different", "feedback"]):
            return "awaiting_approval"
        
        # Check if work was delivered (look for longer responses from non-manager agents)
        for msg in reversed(recent_messages):
            content = msg.get("content", "")
            role = msg.get("role", "")
            # Look for responses from non-manager agents with substantial content
            if role != "Manager" and role != "user" and len(content) > 100:
                return "awaiting_approval"
        
        # Default to initial work phase
        return "working"
    
    async def process_conversation(self, user_message: str):
        """Process user message through the mock agent workflow."""
        
        # Add user message to conversation history
        self.workflow_state["conversation_history"].append({
            "role": "user", 
            "content": user_message
        })
        
        # Use generic phase detection instead of hardcoded workflow
        current_phase = self._get_current_phase()
        
        # Generic coordination logic - let Manager handle most coordination
        if current_phase == "initial":
            # Manager starts and coordinates
            current_agent = "Manager"
        elif current_phase == "awaiting_approval":
            if any(keyword in user_message.lower() for keyword in ["approve", "yes", "looks good", "accept", "good"]):
                # User approved - Manager coordinates next steps
                current_agent = "Manager"
            else:
                # User wants changes - Manager coordinates
                current_agent = "Manager"
        else:
            # Default to Manager for coordination
            current_agent = "Manager"
        
        # Get response from selected agent
        agent = self.agents[current_agent]
        response = await agent.process_message(user_message, self.workflow_state)
        
        # Add agent response to conversation history
        self.workflow_state["conversation_history"].append({
            "role": current_agent,
            "content": response
        })
        
        return current_agent, response
    
    def get_workflow_status(self) -> str:
        """Get current workflow status using generic phase detection."""
        phase = self._get_current_phase()
        phase_descriptions = {
            "initial": "🟡 Ready to start",
            "working": "🔄 Agents working",
            "awaiting_approval": "⏳ Awaiting user feedback", 
            "post_approval": "✅ Approved, continuing workflow",
            "completed": "✅ Complete"
        }
        return phase_descriptions.get(phase, "🔄 Processing")

async def main():
    """Main demo function."""
    team = MockVideoScriptTeam()
    team.display_banner()
    
    console = Console()
    
    # Check for API keys and show appropriate message
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("AZURE_OPENAI_ENDPOINT"):
        console.print(Panel(
            "No API keys detected in api_keys_config.env\n"
            "This demo will run in mock mode.\n"
            "For the full implementation with real AI agents, please configure your API keys.",
            title="API Configuration",
            border_style="cyan"
        ))
    else:
        console.print(Panel(
            "API keys found! In the full implementation, these would power real AI agents.\n"
            "This demo shows the workflow structure and conversation management.",
            title="API Configuration",
            border_style="green"
        ))
    
    console.print("\n💡 **Try these demo commands:**")
    console.print("1. 'Create a video script about AI technology'")
    console.print("2. 'Yes, I approve the script' (after script is created)")
    console.print("3. 'pause' to simulate conversation pausing")
    console.print("4. 'quit' to exit")
    
    while True:
        console.print(f"\n📊 **Status:** {team.get_workflow_status()}")
        
        user_input = input("\n🎤 Your message: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            console.print("\n👋 Thank you for trying the Video Script Multi-Agent Team demo!")
            break
        
        if user_input.lower() == 'pause':
            console.print(Panel(
                "Conversation paused! In the full implementation:\n"
                "• All conversation state would be serialized to JSON\n"
                "• You could resume exactly where you left off\n"
                "• Agent context and workflow phase would be preserved",
                title="Pause Simulation",
                border_style="yellow"
            ))
            continue
        
        # Process the message
        console.print(f"\n🔄 Processing message through agent team...")
        
        try:
            agent_name, response = await team.process_conversation(user_input)
            
            console.print(f"\n🤖 **{agent_name} Agent:**")
            console.print(Panel(response, border_style="green"))
            
            # Show workflow progression using generic detection
            current_phase = team._get_current_phase()
            if "audio" in response.lower() and "recommendation" in response.lower():
                console.print(Panel(
                    "🎉 **Workflow Complete!**\n\n"
                    "In the full implementation, this conversation would be automatically saved.\n"
                    "You would have:\n"
                    "• A complete video script\n"
                    "• Professional audio recommendations\n"
                    "• Full conversation history for reference\n"
                    "• Ability to request revisions from any agent",
                    title="Success",
                    border_style="blue"
                ))
                
                restart = input("\n🔄 Start a new video script? (y/n): ").strip().lower()
                if restart == 'y':
                    team.workflow_state = {
                        "conversation_history": []
                    }
                    console.print("\n🆕 New conversation started!")
                
        except Exception as e:
            console.print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo terminated by user. Goodbye! 👋") 