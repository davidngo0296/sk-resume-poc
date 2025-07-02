"""
Main application for Multi-Agent Team POC using Semantic Kernel.
Generic CLI that works with any multi-agent workflow.
"""

import asyncio
import os
import sys
from typing import Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

from src.orchestration.script_group_chat_manager import ScriptGroupChatManager
from src.orchestration.banner_group_chat_manager import BannerGroupChatManager
from src.serialization.conversation_serializer import ConversationSerializer


class MultiAgentCLI:
    """
    Generic Command Line Interface for Multi-Agent Teams.
    Works with any workflow type through the workflow manager.
    """
    
    def __init__(self):
        self.console = Console()
        self.serializer = ConversationSerializer()
        self.current_conversation_id: Optional[str] = None
        self.workflow_manager = None  # Will be set after workflow selection
    
    def select_workflow_type(self) -> str:
        """Allow user to select which workflow type to use."""
        self.console.print("\n[bold blue]🚀 Multi-Agent Team POC[/bold blue]")
        self.console.print("[italic]Powered by Semantic Kernel Group Chat Orchestration[/italic]\n")
        
        table = Table(title="Available Workflows", show_header=True)
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Workflow", style="green")
        table.add_column("Description", style="white")
        
        table.add_row("1", "Video Script Creation", "Create professional video scripts with audio recommendations")
        table.add_row("2", "Social Banner Design", "Design engaging social media banners with professional evaluation")
        table.add_row("q", "Quit", "Exit the application")
        
        self.console.print(table)
        
        while True:
            choice = Prompt.ask("\nSelect a workflow type")
            
            if choice == "1":
                return "script"
            elif choice == "2":
                return "banner"
            elif choice.lower() in ["q", "quit"]:
                self.console.print("[bold blue]👋 Goodbye![/bold blue]")
                sys.exit(0)
            else:
                self.console.print("[red]Invalid option. Please select 1, 2, or q.[/red]")
    
    def initialize_workflow_manager(self, workflow_type: str):
        """Initialize the appropriate workflow manager based on type."""
        from src.agents.manager_agent import ManagerAgent
        
        manager_agent = ManagerAgent()
        
        if workflow_type == "script":
            from src.agents.script_writer_agent import ScriptWriterAgent
            from src.agents.audio_selector_agent import AudioSelectorAgent
            
            script_writer = ScriptWriterAgent()
            audio_selector = AudioSelectorAgent()
            specialist_agents = [script_writer, audio_selector]
            
            self.workflow_manager = ScriptGroupChatManager(manager_agent, specialist_agents)
            
        elif workflow_type == "banner":
            from src.agents.graphic_designer_agent import GraphicDesignerAgent
            from src.agents.graphic_evaluator_agent import GraphicEvaluatorAgent
            
            graphic_designer = GraphicDesignerAgent()
            graphic_evaluator = GraphicEvaluatorAgent()
            specialist_agents = [graphic_designer, graphic_evaluator]
            
            self.workflow_manager = BannerGroupChatManager(manager_agent, specialist_agents)
            
        else:
            raise ValueError(f"Unsupported workflow type: {workflow_type}")
    
    def display_banner(self):
        """Display the application banner using workflow-specific info."""
        if self.workflow_manager:
            display_info = self.workflow_manager.get_workflow_display_info()
            
            banner = Text(display_info["title"], style="bold blue")
            subtitle = Text(display_info["subtitle"], style="italic")
            
            self.console.print(Panel.fit(
                f"{banner}\n{subtitle}",
                title=display_info["welcome_title"],
                border_style="blue"
            ))
        else:
            # Generic banner when no workflow is selected
            self.console.print(Panel.fit(
                "🚀 Multi-Agent Team POC\nPowered by Semantic Kernel",
                title="Welcome",
                border_style="blue"
            ))
    
    def display_menu(self):
        """Display the main menu options using workflow-specific info."""
        if self.workflow_manager:
            display_info = self.workflow_manager.get_workflow_display_info()
            
            table = Table(title="Available Actions", show_header=False)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Description", style="white")
            
            table.add_row("1", display_info["new_session_action"])
            table.add_row("2", "Resume paused conversation")
            table.add_row("3", "List saved conversations")
            table.add_row("4", "View conversation summary")
            table.add_row("5", "Delete saved conversation")
            table.add_row("6", "Switch workflow type")
            table.add_row("q", "Quit")
            
            self.console.print(table)
        else:
            self.console.print("[red]No workflow selected. Please restart the application.[/red]")
    
    async def start_new_conversation(self):
        """Start a new conversation using workflow-specific prompts."""
        if not self.workflow_manager:
            self.console.print("[red]No workflow selected. Please restart the application.[/red]")
            return
            
        display_info = self.workflow_manager.get_workflow_display_info()
        
        self.console.print(f"\n[bold green]{display_info['session_starting_message']}[/bold green]")
        
        # Check if workflow supports auto-start
        workflow_config = self.workflow_manager.workflow_config if hasattr(self.workflow_manager, 'workflow_config') else {}
        auto_start = workflow_config.get("auto_start", False)
        
        if auto_start:
            # Auto-start workflow without user input
            user_prompt = "auto-start"  # Trigger word for automated workflow
            self.console.print("\n[dim]Starting automated workflow...[/dim]")
        else:
            # Get initial user prompt using workflow-specific text
            user_prompt = Prompt.ask(f"\n{display_info['new_session_prompt']}")
            
            if not user_prompt.strip():
                self.console.print("[red]No prompt provided. Returning to menu.[/red]")
                return
        
        # Reset workflow state to start fresh conversation
        self.workflow_manager.reset_workflow_state()
        
        # Generate conversation ID
        self.current_conversation_id = self.serializer.generate_conversation_id(user_prompt)
        
        self.console.print(f"\n[dim]Conversation ID: {self.current_conversation_id}[/dim]")
        
        try:
            # Start the group chat session
            await self.workflow_manager.start_session()
            
            # Process the initial request
            self.console.print("\n[bold blue]Agents are responding...[/bold blue]")
            async for agent_name, response in self.workflow_manager.process_user_message_streaming(user_prompt):
                agent_colors = self.workflow_manager.get_agent_colors()
                agent_color = agent_colors.get(agent_name, "blue")
                
                self.console.print(Panel(
                    response,
                    title=f"🤖 {agent_name} Agent",
                    border_style=agent_color,
                    title_align="left"
                ))
                
                await asyncio.sleep(0.5)
            
            # Continue the conversation
            await self.continue_conversation()
        
        except Exception as e:
            self.console.print(f"[red]Error during conversation: {e}[/red]")
        finally:
            await self.workflow_manager.stop_session()
    
    async def continue_conversation(self):
        """Continue an ongoing conversation with the agent team."""
        if not self.workflow_manager:
            self.console.print("[red]No workflow selected. Please restart the application.[/red]")
            return
            
        while True:
            self.console.print("\n" + "="*60)
            
            # Show current workflow state
            state = self.workflow_manager.get_workflow_state()
            conversation_history = state.get("conversation_history", [])
            message_count = len(conversation_history)
            self.console.print(f"[dim]Messages: {message_count}[/dim]")
            
            # Check if workflow is complete using workflow-specific logic
            if self.workflow_manager.is_workflow_complete(conversation_history):
                self.console.print("[green]🎉 Workflow appears complete![/green]")
                if Confirm.ask("Save this conversation?"):
                    self.save_current_conversation()
                break
            
            # Get user input
            user_input = Prompt.ask(
                "\nYour response (or 'pause' to save and exit, 'quit' to exit without saving)",
                default=""
            )
            
            if user_input.lower() in ['quit', 'q']:
                if Confirm.ask("Save conversation before quitting?"):
                    self.save_current_conversation()
                break
            
            elif user_input.lower() in ['pause', 'p']:
                self.save_current_conversation()
                self.console.print("[yellow]Conversation paused and saved. You can resume it later.[/yellow]")
                break
            
            elif not user_input.strip():
                continue
            
            try:
                await self._process_streaming_response(user_input)
            
            except Exception as e:
                self.console.print(f"[red]Error processing response: {e}[/red]")
    
    async def _process_streaming_response(self, user_input: str):
        """Process user input with streaming agent responses."""
        if not self.workflow_manager:
            self.console.print("[red]No workflow selected. Please restart the application.[/red]")
            return
            
        self.console.print("\n[bold blue]Agents are responding...[/bold blue]")
        
        try:
            async for agent_name, response in self.workflow_manager.process_user_message_streaming(user_input):
                # Create a panel for each agent response using workflow-specific colors
                agent_colors = self.workflow_manager.get_agent_colors()
                agent_color = agent_colors.get(agent_name, "blue")
                
                self.console.print(Panel(
                    response,
                    title=f"🤖 {agent_name} Agent",
                    border_style=agent_color,
                    title_align="left"
                ))
                
                await asyncio.sleep(0.5)
                
        except Exception as e:
            self.console.print(f"[red]Error in streaming response: {e}[/red]")
    
    def save_current_conversation(self):
        """Save the current conversation state."""
        if not self.current_conversation_id or not self.workflow_manager:
            return
        
        try:
            state_data = self.workflow_manager.pause_conversation()
            file_path = self.serializer.save_conversation(
                self.current_conversation_id,
                state_data
            )
            self.console.print(f"[green]Conversation saved to: {file_path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving conversation: {e}[/red]")
    
    async def resume_conversation(self):
        """Resume a paused conversation."""
        conversations = self.serializer.list_saved_conversations()
        
        if not conversations:
            self.console.print("[yellow]No saved conversations found.[/yellow]")
            return
        
        # Display available conversations
        table = Table(title="Saved Conversations")
        table.add_column("ID", style="cyan")
        table.add_column("Saved At", style="green")
        table.add_column("Status", style="yellow")
        
        for conv in conversations:
            table.add_row(
                conv["conversation_id"],
                conv["saved_at"],
                conv["status"]
            )
        
        self.console.print(table)
        
        # Select conversation to resume
        conv_id = Prompt.ask("\nEnter conversation ID to resume")
        
        # Load conversation
        conversation_data = self.serializer.load_conversation(conv_id)
        if not conversation_data:
            self.console.print(f"[red]Conversation '{conv_id}' not found.[/red]")
            return
        
        self.current_conversation_id = conv_id
        
        try:
            # Resume the conversation
            if not self.workflow_manager:
                self.console.print("[red]No workflow selected. Cannot resume conversation.[/red]")
                return
                
            with self.console.status("[bold green]Resuming conversation..."):
                # Extract workflow_state from the loaded conversation data
                workflow_state = conversation_data.get("workflow_state", {})
                resume_message = await self.workflow_manager.resume_conversation(workflow_state)
            
            self.console.print(f"[green]{resume_message}[/green]")
            
            # Continue the conversation
            await self.continue_conversation()
        
        except Exception as e:
            self.console.print(f"[red]Error resuming conversation: {e}[/red]")
        finally:
            if self.workflow_manager:
                await self.workflow_manager.stop_session()
    
    def list_conversations(self):
        """List all saved conversations."""
        conversations = self.serializer.list_saved_conversations()
        
        if not conversations:
            self.console.print("[yellow]No saved conversations found.[/yellow]")
            return
        
        table = Table(title="Saved Conversations")
        table.add_column("ID", style="cyan")
        table.add_column("Saved At", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("File Path", style="dim")
        
        for conv in conversations:
            table.add_row(
                conv["conversation_id"],
                conv["saved_at"],
                conv["status"],
                conv["file_path"]
            )
        
        self.console.print(table)
    
    def view_conversation_summary(self):
        """View summary of a saved conversation."""
        conversations = self.serializer.list_saved_conversations()
        
        if not conversations:
            self.console.print("[yellow]No saved conversations found.[/yellow]")
            return
        
        # Show available conversations
        for i, conv in enumerate(conversations, 1):
            self.console.print(f"{i}. {conv['conversation_id']} ({conv['status']})")
        
        conv_id = Prompt.ask("\nEnter conversation ID to view summary")
        
        summary = self.serializer.export_conversation_summary(conv_id)
        if summary:
            self.console.print(Panel(summary, title=f"Conversation Summary: {conv_id}"))
        else:
            self.console.print(f"[red]Conversation '{conv_id}' not found.[/red]")
    
    def delete_conversation(self):
        """Delete a saved conversation."""
        conversations = self.serializer.list_saved_conversations()
        
        if not conversations:
            self.console.print("[yellow]No saved conversations found.[/yellow]")
            return
        
        # Show available conversations
        for i, conv in enumerate(conversations, 1):
            self.console.print(f"{i}. {conv['conversation_id']} ({conv['status']})")
        
        conv_id = Prompt.ask("\nEnter conversation ID to delete")
        
        if Confirm.ask(f"Are you sure you want to delete conversation '{conv_id}'?"):
            if self.serializer.delete_conversation(conv_id):
                self.console.print(f"[green]Conversation '{conv_id}' deleted successfully.[/green]")
            else:
                self.console.print(f"[red]Failed to delete conversation '{conv_id}'.[/red]")
    
    async def run(self):
        """Main application loop."""
        # Check for required environment variables
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.console.print("[red]Error: No OpenAI API key or Azure OpenAI endpoint found.[/red]")
            self.console.print("[yellow]Please set your API credentials in the api_keys_config.env file.[/yellow]")
            return
        
        # Select workflow type first
        workflow_type = self.select_workflow_type()
        self.initialize_workflow_manager(workflow_type)
        
        self.display_banner()
        
        # Check if workflow supports auto-start
        workflow_config = {}
        if self.workflow_manager and hasattr(self.workflow_manager, 'workflow_config'):
            workflow_config = self.workflow_manager.workflow_config
        auto_start = workflow_config.get("auto_start", False)
        
        if auto_start:
            # Auto-start workflow immediately without showing menu
            self.console.print("\n[dim]Auto-starting workflow...[/dim]")
            try:
                await self.start_new_conversation()
                # After auto-start workflow completes, show normal menu
            except Exception as e:
                self.console.print(f"[red]Error in auto-start workflow: {e}[/red]")
        
        while True:
            self.display_menu()
            choice = Prompt.ask("\nSelect an option")
            
            try:
                if choice == "1":
                    await self.start_new_conversation()
                elif choice == "2":
                    await self.resume_conversation()
                elif choice == "3":
                    self.list_conversations()
                elif choice == "4":
                    self.view_conversation_summary()
                elif choice == "5":
                    self.delete_conversation()
                elif choice == "6":
                    # Switch workflow type
                    workflow_type = self.select_workflow_type()
                    self.initialize_workflow_manager(workflow_type)
                    self.display_banner()
                elif choice.lower() in ["q", "quit"]:
                    if self.workflow_manager:
                        display_info = self.workflow_manager.get_workflow_display_info()
                        self.console.print(f"[bold blue]{display_info['goodbye_message']}[/bold blue]")
                    else:
                        self.console.print("[bold blue]👋 Goodbye![/bold blue]")
                    break
                else:
                    self.console.print("[red]Invalid option. Please try again.[/red]")
            
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Operation cancelled.[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
            
            if choice != "q":
                Prompt.ask("\nPress Enter to continue")


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv("api_keys_config.env")
    
    # Create and run the CLI application
    app = MultiAgentCLI()
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0) 