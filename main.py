"""
Main application for Multi-Agent Team POC using Semantic Kernel GroupChatOrchestration.
Modernized CLI that works directly with GroupChatOrchestration patterns.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.orchestration.banner_group_chat_orchestration_simple import BannerGroupChatOrchestration
from src.orchestration.script_group_chat_orchestration import ScriptGroupChatOrchestration


class ModernMultiAgentCLI:
    """
    Modern CLI for GroupChatOrchestration workflows.
    Simplified interface that works directly with Semantic Kernel orchestration.
    """
    
    def __init__(self):
        self.console = Console()
        self.workflow = None
        self.workflow_type = None
        self.results_history = []  # Simple history tracking
    
    def select_workflow_type(self) -> str:
        """Allow user to select which workflow type to use."""
        self.console.print("\n[bold blue]🚀 Multi-Agent Team POC[/bold blue]")
        self.console.print("[italic]Powered by Semantic Kernel GroupChatOrchestration[/italic]\n")
        
        table = Table(title="Available Workflows", show_header=True)
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Workflow", style="green")
        table.add_column("Description", style="white")
        
        table.add_row("1", "Video Script Creation", "Create professional video scripts with AI coordination")
        table.add_row("2", "Social Banner Design", "Design engaging banners with AI evaluation")
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
    
    def initialize_workflow(self, workflow_type: str):
        """Initialize the appropriate GroupChatOrchestration workflow."""
        self.workflow_type = workflow_type
        
        try:
            if workflow_type == "script":
                self.workflow = ScriptGroupChatOrchestration()
                
            elif workflow_type == "banner":
                self.workflow = BannerGroupChatOrchestration()
                
            else:
                raise ValueError(f"Unsupported workflow type: {workflow_type}")
                
            # Show workflow info
            info = self.workflow.get_agents_info()
            self.console.print(f"\n[green]✅ {workflow_type.title()} workflow initialized[/green]")
            self.console.print(f"[dim]Agents: {', '.join(info['agent_names'])}[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Error initializing workflow: {e}[/red]")
            self.workflow = None
    
    def display_banner(self):
        """Display the application banner."""
        if self.workflow and self.workflow_type:
            if self.workflow_type == "script":
                title = "📝 Video Script Creation Workflow"
                subtitle = "Professional script writing with AI coordination"
            else:  # banner
                title = "🎨 Social Banner Design Workflow"
                subtitle = "Professional banner creation with AI evaluation"
                
            banner = Text(title, style="bold blue")
            subtitle_text = Text(subtitle, style="italic")
            
            self.console.print(Panel.fit(
                f"{banner}\n{subtitle_text}",
                title="GroupChatOrchestration",
                border_style="blue"
            ))
        else:
            self.console.print(Panel.fit(
                "🚀 Multi-Agent Team POC\nPowered by Semantic Kernel",
                title="Welcome",
                border_style="blue"
            ))
    
    def display_menu(self):
        """Display the main menu options."""
        table = Table(title="Available Actions", show_header=False)
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        
        if self.workflow_type == "script":
            table.add_row("1", "Create new video script")
        else:  # banner
            table.add_row("1", "Create new banner design")
            
        table.add_row("2", "View saved conversations")
        table.add_row("3", "View recent results (legacy)")
        table.add_row("4", "Switch workflow type")
        table.add_row("5", "Export results history")
        table.add_row("r", "Resume saved conversation")
        table.add_row("q", "Quit")
        
        self.console.print(table)
    
    async def start_new_workflow(self):
        """Start a new workflow conversation."""
        if not self.workflow:
            self.console.print("[red]No workflow initialized. Please restart the application.[/red]")
            return
            
        # Get user request based on workflow type
        if self.workflow_type == "script":
            prompt_text = "Describe the video script you want to create (topic, style, duration, etc.)"
            workflow_emoji = "📝"
        else:  # banner
            prompt_text = "Describe the banner you want to create (purpose, style, content, etc.)"
            workflow_emoji = "🎨"
            
        self.console.print(f"\n[bold green]{workflow_emoji} Starting {self.workflow_type} workflow...[/bold green]")
        
        user_request = Prompt.ask(f"\n{prompt_text}")
        
        if not user_request.strip():
            self.console.print("[red]No request provided. Returning to menu.[/red]")
            return
        
        # Process the request with progress indicator
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"Running {self.workflow_type} workflow...", total=None)
                
                # Start the workflow
                result = await self.workflow.start_conversation(user_request)
                
                progress.update(task, description="Workflow completed!", completed=True)
            
            # Check if workflow was paused vs completed
            workflow_title = self.workflow_type.title() if self.workflow_type else "Unknown"
            is_paused = (
                "paused" in result.lower() or
                "workflow paused" in result.lower() or 
                "⏸️" in result
            )
            
            if is_paused:
                # Handle paused workflow
                self.console.print(f"\n[bold yellow]⏸️ {workflow_title} workflow paused![/bold yellow]")
                
                result_panel = Panel(
                    result,
                    title=f"⏸️ {workflow_title} Paused",
                    border_style="yellow",
                    title_align="left"
                )
                self.console.print(result_panel)
                
                self.console.print("[dim]💡 You can resume this workflow later using option 'r' from the main menu.[/dim]")
                
                # Don't save paused workflows to regular history or ask to save files
                
            else:
                # Handle completed workflow
                self.console.print(f"\n[bold green]✅ {workflow_title} workflow completed![/bold green]")
                
                result_panel = Panel(
                    result,
                    title=f"{workflow_emoji} {workflow_title} Result",
                    border_style="green",
                    title_align="left"
                )
                self.console.print(result_panel)
                
                # Save to history
                self._save_to_history(user_request, result)
                
                # Ask if user wants to save
                if Confirm.ask("\nSave this result to file?"):
                    self._save_result_to_file(user_request, result)
        
        except Exception as e:
            self.console.print(f"[red]❌ Error during workflow: {e}[/red]")
    
    def _save_to_history(self, request: str, result: str):
        """Save result to in-memory history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "workflow_type": self.workflow_type,
            "request": request,
            "result": result
        }
        self.results_history.append(entry)
        
        # Keep only last 10 results
        if len(self.results_history) > 10:
            self.results_history = self.results_history[-10:]
    
    def _save_result_to_file(self, request: str, result: str):
        """Save result to a file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_{self.workflow_type}_{timestamp}.txt"
            
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            filepath = os.path.join("results", filename)
            
            workflow_title = self.workflow_type.title() if self.workflow_type else "Unknown"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {workflow_title} Workflow Result\n\n")
                f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Request:** {request}\n\n")
                f.write(f"**Result:**\n{result}\n")
            
            self.console.print(f"[green]✅ Result saved to: {filepath}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Error saving result: {e}[/red]")
    
    def view_recent_results(self):
        """Display recent workflow results."""
        if not self.results_history:
            self.console.print("[yellow]No recent results found.[/yellow]")
            return
        
        table = Table(title="Recent Results")
        table.add_column("Time", style="cyan", no_wrap=True)
        table.add_column("Type", style="green", no_wrap=True)
        table.add_column("Request", style="white", max_width=50)
        table.add_column("Preview", style="dim", max_width=40)
        
        for entry in reversed(self.results_history[-5:]):  # Show last 5
            timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")
            request_preview = entry["request"][:47] + "..." if len(entry["request"]) > 50 else entry["request"]
            result_preview = entry["result"][:37] + "..." if len(entry["result"]) > 40 else entry["result"]
            
            table.add_row(
                timestamp,
                entry["workflow_type"],
                request_preview,
                result_preview
            )
        
        self.console.print(table)
        
        # Option to view full result
        if Confirm.ask("\nView full result from history?"):
            try:
                index = int(Prompt.ask("Enter result number (1-5, newest first)")) - 1
                if 0 <= index < len(self.results_history[-5:]):
                    entry = list(reversed(self.results_history[-5:]))[index]
                    
                    result_panel = Panel(
                        f"**Request:** {entry['request']}\n\n**Result:**\n{entry['result']}",
                        title=f"{entry['workflow_type'].title()} Result - {entry['timestamp'][:19]}",
                        border_style="blue"
                    )
                    self.console.print(result_panel)
                else:
                    self.console.print("[red]Invalid result number.[/red]")
            except ValueError:
                self.console.print("[red]Please enter a valid number.[/red]")
    
    def export_results_history(self):
        """Export all results to a JSON file."""
        if not self.results_history:
            self.console.print("[yellow]No results to export.[/yellow]")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_export_{timestamp}.json"
            
            os.makedirs("exports", exist_ok=True)
            filepath = os.path.join("exports", filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results_history, f, indent=2, ensure_ascii=False)
            
            self.console.print(f"[green]✅ Results exported to: {filepath}[/green]")
            self.console.print(f"[dim]Exported {len(self.results_history)} results[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Error exporting results: {e}[/red]")
    
    def view_saved_conversations(self):
        """View all saved workflow conversations with detailed metadata."""
        try:
            from src.serialization.conversation_serializer import WorkflowSerializer
            serializer = WorkflowSerializer()
            conversations = serializer.list_saved_conversations()
            
            if not conversations:
                self.console.print("[yellow]No saved conversations found.[/yellow]")
                return
            
            table = Table(title="Saved Conversations")
            table.add_column("ID", style="cyan", no_wrap=True, max_width=15)
            table.add_column("Type", style="green", no_wrap=True)
            table.add_column("Status", style="white", no_wrap=True)
            table.add_column("Request", style="dim", max_width=30)
            table.add_column("Messages", style="yellow", no_wrap=True)
            table.add_column("Progress", style="blue", no_wrap=True)
            table.add_column("Last Updated", style="magenta", no_wrap=True)
            table.add_column("Can Resume", style="bright_green", no_wrap=True)
            
            for conv in conversations[:10]:  # Show last 10
                conversation_id_short = conv["conversation_id"][-12:] if len(conv["conversation_id"]) > 12 else conv["conversation_id"]
                request_preview = conv["user_request"][:27] + "..." if len(conv["user_request"]) > 30 else conv["user_request"]
                last_updated = conv["last_updated"][:10] if len(conv["last_updated"]) > 10 else conv["last_updated"]
                progress = f"{conv.get('completion_percentage', 0)*100:.0f}%" if conv.get('completion_percentage') else "N/A"
                
                table.add_row(
                    conversation_id_short,
                    conv["workflow_type"],
                    conv["status"],
                    request_preview,
                    str(conv["total_messages"]),
                    progress,
                    last_updated,
                    "✅" if conv["can_resume"] else "❌"
                )
            
            self.console.print(table)
            
            # Option to view full conversation details
            if conversations and Confirm.ask("\nView full conversation details?"):
                try:
                    conv_id = Prompt.ask("Enter conversation ID (first few characters are enough)")
                    
                    # Find matching conversation
                    matching_conv = None
                    for conv in conversations:
                        if conv["conversation_id"].startswith(conv_id) or conv_id in conv["conversation_id"]:
                            matching_conv = conv
                            break
                    
                    if matching_conv:
                        self._display_conversation_details(matching_conv, serializer)
                    else:
                        self.console.print("[red]Conversation not found.[/red]")
                        
                except Exception as e:
                    self.console.print(f"[red]Error viewing conversation: {e}[/red]")
                    
        except Exception as e:
            self.console.print(f"[red]❌ Error loading saved conversations: {e}[/red]")
    
    def _display_conversation_details(self, conv_info: dict, serializer):
        """Display detailed information about a specific conversation."""
        try:
            snapshot = serializer.load_conversation_snapshot(conv_info["conversation_id"])
            if not snapshot:
                self.console.print("[red]Could not load conversation snapshot.[/red]")
                return
            
            # Create detailed info panel
            details = f"""[bold]Conversation Details[/bold]

[cyan]ID:[/cyan] {snapshot.conversation_id}
[cyan]Workflow:[/cyan] {snapshot.metadata.workflow_name}
[cyan]Type:[/cyan] {snapshot.metadata.workflow_type}
[cyan]Status:[/cyan] {snapshot.metadata.status.value if hasattr(snapshot.metadata.status, 'value') else snapshot.metadata.status}
[cyan]Created:[/cyan] {snapshot.metadata.created_at[:19]}
[cyan]Last Updated:[/cyan] {snapshot.metadata.last_updated[:19]}
[cyan]Total Messages:[/cyan] {snapshot.metadata.total_messages}
[cyan]Current Phase:[/cyan] {snapshot.workflow_state.current_phase}
[cyan]Can Resume:[/cyan] {'Yes' if conv_info["can_resume"] else 'No'}

[cyan]User Request:[/cyan]
{snapshot.metadata.user_request}

[cyan]Checkpoint Reason:[/cyan]
{snapshot.checkpoint_reason}
"""
            
            if snapshot.resume_instructions:
                details += f"\n[cyan]Resume Instructions:[/cyan]\n{snapshot.resume_instructions}"
            
            self.console.print(Panel(details, title="Conversation Details", border_style="blue"))
            
            # Show agent participation
            if snapshot.workflow_state.agent_states:
                agent_table = Table(title="Agent Participation")
                agent_table.add_column("Agent", style="green")
                agent_table.add_column("Type", style="cyan")
                agent_table.add_column("Messages", style="yellow")
                agent_table.add_column("Last Message Index", style="dim")
                
                for agent_state in snapshot.workflow_state.agent_states:
                    agent_table.add_row(
                        agent_state.name,
                        agent_state.type,
                        str(agent_state.message_count),
                        str(agent_state.last_message_index) if agent_state.last_message_index is not None else "N/A"
                    )
                
                self.console.print(agent_table)
                
        except Exception as e:
            self.console.print(f"[red]Error displaying conversation details: {e}[/red]")
    
    async def resume_saved_conversation(self):
        """Resume a saved conversation that was paused for user input."""
        try:
            from src.serialization.conversation_serializer import WorkflowSerializer, WorkflowStatus
            serializer = WorkflowSerializer()
            conversations = serializer.list_saved_conversations()
            
            if not conversations:
                self.console.print("[yellow]No saved conversations found.[/yellow]")
                return
            
            # Filter conversations that are waiting for user input
            # Include both properly saved WAITING_FOR_INPUT and legacy paused workflows
            waiting_conversations = []
            for conv in conversations:
                # Check if status is WAITING_FOR_INPUT (new format)
                if conv["status"] == WorkflowStatus.WAITING_FOR_INPUT.value:
                    waiting_conversations.append(conv)
                # Check if this is a paused workflow with "completed" status (legacy bug)
                elif (conv["status"] == WorkflowStatus.COMPLETED.value and 
                      conv.get("checkpoint_reason") and 
                      ("pause" in conv["checkpoint_reason"].lower() or 
                       "waiting for" in conv["checkpoint_reason"].lower())):
                    # This is a paused workflow that was incorrectly saved as completed
                    conv["status"] = "waiting_for_input"  # Update display status
                    waiting_conversations.append(conv)
                # Also check if last message in conversation history is "pause"
                elif conv["status"] == WorkflowStatus.COMPLETED.value:
                    try:
                        snapshot = serializer.load_conversation_snapshot(conv["conversation_id"])
                        if (snapshot and snapshot.chat_history and 
                            len(snapshot.chat_history) > 0):
                            last_msg = snapshot.chat_history[-1]
                            if (last_msg.get("role") == "user" and 
                                last_msg.get("content", "").strip().lower() == "pause"):
                                conv["status"] = "waiting_for_input"  # Update display status
                                waiting_conversations.append(conv)
                    except Exception as e:
                        print(f"⚠️ Error checking conversation {conv['conversation_id']}: {e}")
                        continue
            
            if not waiting_conversations:
                self.console.print("[yellow]No workflows are currently waiting for user input.[/yellow]")
                self.console.print("[dim]Only paused workflows can be resumed.[/dim]")
                return
            
            # Display waiting conversations
            table = Table(title="Workflows Waiting for User Input")
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("ID", style="cyan", no_wrap=True, max_width=15)
            table.add_column("Type", style="green", no_wrap=True)
            table.add_column("Request", style="dim", max_width=40)
            table.add_column("Last Updated", style="magenta", no_wrap=True)
            
            for i, conv in enumerate(waiting_conversations[:10], 1):
                conversation_id_short = conv["conversation_id"][-12:] if len(conv["conversation_id"]) > 12 else conv["conversation_id"]
                request_preview = conv["user_request"][:37] + "..." if len(conv["user_request"]) > 40 else conv["user_request"]
                last_updated = conv["last_updated"][:10] if len(conv["last_updated"]) > 10 else conv["last_updated"]
                
                table.add_row(
                    str(i),
                    conversation_id_short,
                    conv["workflow_type"],
                    request_preview,
                    last_updated
                )
            
            self.console.print(table)
            
            # Let user select which conversation to resume
            if len(waiting_conversations) == 1:
                selected_conv = waiting_conversations[0]
                self.console.print(f"\n[green]Auto-selecting the only available workflow: {selected_conv['conversation_id'][-12:]}[/green]")
            else:
                choice = Prompt.ask("\nSelect a workflow to resume (1-10)")
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(waiting_conversations):
                        selected_conv = waiting_conversations[choice_idx]
                    else:
                        self.console.print("[red]Invalid selection.[/red]")
                        return
                except ValueError:
                    self.console.print("[red]Invalid selection.[/red]")
                    return
            
            # Load the conversation snapshot
            snapshot = serializer.load_conversation_snapshot(selected_conv["conversation_id"])
            if not snapshot:
                self.console.print("[red]Could not load conversation snapshot.[/red]")
                return
            
            # Display context
            self.console.print(f"\n[bold blue]🔄 Resuming workflow: {snapshot.metadata.workflow_type}[/bold blue]")
            self.console.print(f"[cyan]Request:[/cyan] {snapshot.metadata.user_request}")
            self.console.print(f"[cyan]Status:[/cyan] {snapshot.resume_instructions}")
            
            # Show recent conversation context
            if snapshot.chat_history:
                self.console.print("\n[bold]Recent conversation:[/bold]")
                recent_messages = snapshot.chat_history[-3:]  # Show last 3 messages
                for msg in recent_messages:
                    name = msg.get('name', 'Unknown')
                    content = msg.get('content', '')
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    self.console.print(f"[dim]{name}:[/dim] {content_preview}")
            
            # Get user input
            self.console.print("\n[bold green]💬 Your approval is needed![/bold green]")
            self.console.print("💡 Type 'approve' to accept the script")
            self.console.print("💡 Type 'reject' to request a rewrite")
            self.console.print("💡 Type 'pause' to pause again")
            self.console.print("💡 Or provide specific feedback for improvements")
            
            user_input = Prompt.ask("\nYour response").strip()
            
            if user_input.lower() == 'pause':
                self.console.print("[yellow]Workflow remains paused.[/yellow]")
                return
            
            if not user_input:
                self.console.print("[red]No input provided. Workflow remains paused.[/red]")
                return
            
            # Determine the appropriate workflow type and resume
            workflow_type = snapshot.metadata.workflow_type
            if workflow_type == "script_writing":
                if not isinstance(self.workflow, ScriptGroupChatOrchestration):
                    self.workflow = ScriptGroupChatOrchestration()
                
                # Resume the workflow
                self.console.print(f"\n[bold green]🚀 Resuming {workflow_type} workflow...[/bold green]")
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Resuming workflow...", total=None)
                    
                    result = await self.workflow.resume_conversation(snapshot, user_input)
                    
                    progress.update(task, description="Workflow resumed and completed!", completed=True)
                
                # Display the result
                self.console.print(f"\n[bold green]✅ {workflow_type.title()} workflow resumed and completed![/bold green]")
                
                result_panel = Panel(
                    result,
                    title=f"📝 Resume Result",
                    border_style="green",
                    title_align="left"
                )
                self.console.print(result_panel)
                
                # Save to history
                self._save_to_history(f"Resumed: {snapshot.metadata.user_request}", result)
                
            else:
                self.console.print(f"[red]Unsupported workflow type for resume: {workflow_type}[/red]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Error resuming conversation: {e}[/red]")
    
    async def run(self):
        """Main application loop."""
        # Check for required environment variables
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.console.print("[red]❌ Error: No OpenAI API key or Azure OpenAI endpoint found.[/red]")
            self.console.print("[yellow]Please set your API credentials in the api_keys_config.env file.[/yellow]")
            return
        
        # Select and initialize workflow
        workflow_type = self.select_workflow_type()
        self.initialize_workflow(workflow_type)
        
        if not self.workflow:
            self.console.print("[red]❌ Failed to initialize workflow. Exiting.[/red]")
            return
        
        self.display_banner()
        
        # Main interaction loop
        while True:
            self.display_menu()
            choice = Prompt.ask("\nSelect an option")
            
            try:
                if choice == "1":
                    await self.start_new_workflow()
                elif choice == "2":
                    self.view_saved_conversations()
                elif choice == "3":
                    self.view_recent_results()
                elif choice == "4":
                    # Switch workflow type
                    workflow_type = self.select_workflow_type()
                    self.initialize_workflow(workflow_type)
                    if self.workflow:
                        self.display_banner()
                elif choice == "5":
                    self.export_results_history()
                elif choice.lower() == "r":
                    await self.resume_saved_conversation()
                elif choice.lower() in ["q", "quit"]:
                    self.console.print("[bold blue]👋 Thank you for using the Multi-Agent Team POC![/bold blue]")
                    break
                else:
                    self.console.print("[red]Invalid option. Please try again.[/red]")
            
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Operation cancelled.[/yellow]")
            except Exception as e:
                self.console.print(f"[red]❌ Error: {e}[/red]")
            
            if choice not in ["q", "quit"]:
                Prompt.ask("\nPress Enter to continue")


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv("api_keys_config.env")
    
    # Create and run the modern CLI application
    app = ModernMultiAgentCLI()
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0) 