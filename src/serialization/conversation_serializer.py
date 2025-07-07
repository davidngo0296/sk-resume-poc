"""
Enhanced workflow serialization system for saving and resuming any workflow with full conversation history and state.
"""

import json
import jsonpickle
from datetime import datetime
from typing import Dict, Any, Optional, List, Type, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.agents.orchestration.group_chat import GroupChatManager


class WorkflowStatus(Enum):
    """Workflow status enumeration."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class WorkflowMetadata:
    """Metadata about the workflow."""
    workflow_type: str
    workflow_name: str
    created_at: str
    last_updated: str
    status: WorkflowStatus
    user_request: str
    total_messages: int
    current_agent: Optional[str] = None
    completion_percentage: Optional[float] = None
    tags: Optional[List[str]] = None
    

@dataclass
class AgentState:
    """State information for an individual agent."""
    name: str
    type: str
    instructions: Optional[str] = None
    last_message_index: Optional[int] = None
    message_count: int = 0
    custom_state: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowState:
    """Complete workflow state information."""
    current_phase: str
    workflow_flags: Dict[str, Any]
    agent_states: List[AgentState]
    termination_conditions: Dict[str, Any]
    selection_criteria: Dict[str, Any]
    custom_manager_state: Optional[Dict[str, Any]] = None


@dataclass
class ConversationSnapshot:
    """Complete conversation snapshot with all context."""
    conversation_id: str
    metadata: WorkflowMetadata
    workflow_state: WorkflowState
    chat_history: List[Dict[str, Any]]  # Serialized ChatMessageContent objects
    checkpoint_reason: str
    resume_instructions: Optional[str] = None


class WorkflowSerializer:
    """
    Enhanced workflow serialization system for saving and resuming any workflow.
    
    Features:
    - Full conversation history preservation
    - Workflow state tracking and resumption
    - Generic support for any workflow type
    - Agent state management
    - Resume capability with context preservation
    """
    
    def __init__(self, save_directory: str = "saved_conversations"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        self.results_directory = Path("results")
        self.results_directory.mkdir(exist_ok=True)
    
    def create_conversation_snapshot(self, 
                                   conversation_id: str,
                                   workflow_name: str,
                                   workflow_type: str,
                                   user_request: str,
                                   chat_history: ChatHistory,
                                   group_chat_manager: GroupChatManager,
                                   agents: List[Any],
                                   checkpoint_reason: str = "Manual save",
                                   status: WorkflowStatus = WorkflowStatus.RUNNING,
                                   current_agent: Optional[str] = None,
                                   resume_instructions: Optional[str] = None,
                                   custom_manager_state: Optional[Dict[str, Any]] = None) -> ConversationSnapshot:
        """
        Create a complete conversation snapshot.
        
        Args:
            conversation_id: Unique identifier for the conversation
            workflow_name: Human-readable workflow name
            workflow_type: Type of workflow (e.g., 'script_writing', 'banner_design')
            user_request: Original user request
            chat_history: Current chat history
            group_chat_manager: The workflow's group chat manager
            agents: List of agents in the workflow
            checkpoint_reason: Why this snapshot was created
            status: Current workflow status
            current_agent: Currently active agent
            resume_instructions: Instructions for resuming
            custom_manager_state: Custom state from the manager
            
        Returns:
            ConversationSnapshot object
        """
        now = datetime.now().isoformat()
        
        # Extract agent states
        agent_states = []
        for agent in agents:
            agent_name = getattr(agent, 'name', 'Unknown')
            agent_type = type(agent).__name__
            agent_instructions = getattr(agent, 'instructions', None)
            
            # Count messages from this agent
            message_count = sum(1 for msg in chat_history.messages 
                              if hasattr(msg, 'name') and msg.name == agent_name)
            
            # Find last message index
            last_message_index = None
            for i, msg in enumerate(reversed(chat_history.messages)):
                if hasattr(msg, 'name') and msg.name == agent_name:
                    last_message_index = len(chat_history.messages) - 1 - i
                    break
            
            agent_states.append(AgentState(
                name=agent_name,
                type=agent_type,
                instructions=agent_instructions,
                last_message_index=last_message_index,
                message_count=message_count
            ))
        
        # Extract workflow state from manager
        workflow_flags: Dict[str, Any] = {}
        current_phase = "unknown"
        
        # Try to extract common workflow state attributes
        if hasattr(group_chat_manager, '_workflow_state'):
            current_phase = str(getattr(group_chat_manager, '_workflow_state', 'unknown'))
        
        # Extract various workflow flags
        for attr_name in dir(group_chat_manager):
            if attr_name.startswith('_') and not attr_name.startswith('__'):
                try:
                    attr_value = getattr(group_chat_manager, attr_name)
                    if isinstance(attr_value, (str, int, float, bool, list, dict)):
                        workflow_flags[attr_name] = attr_value
                    elif attr_value is None:
                        workflow_flags[attr_name] = "None"  # Convert None to string for serialization
                except (AttributeError, TypeError):
                    # Skip attributes that can't be accessed or serialized
                    continue
        
        # Serialize chat history
        serialized_history = []
        for msg in chat_history.messages:
            msg_dict = {
                'role': msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                'content': str(msg.content),
                'timestamp': now  # Add timestamp if not present
            }
            
            # Add agent name if available
            if hasattr(msg, 'name'):
                name = getattr(msg, 'name', None)
                if name is not None:
                    msg_dict['name'] = name
            
            # Add any other custom attributes
            for attr in ['metadata', 'encoding', 'mime_type']:
                if hasattr(msg, attr):
                    value = getattr(msg, attr)
                    if value is not None:
                        msg_dict[attr] = str(value)  # Convert to string to avoid type issues
            
            serialized_history.append(msg_dict)
        
        # Create metadata
        metadata = WorkflowMetadata(
            workflow_type=workflow_type,
            workflow_name=workflow_name,
            created_at=now,
            last_updated=now,
            status=status,
            user_request=user_request,
            total_messages=len(chat_history.messages),
            current_agent=current_agent,
            completion_percentage=self._calculate_completion_percentage(workflow_type, current_phase),
            tags=self._generate_tags(workflow_type, user_request)
        )
        
        # Create workflow state
        workflow_state = WorkflowState(
            current_phase=current_phase,
            workflow_flags=workflow_flags,
            agent_states=agent_states,
            termination_conditions={},
            selection_criteria={},
            custom_manager_state=custom_manager_state
        )
        
        return ConversationSnapshot(
            conversation_id=conversation_id,
            metadata=metadata,
            workflow_state=workflow_state,
            chat_history=serialized_history,
            checkpoint_reason=checkpoint_reason,
            resume_instructions=resume_instructions
        )
    
    def save_conversation_snapshot(self, snapshot: ConversationSnapshot) -> str:
        """
        Save a conversation snapshot to file.
        
        Args:
            snapshot: ConversationSnapshot to save
            
        Returns:
            File path where snapshot was saved
        """
        # Convert to dictionary for serialization
        snapshot_dict = asdict(snapshot)
        
        # Handle enum serialization
        if isinstance(snapshot.metadata.status, WorkflowStatus):
            snapshot_dict['metadata']['status'] = snapshot.metadata.status.value
        
        # Use jsonpickle for robust serialization
        serialized_data = jsonpickle.encode(snapshot_dict, indent=2)
        
        file_path = self.save_directory / f"{snapshot.conversation_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(serialized_data))
        
        print(f"✅ Conversation snapshot saved: {file_path}")
        return str(file_path)
    
    def load_conversation_snapshot(self, conversation_id: str) -> Optional[ConversationSnapshot]:
        """
        Load a conversation snapshot from file with backward compatibility.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            ConversationSnapshot or None if not found
        """
        file_path = self.save_directory / f"{conversation_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                serialized_data = f.read()
            
            snapshot_dict = jsonpickle.decode(serialized_data)
            
            if not isinstance(snapshot_dict, dict):
                return None
            
            # Check if this is a legacy format and migrate it
            if self._is_legacy_format(snapshot_dict):
                migrated_snapshot = self._migrate_legacy_format(snapshot_dict, conversation_id)
                if not migrated_snapshot:
                    return None
                # Migration returns a ConversationSnapshot object directly
                return migrated_snapshot
            
            # Handle enum deserialization for modern format
            if (isinstance(snapshot_dict, dict) and 
                'metadata' in snapshot_dict and 
                isinstance(snapshot_dict['metadata'], dict) and 
                'status' in snapshot_dict['metadata']):
                status_value = snapshot_dict['metadata']['status']
                if isinstance(status_value, str):
                    snapshot_dict['metadata']['status'] = WorkflowStatus(status_value)
            
            # Convert back to ConversationSnapshot - need to reconstruct dataclass objects
            if isinstance(snapshot_dict, dict):
                try:
                    # Reconstruct WorkflowMetadata
                    metadata_dict = snapshot_dict.get('metadata', {})
                    metadata = WorkflowMetadata(**metadata_dict)
                    
                    # Reconstruct WorkflowState
                    workflow_state_dict = snapshot_dict.get('workflow_state', {})
                    
                    # Reconstruct AgentState objects
                    agent_states = []
                    for agent_dict in workflow_state_dict.get('agent_states', []):
                        agent_states.append(AgentState(**agent_dict))
                    
                    workflow_state = WorkflowState(
                        current_phase=workflow_state_dict.get('current_phase', 'unknown'),
                        workflow_flags=workflow_state_dict.get('workflow_flags', {}),
                        agent_states=agent_states,
                        termination_conditions=workflow_state_dict.get('termination_conditions', {}),
                        selection_criteria=workflow_state_dict.get('selection_criteria', {}),
                        custom_manager_state=workflow_state_dict.get('custom_manager_state')
                    )
                    
                    # Create the complete snapshot
                    return ConversationSnapshot(
                        conversation_id=snapshot_dict.get('conversation_id', ''),
                        metadata=metadata,
                        workflow_state=workflow_state,
                        chat_history=snapshot_dict.get('chat_history', []),
                        checkpoint_reason=snapshot_dict.get('checkpoint_reason', ''),
                        resume_instructions=snapshot_dict.get('resume_instructions')
                    )
                except Exception as e:
                    print(f"❌ Error reconstructing ConversationSnapshot: {e}")
                    return None
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error loading conversation snapshot {conversation_id}: {e}")
            return None
    
    def _is_legacy_format(self, data: dict) -> bool:
        """Check if the loaded data is in legacy format."""
        # Legacy format has 'saved_at' field and empty/missing metadata structure
        has_saved_at = 'saved_at' in data
        has_workflow_state = 'workflow_state' in data
        metadata = data.get('metadata', {})
        
        # Legacy format: has saved_at, has workflow_state, and metadata is empty or missing critical fields
        is_legacy_metadata = (
            not metadata or  # Empty metadata
            ('workflow_name' not in metadata or 'status' not in metadata)  # Missing critical fields
        )
        
        is_legacy = has_saved_at and has_workflow_state and is_legacy_metadata
        
        # Also check if it has the specific legacy structure
        if is_legacy:
            # Legacy format has workflow_state.workflow_state.conversation_history
            workflow_state = data.get('workflow_state', {})
            if isinstance(workflow_state, dict) and 'workflow_state' in workflow_state:
                inner_state = workflow_state['workflow_state']
                if isinstance(inner_state, dict) and 'conversation_history' in inner_state:
                    print(f"🔍 Detected legacy format with nested workflow_state structure")
                    return True
        
        # Removed verbose debug message - only print for actual legacy detection
        return False
    
    def _migrate_legacy_format(self, legacy_data: dict, conversation_id: str) -> Optional[ConversationSnapshot]:
        """Migrate legacy conversation format to new ConversationSnapshot format."""
        try:
            # Extract legacy fields
            saved_at = legacy_data.get('saved_at', datetime.now().isoformat())
            workflow_state = legacy_data.get('workflow_state', {})
            
            # Handle nested workflow_state structure
            conversation_history = []
            current_script = None
            script_approved = False
            
            if isinstance(workflow_state, dict) and 'workflow_state' in workflow_state:
                inner_state = workflow_state['workflow_state']
                if isinstance(inner_state, dict):
                    conversation_history = inner_state.get('conversation_history', [])
                    current_script = inner_state.get('current_script')
                    script_approved = inner_state.get('script_approved', False)
            
            # Extract user request from conversation history
            user_request = 'Legacy conversation - no user request recorded'
            if conversation_history:
                first_message = conversation_history[0]
                if isinstance(first_message, dict) and first_message.get('role') == 'user':
                    user_request = first_message.get('content', user_request)
            
            # Determine workflow status based on script approval
            if script_approved:
                workflow_status = WorkflowStatus.COMPLETED.value
                current_phase = 'complete'
                completion_percentage = 1.0
            else:
                workflow_status = WorkflowStatus.PAUSED.value
                current_phase = 'paused'
                completion_percentage = 0.6
            
            # Create new metadata structure as dataclass
            metadata = WorkflowMetadata(
                workflow_type='script_writing',
                workflow_name='Legacy Script Writing Conversation',
                created_at=saved_at,
                last_updated=saved_at,
                status=WorkflowStatus(workflow_status),
                user_request=user_request,
                total_messages=len(conversation_history),
                current_agent=None,
                completion_percentage=completion_percentage,
                tags=['legacy', 'script_writing']
            )
            
            # Create new workflow state structure as dataclass
            new_workflow_state = WorkflowState(
                current_phase=current_phase,
                workflow_flags={
                    '_script_approved': script_approved,
                    '_current_script': current_script
                },
                agent_states=[],
                termination_conditions={},
                selection_criteria={},
                custom_manager_state=None
            )
            
            # Convert conversation history to new format
            new_chat_history = []
            for msg in conversation_history:
                if isinstance(msg, dict):
                    new_msg = {
                        'role': msg.get('role', 'assistant'),
                        'content': msg.get('content', ''),
                        'timestamp': saved_at
                    }
                    if 'name' in msg:
                        new_msg['name'] = msg['name']
                    new_chat_history.append(new_msg)
            
            # Create new snapshot as dataclass
            migrated_snapshot = ConversationSnapshot(
                conversation_id=conversation_id,
                metadata=metadata,
                workflow_state=new_workflow_state,
                chat_history=new_chat_history,
                checkpoint_reason='Legacy conversation migrated',
                resume_instructions=f'This is a migrated legacy conversation. Script approved: {script_approved}'
            )
            
            print(f"📁 Migrated legacy conversation: {conversation_id} (status: {workflow_status})")
            return migrated_snapshot
            
        except Exception as e:
            print(f"⚠️ Failed to migrate legacy conversation {conversation_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def restore_chat_history(self, snapshot: ConversationSnapshot) -> ChatHistory:
        """
        Restore ChatHistory from a snapshot.
        
        Args:
            snapshot: ConversationSnapshot containing the chat history
            
        Returns:
            Restored ChatHistory object
        """
        chat_history = ChatHistory()
        
        for msg_dict in snapshot.chat_history:
            # Convert role back to AuthorRole
            role_str = msg_dict.get('role', 'assistant')
            if role_str == 'user':
                role = AuthorRole.USER
            elif role_str == 'system':
                role = AuthorRole.SYSTEM
            else:
                role = AuthorRole.ASSISTANT
            
            # Create ChatMessageContent
            msg = ChatMessageContent(
                role=role,
                content=msg_dict.get('content', '')
            )
            
            # Add agent name if available
            if 'name' in msg_dict:
                msg.name = msg_dict['name']
            
            chat_history.add_message(msg)
        
        return chat_history
    
    def create_result_file(self, snapshot: ConversationSnapshot, result_content: str) -> str:
        """
        Create a formatted result file from a conversation snapshot.
        
        Args:
            snapshot: ConversationSnapshot containing the conversation
            result_content: Final result content to include
            
        Returns:
            Path to the created result file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{snapshot.metadata.workflow_type}_{timestamp}.txt"
        result_path = self.results_directory / result_filename
        
        # Create comprehensive result file
        result_text = f"""# {snapshot.metadata.workflow_name} Result

**Conversation ID:** {snapshot.conversation_id}
**Workflow Type:** {snapshot.metadata.workflow_type}
**Status:** {snapshot.metadata.status.value if isinstance(snapshot.metadata.status, WorkflowStatus) else str(snapshot.metadata.status)}
**Timestamp:** {snapshot.metadata.last_updated}
**User Request:** {snapshot.metadata.user_request}
**Total Messages:** {snapshot.metadata.total_messages}

## Workflow State
**Current Phase:** {snapshot.workflow_state.current_phase}
**Checkpoint Reason:** {snapshot.checkpoint_reason}

## Agent Participation
"""
        
        # Add agent participation summary
        for agent_state in snapshot.workflow_state.agent_states:
            result_text += f"- **{agent_state.name}** ({agent_state.type}): {agent_state.message_count} messages\n"
        
        result_text += "\n## Full Conversation History\n\n"
        
        # Add complete conversation
        for i, msg_dict in enumerate(snapshot.chat_history, 1):
            agent_name = msg_dict.get('name', msg_dict.get('role', 'System'))
            content = msg_dict.get('content', '')
            
            result_text += f"**{i}. {agent_name}:**\n{content}\n\n"
            result_text += "-" * 80 + "\n\n"
        
        result_text += f"## Final Result\n\n{result_content}\n\n"
        
        # Add resume information if available
        if snapshot.resume_instructions:
            result_text += f"## Resume Instructions\n\n{snapshot.resume_instructions}\n\n"
        
        result_text += f"## Serialization Info\n\n"
        result_text += f"- **Serialization File:** {snapshot.conversation_id}.json\n"
        result_text += f"- **Can Resume:** {'Yes' if snapshot.metadata.status != WorkflowStatus.COMPLETED else 'No'}\n"
        result_text += f"- **Workflow Flags:** {len(snapshot.workflow_state.workflow_flags)} state variables saved\n"
        
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        
        print(f"✅ Result file created: {result_path}")
        return str(result_path)
    
    def list_saved_conversations(self) -> List[Dict[str, Any]]:
        """
        List all saved conversations with detailed metadata.
        
        Returns:
            List of conversation info dictionaries
        """
        conversations = []
        
        for file_path in self.save_directory.glob("*.json"):
            try:
                snapshot = self.load_conversation_snapshot(file_path.stem)
                if snapshot:
                    conversations.append({
                        "conversation_id": snapshot.conversation_id,
                        "workflow_name": snapshot.metadata.workflow_name,
                        "workflow_type": snapshot.metadata.workflow_type,
                        "status": snapshot.metadata.status.value if isinstance(snapshot.metadata.status, WorkflowStatus) else str(snapshot.metadata.status),
                        "user_request": snapshot.metadata.user_request,
                        "total_messages": snapshot.metadata.total_messages,
                        "current_phase": snapshot.workflow_state.current_phase,
                        "current_agent": snapshot.metadata.current_agent,
                        "completion_percentage": snapshot.metadata.completion_percentage,
                        "last_updated": snapshot.metadata.last_updated,
                        "checkpoint_reason": snapshot.checkpoint_reason,
                        "can_resume": snapshot.metadata.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED],
                        "file_path": str(file_path)
                    })
            
            except Exception as e:
                # Silently skip corrupted files to avoid cluttering output
                continue
        
        # Sort by last updated, most recent first
        conversations.sort(key=lambda x: x["last_updated"], reverse=True)
        return conversations
    
    def _calculate_completion_percentage(self, workflow_type: str, current_phase: str) -> float:
        """Calculate estimated completion percentage based on workflow type and phase."""
        phase_weights = {
            'script_writing': {
                'start': 0.0,
                'script_written': 0.6,
                'user_approved': 0.8,
                'complete': 1.0
            },
            'banner_design': {
                'start': 0.0,
                'design_created': 0.5,
                'design_evaluated': 0.8,
                'complete': 1.0
            }
        }
        
        if workflow_type in phase_weights and current_phase in phase_weights[workflow_type]:
            return phase_weights[workflow_type][current_phase]
        
        return 0.5  # Default to 50% if unknown
    
    def _generate_tags(self, workflow_type: str, user_request: str) -> List[str]:
        """Generate tags based on workflow type and user request."""
        tags = [workflow_type]
        
        # Add content-based tags
        request_lower = user_request.lower()
        if 'video' in request_lower:
            tags.append('video')
        if 'script' in request_lower:
            tags.append('script')
        if 'banner' in request_lower:
            tags.append('banner')
        if 'design' in request_lower:
            tags.append('design')
        
        return tags
    
    def generate_conversation_id(self, workflow_type: str, user_prompt: str = "") -> str:
        """
        Generate a unique conversation ID.
        
        Args:
            workflow_type: Type of workflow
            user_prompt: Optional user prompt to include in ID
            
        Returns:
            Unique conversation identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a safe filename from user prompt
        if user_prompt:
            safe_prompt = "".join(c for c in user_prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')
            if safe_prompt:
                return f"{timestamp}_{workflow_type}_{safe_prompt}"
        
        return f"{timestamp}_{workflow_type}"


# Alias for backward compatibility
ConversationSerializer = WorkflowSerializer 