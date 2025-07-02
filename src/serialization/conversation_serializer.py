"""
Conversation serializer for pausing and resuming multi-agent conversations.
"""

import json
import jsonpickle
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConversationSerializer:
    """
    Handles serialization and deserialization of conversation state for pause/resume functionality.
    """
    
    def __init__(self, save_directory: str = "saved_conversations"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
    
    def save_conversation(self, 
                         conversation_id: str, 
                         workflow_state: Dict[str, Any], 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save conversation state to file.
        
        Args:
            conversation_id: Unique identifier for the conversation
            workflow_state: Current workflow state from GroupChatManager
            metadata: Additional metadata to save
            
        Returns:
            File path where conversation was saved
        """
        save_data = {
            "conversation_id": conversation_id,
            "saved_at": datetime.now().isoformat(),
            "workflow_state": workflow_state,
            "metadata": metadata or {}
        }
        
        # Use jsonpickle for more complex object serialization
        serialized_data = jsonpickle.encode(save_data, indent=2)
        
        file_path = self.save_directory / f"{conversation_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(serialized_data))
        
        return str(file_path)
    
    def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation state from file.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Deserialized conversation state or None if not found
        """
        file_path = self.save_directory / f"{conversation_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                serialized_data = f.read()
            
            decoded_data = jsonpickle.decode(serialized_data)
            # Type check to ensure we got a dictionary
            if isinstance(decoded_data, dict):
                return decoded_data
            else:
                print(f"Warning: Decoded data is not a dictionary for {conversation_id}")
                return None
        
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading conversation {conversation_id}: {e}")
            return None
    
    def list_saved_conversations(self) -> List[Dict[str, str]]:
        """
        List all saved conversations with basic metadata.
        
        Returns:
            List of conversation info dictionaries
        """
        conversations = []
        
        for file_path in self.save_directory.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    decoded_data = jsonpickle.decode(f.read())
                
                # Type check to ensure we got a dictionary
                if isinstance(decoded_data, dict):
                    data = decoded_data
                    workflow_state = data.get("workflow_state", {})
                    if isinstance(workflow_state, dict):
                        # Use generic status instead of task-specific phase
                        conversation_history = workflow_state.get("conversation_history", [])
                        if conversation_history:
                            status = f"{len(conversation_history)} messages"
                        else:
                            status = "Empty"
                    else:
                        status = "Unknown"
                    
                    conversations.append({
                        "conversation_id": data.get("conversation_id", file_path.stem),
                        "saved_at": data.get("saved_at", "Unknown"),
                        "status": status,
                        "file_path": str(file_path)
                    })
            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        # Sort by save time, most recent first
        conversations.sort(key=lambda x: x["saved_at"], reverse=True)
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a saved conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            True if deleted successfully, False otherwise
        """
        file_path = self.save_directory / f"{conversation_id}.json"
        
        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting conversation {conversation_id}: {e}")
            return False
    
    def generate_conversation_id(self, user_prompt: str = "") -> str:
        """
        Generate a unique conversation ID based on timestamp and prompt.
        
        Args:
            user_prompt: Optional user prompt to include in ID generation
            
        Returns:
            Unique conversation identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a safe filename from user prompt
        if user_prompt:
            safe_prompt = "".join(c for c in user_prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')
            if safe_prompt:
                return f"{timestamp}_{safe_prompt}"
        
        return f"conversation_{timestamp}"
    
    def export_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """
        Export a human-readable summary of the conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Formatted conversation summary or None if conversation not found
        """
        data = self.load_conversation(conversation_id)
        if not data:
            return None
        
        workflow_state = data.get("workflow_state", {})
        conversation_history = workflow_state.get("conversation_history", [])
        
        summary = f"""
Conversation Summary: {conversation_id}
Saved: {data.get('saved_at', 'Unknown')}
Messages: {len(conversation_history)}

Conversation History:
"""
        
        for i, msg in enumerate(conversation_history, 1):
            role = msg.get('role', 'Unknown')
            content = msg.get('content', 'No content')
            # Truncate long messages for summary
            if len(content) > 150:
                content = content[:147] + "..."
            summary += f"{i}. {role}: {content}\n"
        
        return summary 