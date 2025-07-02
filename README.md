# 🚀 Multi-Agent Team POC

A Proof of Concept for a multi-agent team using **Semantic Kernel** with **Group Chat Orchestration** pattern to handle different collaborative workflows.

## 🎯 Overview

This POC demonstrates a sophisticated multi-agent system that can handle different types of collaborative workflows. The system features multiple specialized workflows each with their own expert agents:

### 🎬 Video Script Creation Workflow
- **Manager Agent** - Orchestrates the workflow and handles all user interactions
- **Script Writer Agent** - Creates engaging video scripts based on user requirements  
- **Audio Selector Agent** - Suggests appropriate audio elements and music for approved scripts

### 🎨 Social Banner Design Workflow
- **Manager Agent** - Orchestrates the workflow and handles all user interactions
- **Graphic Designer Agent** - Creates banner designs by selecting from available image resources
- **Graphic Evaluator Agent** - Evaluates banner designs using **OpenAI Vision API** to detect cats in actual image content

## ✨ Key Features

### 🤖 Multi-Agent Collaboration
- **Group Chat Orchestration**: Agents collaborate in a structured conversation flow
- **Specialized Roles**: Each agent has distinct expertise and responsibilities
- **Intelligent Coordination**: Manager agent ensures proper workflow execution

### 💾 Conversation Management
- **Pause & Resume**: Save conversations and resume them at any time
- **State Persistence**: All conversation context and workflow state is preserved
- **Conversation History**: Complete audit trail of all interactions

### 🔄 Workflow Management
- **Approval Process**: User approval required before proceeding to audio selection
- **Phase Tracking**: Clear visibility into current workflow status
- **Error Handling**: Robust error handling and recovery mechanisms

### 🎨 User-Friendly Interface
- **Rich CLI Interface**: Beautiful console interface with colors and tables
- **Interactive Menus**: Easy navigation through different options
- **Progress Indicators**: Visual feedback during agent processing

### ✅ Flexible Approval Workflows
- **Configurable Approval Points**: Control which agents require user approval before proceeding
- **Task-Specific Rules**: Different workflow types can have different approval requirements
- **Human-in-the-Loop**: Ensure critical decisions get human oversight when needed
- **Automated Workflows**: Skip approval for non-critical tasks to enable full automation

### 🔍 AI-Powered Visual Content Analysis (Banner Design)
- **OpenAI Vision API**: Uses GPT-4 with vision capabilities to analyze actual image content
- **Text-Based Analysis**: Converts visual content to detailed text descriptions, then analyzes the text
- **Zero Filename Dependency**: Completely ignores filenames and analyzes actual visual content
- **Smart Content Detection**: AI detects specific visual elements (e.g., cats) with high accuracy
- **Confidence Scoring**: Provides confidence levels based on AI's explicit content identification
- **Detailed Reasoning**: Includes AI's visual description in evaluation feedback

## 🏗️ Architecture

### Multi-Agent System

#### Video Script Creation Workflow
```
User Request
     ↓
Manager Agent (Orchestrator)
     ↓
Script Writer Agent → Creates Script → User Approval
     ↓
Audio Selector Agent → Suggests Audio → Complete
```

#### Social Banner Design Workflow
```
User Request
     ↓
Manager Agent (Orchestrator)
     ↓
Graphic Designer Agent → Selects Banner → Send to Evaluator
     ↓
Graphic Evaluator Agent → OpenAI Vision API → Text Analysis → Cat Detection
     ↓
If Approved: Present to User | If Rejected: Try Again
```

### Technical Stack
- **Semantic Kernel**: Microsoft's AI orchestration framework
- **Group Chat Orchestration**: Advanced multi-agent coordination pattern
- **Python**: Main programming language
- **Rich**: Beautiful CLI interface
- **JSON Serialization**: State persistence
- **OpenAI API**: GPT-4 with vision capabilities for image analysis
- **Pillow (PIL)**: Image encoding and basic processing utilities

## 🏗️ Clean Architecture

### Generic vs Task-Specific Components

The system follows a **clean separation** between generic infrastructure and task-specific logic:

#### **Generic Components** (`main.py`)
- **`MultiAgentCLI`**: Generic command-line interface that works with any workflow
- **UI Management**: Menu, panels, and interaction handling
- **Conversation Management**: Save, load, pause, resume functionality
- **Session Orchestration**: Starting/stopping agent sessions

#### **Task-Specific Components** (`script_group_chat_manager.py`)
- **Workflow Configuration**: Agent roles, approval rules, sequences
- **UI Customization**: Colors, text, prompts specific to video scripts
- **Completion Detection**: Logic to determine when video script workflow is done
- **Delegation Intelligence**: Smart routing based on video script context

#### **Creating New Workflows**
To create a different workflow (e.g., blog writing, code review):

1. **Create new workflow manager**:
```python
class BlogGroupChatManager(MultiAgentGroupChatManager):
    def _setup_blog_workflow(self):
        # Configure for blog writing
    
    def get_workflow_display_info(self):
        # Blog-specific UI text
    
    def is_workflow_complete(self):
        # Blog-specific completion logic
```

2. **Update one line in main.py**:
```python
# Change this line:
self.workflow_manager = ScriptGroupChatManager(manager_agent, specialist_agents)
# To this:
self.workflow_manager = BlogGroupChatManager(manager_agent, specialist_agents)
```

3. **Done!** The entire UI automatically adapts to the new workflow.

### Benefits of This Architecture
- **Reusability**: `main.py` works with any workflow type
- **Maintainability**: Task logic is isolated in workflow managers
- **Extensibility**: Easy to create new workflow types
- **Testability**: Components can be tested independently

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- OpenAI API key OR Azure OpenAI endpoint (for full implementation)
- pip package manager

### Quick Demo (No API Keys Required)

Want to see the concept in action without setting up API keys? Try the demo:

```bash
git clone <repository-url>
cd semanticKernel_POC
pip install -r requirements.txt
python demo.py
```

The demo simulates the multi-agent workflow and shows you exactly how the system works!

### Full Implementation Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd semanticKernel_POC
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**:
   Edit `api_keys_config.env` and add your credentials:
   ```bash
   # For OpenAI
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_ORG_ID=your_organization_id_here
   
   # OR for Azure OpenAI
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   ```

4. **Run the full application**:
   ```bash
   python main.py
   ```

## 📖 Usage Guide

### Demo Mode

The demo (`python demo.py`) showcases:
- Complete multi-agent workflow simulation
- Agent role specialization and coordination
- Conversation state management
- Pause/resume functionality simulation
- Beautiful CLI interface with Rich

**Try these demo commands:**
1. `"Create a video script about AI technology"`
2. `"Yes, I approve the script"` (after script is created)
3. `"pause"` to simulate conversation pausing
4. `"quit"` to exit

### Full Implementation

#### Starting a New Workflow

1. Launch the application: `python main.py`
2. **Select Workflow Type**:
   - **Option 1**: Video Script Creation
   - **Option 2**: Social Banner Design
3. **For Video Script Creation**:
   - Describe your video script requirements
   - Review and approve the script when presented
   - Receive audio suggestions after approval
4. **For Banner Design**:
   - Describe your desired banner (purpose, style, audience)
   - Watch as the designer selects banners and evaluator analyzes them
   - Get approved banners (containing cats) or see rejections with feedback

#### Pausing & Resuming

**To Pause a Conversation:**
- Type `pause` during any interaction
- The conversation state is automatically saved
- You can safely exit the application

**To Resume a Conversation:**
- Select option **2** - "Resume paused conversation"
- Choose from the list of saved conversations
- Continue from exactly where you left off

#### Managing Conversations

- **List Conversations**: View all saved conversations with status
- **View Summary**: Get detailed summary of any conversation
- **Delete Conversations**: Remove unwanted saved conversations

## 🔧 Configuration

### Approval Workflow Configuration

You can configure different approval requirements for different workflow types. This allows the same multi-agent framework to support both human-in-the-loop and fully automated workflows.

#### Video Script Creation (Approval Required)
```python
workflow_config = {
    "requires_approval_after": ["ScriptWriter"],  # User must approve script
    "auto_proceed_agents": [],
    "workflow_sequence": ["ScriptWriter", "AudioSelector"]
}
```

#### Blog Writing (Fully Automated)
```python
workflow_config = {
    "requires_approval_after": [],  # No approval needed
    "auto_proceed_agents": ["ContentWriter", "SEOOptimizer"],
    "workflow_sequence": ["ContentWriter", "SEOOptimizer"]
}
```

#### Code Review (Partial Approval)
```python
workflow_config = {
    "requires_approval_after": ["SecurityAnalyzer"],  # Only security needs approval
    "auto_proceed_agents": ["CodeAnalyzer"],
    "workflow_sequence": ["CodeAnalyzer", "SecurityAnalyzer", "PerformanceAnalyzer"]
}
```

#### Configuration Options
- **`requires_approval_after`**: List of agent names that require user approval after their work
- **`auto_proceed_agents`**: List of agent names that can automatically proceed to invoke next agents
- **`workflow_sequence`**: Optional list defining the typical order of agent involvement

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes (if not using Azure) |
| `OPENAI_ORG_ID` | OpenAI organization ID | No |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Yes (if not using OpenAI) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes (if using Azure) |
| `CHAT_MODEL_ID` | Model to use (default: gpt-4o-mini) | No |

### Model Configuration
The system is configured to use `gpt-4o-mini` by default for cost-effectiveness. You can change this in the environment configuration.

## 🎭 Agent Roles

### Manager Agent
- **Primary Responsibility**: User interaction and workflow coordination
- **Key Functions**:
  - Analyzes user requests
  - Delegates tasks to appropriate agents
  - Manages approval workflows
  - Handles conversation pausing/resuming

### Script Writer Agent
- **Primary Responsibility**: Creative video script creation
- **Key Functions**:
  - Creates engaging video scripts
  - Adapts style based on video purpose
  - Includes visual cues and scene descriptions
  - Handles script revisions

### Audio Selector Agent
- **Primary Responsibility**: Audio element recommendations
- **Key Functions**:
  - Suggests background music
  - Recommends sound effects
  - Provides licensing information
  - Considers platform-specific requirements

## 📁 Project Structure

```
semanticKernel_POC/
├── demo.py                              # Interactive demo (no API keys needed)
├── main.py                              # Main application entry point
├── requirements.txt                     # Python dependencies
├── api_keys_config.env                 # API configuration
├── README.md                           # This file
├── src/
│   ├── agents/                         # Agent implementations
│   │   ├── base_agent.py              # Base agent class
│   │   ├── manager_agent.py           # Manager agent
│   │   ├── script_writer_agent.py     # Script writer agent
│   │   └── audio_selector_agent.py    # Audio selector agent
│   ├── orchestration/                  # Group chat orchestration
│   │   └── group_chat_manager.py      # Main orchestration logic
│   └── serialization/                  # Conversation persistence
│       └── conversation_serializer.py # Save/load functionality
└── saved_conversations/                # Saved conversation files
```

## 🛠️ Technical Details

### Group Chat Orchestration
The system uses Semantic Kernel's Group Chat Orchestration pattern, which provides:
- **Intelligent Agent Selection**: Automatic routing to appropriate agents
- **Conversation Flow Control**: Structured dialogue management
- **Context Preservation**: Shared conversation context across agents

### State Management
- **Workflow Phases**: Tracks current state (initial, script_creation, awaiting_approval, etc.)
- **Conversation History**: Complete record of all interactions
- **Serializable State**: JSON-based persistence for pause/resume

### Error Handling
- **Graceful Degradation**: Fallback responses when agents fail
- **User Feedback**: Clear error messages and recovery suggestions
- **Session Recovery**: Automatic session restart on resume

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

2. **API Key Issues**:
   - Verify API keys in `api_keys_config.env`
   - Check API key permissions and quotas
   - **Try the demo first**: `python demo.py` works without API keys

3. **Conversation Loading Errors**:
   - Check `saved_conversations/` directory permissions
   - Verify JSON file integrity

### Getting Help
- **Start with the demo**: `python demo.py` to understand the workflow
- Check the conversation summary for debugging context
- Review error messages in the console output
- Ensure API endpoints are accessible

## 🎯 Demo vs Full Implementation

| Feature | Demo Mode | Full Implementation |
|---------|-----------|-------------------|
| **Agent Intelligence** | Pre-scripted responses | Real LLM-powered agents |
| **API Requirements** | None | OpenAI/Azure OpenAI API |
| **Conversation Flow** | Simulated orchestration | Semantic Kernel Group Chat |
| **Script Quality** | Template-based | AI-generated and creative |
| **Audio Suggestions** | Static recommendations | Context-aware AI suggestions |
| **Persistence** | Simulated | Real JSON serialization |

## 📈 Future Enhancements

- **Web Interface**: React-based frontend for better user experience
- **Voice Integration**: Voice-to-text input and text-to-speech output
- **Advanced Audio AI**: Integration with audio generation models
- **Template Library**: Pre-built script templates for common video types
- **Collaboration Features**: Multi-user workflow support
- **Real-time Streaming**: Live agent responses with streaming
- **Integration APIs**: REST endpoints for external integrations

## 🐛 Recent Bug Fixes

### Clean Architecture Refactoring (v1.2)
**Issue**: `main.py` contained hardcoded video script-specific logic, making it impossible to reuse for other workflow types.

**Problems**:
- Hardcoded workflow configuration (agent roles, approval rules)
- Hardcoded UI text ("video script creation")
- Hardcoded completion detection logic (AudioSelector-specific)
- Hardcoded agent color mapping

**Solution**:
- Moved all task-specific logic to `ScriptGroupChatManager`
- Made `main.py` completely generic (`MultiAgentCLI`)
- Added workflow methods: `get_workflow_display_info()`, `get_agent_colors()`, `is_workflow_complete()`
- Created clean separation between infrastructure and business logic

**Result**: `main.py` now works with any workflow type. Creating new workflows (blog writing, code review, etc.) requires only a new workflow manager class.

### Conversation Isolation Fix (v1.1)
**Issue**: Saved conversation files contained chat history from multiple different conversations mixed together.

**Root Cause**: 
1. Workflow state was not being reset between new conversations, causing history accumulation
2. Double-nesting in serialization created `workflow_state.workflow_state.conversation_history` structure

**Fix Applied**:
- Added `reset_workflow_state()` method to clear conversation history when starting new conversations
- Simplified serialization structure to prevent double-nesting
- Updated resume logic to handle the cleaner data structure

**Result**: Each conversation now maintains completely isolated history, and saved files have a clean structure.

## 🤝 Contributing

This is a POC project demonstrating Semantic Kernel capabilities. For suggestions or improvements:
1. Try the demo first to understand the concept
2. Create detailed issue descriptions
3. Provide clear reproduction steps
4. Include environment details

## 📝 License

This project is created for demonstration purposes. Please ensure compliance with OpenAI/Azure OpenAI terms of service when using the APIs.

---

**Built with ❤️ using Semantic Kernel and Python**

*Start with `python demo.py` to see the magic in action!* 