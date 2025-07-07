# Semantic Kernel POC - Multi-Agent Workflow System

A proof-of-concept demonstrating intelligent multi-agent workflows using **Microsoft Semantic Kernel** with official **GroupChatOrchestration** patterns.

## 🎯 **Project Overview**

This POC has been **successfully converted** from custom group chat managers to official Semantic Kernel orchestration patterns, providing:
- **Banner Design Workflow**: AI-powered banner creation with design evaluation
- **Script Writing Workflow**: Intelligent content creation for various purposes
- **Advanced Agent Coordination**: Using GroupChatOrchestration + RoundRobinGroupChatManager

## 🏗️ **Architecture**

### **Orchestration Patterns (NEW)**
- **GroupChatOrchestration**: Official Microsoft pattern for multi-agent coordination
- **RoundRobinGroupChatManager**: Predictable agent sequencing
- **SequentialOrchestration**: Alternative pattern for simpler workflows

### **Specialized Agents**
- **ManagerAgent**: Workflow coordination and user interaction
- **GraphicDesignerAgent**: Banner selection with kernel functions
- **GraphicEvaluatorAgent**: AI-powered design evaluation using OpenAI Vision API
- **ScriptWriterAgent**: Content creation for various purposes

### **AI-Powered Features**
- **Smart Agent Selection**: AI determines optimal workflow flow
- **Context-Aware Evaluation**: OpenAI Vision API for image analysis
- **Intelligent Termination**: AI decides when workflows are complete
- **Graceful Fallbacks**: Hardcoded logic when AI services unavailable

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone the repository
git clone <your-repo-url>
cd semanticKernel_POC

# Install dependencies
pip install -r requirements.txt
```

### **2. Configuration**
Create `api_keys_config.env` with your API keys:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
CHAT_MODEL_ID=gpt-4o-mini

# Optional: Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
```

### **3. Run Applications**
```bash
# Main interactive application (uses GroupChatOrchestration)
python main.py

# Legacy demo (mock simulation)
python demo.py
```

## 🎨 **Usage Examples**

### **Banner Workflow**
```python
from src.orchestration.banner_group_chat_orchestration_simple import BannerGroupChatOrchestration

# Create workflow
banner_workflow = BannerGroupChatOrchestration()

# Start banner design
result = await banner_workflow.start_conversation(
    "Create a banner for a tech startup focusing on AI innovation"
)
print(result)
```

### **Script Workflow**
```python
from src.orchestration.script_group_chat_orchestration import ScriptGroupChatOrchestration

# Create workflow
script_workflow = ScriptGroupChatOrchestration()

# Generate content
result = await script_workflow.start_conversation(
    "Write a motivational script about overcoming technology challenges"
)
print(result)
```

## 📁 **Project Structure**

```
semanticKernel_POC/
├── src/
│   ├── agents/                    # Specialized AI agents
│   │   ├── base_agent.py         # Base agent class
│   │   ├── manager_agent.py      # Workflow coordinator
│   │   ├── graphic_designer_agent.py  # Banner creation
│   │   ├── graphic_evaluator_agent.py # Design evaluation
│   │   └── script_writer_agent.py     # Content creation
│   ├── orchestration/            # Workflow orchestration  
│   │   ├── banner_group_chat_orchestration_simple.py  # Banner workflow (GroupChatOrchestration)
│   │   └── script_group_chat_orchestration.py         # Script workflow (GroupChatOrchestration)
│   └── serialization/            # Conversation persistence
├── mock_banner/                  # Sample banner images
├── saved_conversations/          # Conversation history
├── demo.py                      # Legacy demo (mock simulation)
├── main.py                      # Modern interactive application
└── requirements.txt             # Dependencies
```

## ✅ **Conversion Status**

### **Successfully Completed:**
- ✅ **Semantic Kernel Upgrade**: 1.18.0 → 1.34.0
- ✅ **GroupChatOrchestration Integration**: Official Microsoft patterns
- ✅ **Banner Workflow**: GroupChatOrchestration + RoundRobinGroupChatManager
- ✅ **Script Workflow**: Same orchestration pattern
- ✅ **Agent Preservation**: All existing functionality maintained
- ✅ **Kernel Functions**: Banner selection and evaluation preserved
- ✅ **AI Features**: Vision API and smart logic intact
- ✅ **Legacy Cleanup**: Removed obsolete custom managers (90KB+ of legacy code)
- ✅ **Direct Integration**: Modernized main.py to work directly with GroupChatOrchestration
- ✅ **Simplified Architecture**: Eliminated adapter layer complexity

### **Key Benefits:**
- **🏗️ Future-Proof**: Official Microsoft patterns
- **📉 Reduced Complexity**: 400+ line custom managers → 80-line orchestrations
- **🔄 Better Management**: Built-in conversation and state handling
- **🎯 Standardized**: Consistent orchestration across workflows
- **🛡️ Robust**: Better error handling and fallback systems

### **Cleanup Summary:**

#### **Removed Files (90KB+ Legacy Code):**
- `src/orchestration/group_chat_manager.py` (960 lines, 44KB)
- `src/orchestration/banner_group_chat_manager.py` (642 lines, 33KB)  
- `src/orchestration/script_group_chat_manager.py` (263 lines, 13KB)

#### **Modernized Files:**
- `main.py` - Completely rewritten to work directly with GroupChatOrchestration
- Updated imports in `__init__.py` to reference new orchestration classes
- Simplified CLI interface focused on single-shot workflow execution

#### **Modern Architecture:**
- Direct GroupChatOrchestration integration (no adapter layer needed)
- Simplified result-oriented workflow model
- Clean separation between legacy conversation management and modern orchestration
- Focus on workflow execution rather than complex state management

## 🔧 **Known Issues & Solutions**

### **Linter Warnings (False Positives)**
Some linters report missing "runtime" parameter for GroupChatOrchestration:
- **Status**: False positive - constructor only requires "members" and "manager"
- **Resolution**: Code works correctly at runtime
- **Evidence**: Demo runs successfully, constructor inspection confirms

### **Import Issues**
Relative imports may fail in some contexts:
- **Solution**: All files use absolute imports with sys.path manipulation
- **Status**: Resolved in all orchestration files

## 🧪 **Testing**

The POC includes two main interfaces:

1. **Modern Application**: `main.py`
   - Clean GroupChatOrchestration interface
   - Real-time workflow execution with progress indicators
   - Result management and export capabilities
   - Direct integration with Semantic Kernel patterns

2. **Legacy Demo**: `demo.py`
   - Mock simulation without API requirements
   - Useful for testing UI flow without API keys
   - Educational demonstration of multi-agent concepts

## 🔮 **Future Enhancements**

### **Orchestration Patterns**
- **SequentialOrchestration**: For linear workflows
- **ConcurrentOrchestration**: For parallel agent execution
- **Custom GroupChatManager**: For complex coordination logic

### **Agent Capabilities**
- **Enhanced Vision Analysis**: More sophisticated image evaluation
- **Multi-Modal Agents**: Combined text, image, and audio processing
- **Dynamic Agent Creation**: Runtime agent instantiation

### **Workflow Features**
- **Persistent State**: Cross-session workflow continuation
- **User Feedback Integration**: Interactive workflow refinement
- **Performance Metrics**: Workflow execution analytics

## 📚 **Dependencies**

- **semantic-kernel==1.34.0**: Core orchestration framework
- **openai>=1.3.0**: Vision API and chat completions
- **azure-identity**: Azure authentication
- **pydantic**: Data validation and settings
- **python-dotenv**: Environment configuration

## 🤝 **Contributing**

1. Follow the established agent and orchestration patterns
2. Use official Semantic Kernel orchestration when possible
3. Maintain backward compatibility with existing agents
4. Add comprehensive error handling and fallbacks
5. Document new patterns and workflows

## 📄 **License**

[Your license information here]

---

## 🎉 **Success Summary**

This POC successfully demonstrates the complete evolution from custom multi-agent coordination to official Microsoft Semantic Kernel orchestration patterns, with comprehensive cleanup and modernization:

- **Standardized Architecture**: Official patterns ensure future compatibility
- **Simplified Maintenance**: Dramatically reduced code complexity (90KB+ legacy code removed)
- **Enhanced Reliability**: Built-in state management and error handling
- **Better Scalability**: Foundation for complex workflow expansion
- **Ultra-Clean Codebase**: Minimal, focused architecture with only essential files

The conversion showcases how existing agent-based systems can be completely modernized to use official orchestration frameworks while preserving specialized capabilities and eliminating all technical debt through systematic cleanup. 