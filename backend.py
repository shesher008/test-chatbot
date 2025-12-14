from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages
from langchain_core.tools import Tool,tool, BaseTool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
import asyncio
import threading
import os
import sys
import atexit

# Add this at the VERY TOP of backend.py
import os
import sys

# Try to load from Streamlit secrets first (for deployment)
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        # Override environment variables from Streamlit secrets
        if "NVIDIA_API_KEY" in st.secrets:
            os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]
        if "DATABASE_URL" in st.secrets:
            os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
        print("✅ Loaded secrets from Streamlit")
except ImportError:
    pass  # Streamlit not available, fall back to .env
except Exception as e:
    print(f"⚠️ Could not load Streamlit secrets: {e}")

# ==================== 1. CRITICAL INITIALIZATION STEPS ====================

# WINDOWS FIX - Alternative approach
if sys.platform == "win32":
    try:
        from asyncio import WindowsSelectorEventLoopPolicy
        # Only set if not already set
        if not isinstance(asyncio.get_event_loop_policy(), WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
            print("✅ Windows async policy set for database compatibility.")
    except ImportError:
        pass

# Load environment variables (CRITICAL: Must be at the top)
load_dotenv()

# ==================== 2. Async Loop Setup ====================
# Dedicated async loop for backend tasks (CRITICAL for LangGraph/Streamlit interaction)
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    """Submits a coroutine to the async thread and waits for the result (Synchronous Block)."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    """A wrapper to run a coroutine synchronously."""
    return _submit_async(coro).result()

def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop (Non-Blocking)."""
    # Simply submits the task, the user must handle the result/future object
    return _submit_async(coro) 

# ==================== 3. MULTIPLE NIM MODELS ====================
class LLMManager:
    """Manager for multiple NVIDIA NIM models"""
    
    def __init__(self):
        if not os.getenv("NVIDIA_API_KEY"):
            print("⚠️ Warning: NVIDIA_API_KEY not found in .env file")
        
        # Initialize models only once
        self.models = {
            "coding": ChatNVIDIA(model="qwen/qwen3-235b-a22b", temperature=0.2, top_p=0.9),
            "chatting": ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.7, top_p=0.95),
            "pentesting": ChatNVIDIA(model="qwen/qwen3-235b-a22b", temperature=0.3, top_p=0.9),
            "math": ChatNVIDIA(model="qwen/qwen3-next-80b-a3b-thinking", temperature=0.1, top_p=0.8),
            "creative": ChatNVIDIA(model="meta/llama-3.1-405b-instruct", temperature=0.8, top_p=0.95),
            "default": ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0.5, top_p=0.9)
        }
    
    def detect_model_type(self, user_input: str) -> str:
        """Detect which model to use based on user input (simplified logic)"""
        input_lower = user_input.lower()
        
        if "code" in input_lower or "program" in input_lower:
            return "coding"
        if "hack" in input_lower or "security" in input_lower:
            return "pentesting"
        if "calculate" in input_lower or "solve" in input_lower or "equation" in input_lower:
            return "math"
        if "write" in input_lower or "story" in input_lower or "poem" in input_lower:
            return "creative"
        
        return "default"

# Initialize LLM Manager
llm_manager = LLMManager()

# ==================== 4. Tools ====================
@tool
def select_model(model_type: str) -> str:
    """
    Select which LLM model to use. 
    Available: coding, chatting, pentesting, math, creative, default
    The execution of this tool will update the model used for the next chat turn.
    """
    available_models = list(llm_manager.models.keys())
    if model_type not in available_models:
        return f"Error: Model '{model_type}' not found. Available: {', '.join(available_models)}"
    # NOTE: The actual state update happens in execute_tools
    return f"Success: Request to switch to {model_type} model submitted."

# DuckDuckGo search tool
@tool
def duckduckgo_search(query: str) -> str:
    """
    Perform an internet search using DuckDuckGo and return the results.
    
    This function utilizes the DuckDuckGoSearchAPIWrapper to execute web searches
    without tracking or personalized filtering. It's particularly useful for
    obtaining factual information, current events, and general web content.
    
    Args:
        query (str): The search query string. Can include keywords, questions,
                     or search terms. For best results, be specific and concise.
    
    Returns:
        str: A string containing the search results summary. The format typically
             includes relevant web snippets, titles, and brief descriptions.
    
    Example:
        >>> results = duckduckgo_search("Python programming tutorials")
        >>> print(results[:200])  # First 200 characters of results
    
    Notes:
        - Results are not personalized (unlike some search engines)
        - No search history is tracked
        - Rate limits may apply with extensive use
        - Results quality may vary based on query specificity
    """
    search = DuckDuckGoSearchAPIWrapper()
    return search.run(query)


# Add tools to the list
tools = [select_model, duckduckgo_search]

# ==================== 5. State ====================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_model: str

# ==================== 6. Graph Nodes ====================
async def chat_node(state: ChatState):
    """Main chat node with NO recursion limit"""
    messages = state["messages"]
    current_model = state.get("current_model", "default")
    
    last_user_message = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    
    # Auto-detect model only if it's the start of a conversation and the model is default
    if last_user_message and current_model == "default":
        current_model = llm_manager.detect_model_type(last_user_message)
    
    selected_llm = llm_manager.models.get(current_model, llm_manager.models["default"])
    llm_with_tools = selected_llm.bind_tools(tools)
    
    system_content = f"""You are a helpful and specialized AI assistant using the {current_model} model.
IMPORTANT RULES:
1. Always use the `{current_model.upper()}` model's specialization when answering.
2. Only use the `select_model` tool if the user **explicitly** asks to change the AI's specialty (e.g., "switch to coding model").
3. After generating a response or a tool call, STOP.
4. Keep responses concise unless a detailed answer is necessary.
Current question: {last_user_message or 'No question'}
"""
    system_message = SystemMessage(content=system_content)
    enhanced_messages = [system_message] + messages[-4:]
    
    response = await llm_with_tools.ainvoke(enhanced_messages)
    
    return {
        "messages": [response],
        "current_model": current_model
    }

async def execute_tools(state: ChatState):
    """Execute tool calls and update state if model is switched."""
    messages = state["messages"]
    current_model = state.get("current_model", "default")
    
    if not messages:
        return {"messages": [], "current_model": current_model}
    
    last_message = messages[-1]
    results = []
    new_model = current_model  # Start with current model
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call.get('args', {})
            
            for tool_func in tools:
                if tool_func.name == tool_name:
                    try:
                        result = tool_func.invoke(tool_args)
                        
                        # Better model switching logic
                        if tool_name == "select_model":
                            requested_model = tool_args.get("model_type", "default")
                            if requested_model in llm_manager.models:
                                new_model = requested_model
                                result = f"Model switched to {requested_model}"
                            else:
                                result = f"Error: Model '{requested_model}' not found"
                        
                        results.append({
                            "tool_call_id": tool_call.get('id', ''),
                            "tool_name": tool_name,
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "tool_call_id": tool_call.get('id', ''),
                            "tool_name": tool_name,
                            "result": f"Error: {str(e)}"
                        })

    tool_messages = []
    if results:
        for res in results:
            tool_messages.append(ToolMessage(
                content=str(res["result"]),
                tool_call_id=res["tool_call_id"],
                name=res["tool_name"]
            ))
        
        # Return state update: Messages and the NEW model if switched
        return {
            "messages": tool_messages,
            "current_model": new_model # <<-- Persists the change
        }
    
    return {"messages": [], "current_model": current_model}

def should_continue(state: ChatState) -> Literal["tools", END]:
    """Determine whether to use tools or end - NO RECURSION LIMIT"""
    messages = state["messages"]
    
    if not messages:
        return END
    
    last_message = messages[-1]
    
    # 1. If the last message contains tool calls, go to the tool node.
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # 2. If the last message contains content (an answer), we are done.
    if hasattr(last_message, 'content') and last_message.content:
        return END
    
    return END

# ==================== 7. Checkpointer Initialization ====================
# Global variable to hold the checkpointer instance
checkpointer = None

async def _init_checkpointer():
    """Initialize PostgreSQL connection and Checkpointer with explicit schema."""
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("❌ CRITICAL: DATABASE_URL not found in .env file.", file=sys.stderr)
        raise ValueError("DATABASE_URL environment variable is required.")
    
    print("🔗 Initializing Neon PostgreSQL Checkpointer...")
    
    try:
        # 1. Create the saver with explicit table configurations
        saver = AsyncPostgresSaver.from_conn_string(
            database_url,
            # Explicit table names (optional but good practice)
            checkpoint_table_name="langgraph_checkpoints",
            metadata_table_name="langgraph_metadata",
            # Optional: Custom schema
            # schema_name="langgraph_schema"
        )
        
        # 2. Initialize database schema with explicit setup
        await saver.setup()
        print("✅ Neon PostgreSQL database connected successfully!")
        
        # 3. Verify the expected columns exist
        # LangGraph typically creates these columns automatically:
        # - thread_id: VARCHAR (or TEXT)
        # - checkpoint: JSONB (for storing state)
        # - checkpoint_id: SERIAL/BIGSERIAL (auto-incrementing ID)
        # - parent_checkpoint_id: BIGINT (for checkpoint chains)
        # - created_at: TIMESTAMP
        # - metadata: JSONB (additional metadata)
        
        # For your custom state field (current_model), it will be stored
        # within the 'checkpoint' JSONB column as part of the state object
        
        return saver
        
    except Exception as e:
        print(f"❌ Error connecting to database: {e}", file=sys.stderr)
        raise

# Run the async initialization synchronously
try:
    checkpointer = run_async(_init_checkpointer())
except Exception as e:
    print(f"⚠️ Checkpointer initialization failed. Error: {e}. Chat history will NOT be persistent.")
    checkpointer = None


# ==================== 8. Build Graph ====================
graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("tools", execute_tools)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", should_continue)
graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer=checkpointer)

# ==================== 9. Helper Functions (Exposed to Frontend) ====================

async def _alist_threads():
    """Asynchronously retrieve all thread IDs from the checkpointer."""
    if not checkpointer:
        return []
    all_threads = set()
    try:
        # Check if the checkpointer is active before trying to use it
        async for checkpoint in checkpointer.alist(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
    except Exception as e:
        # The 'connection is closed' error should now be caught here
        print(f"Error retrieving threads: {e}") 
        return []
    return list(all_threads)

def retrieve_all_threads():
    """Synchronous wrapper for retrieving all thread IDs."""
    return run_async(_alist_threads())

def get_available_models() -> list:
    return list(llm_manager.models.keys())

async def switch_model_async(thread_id: str, model_type: str) -> dict:
    """Asynchronously switch and save the model type for a given thread."""
    if not checkpointer:
        return {"success": False, "error": "Backend is running without a checkpointer (DB error). Cannot persist model change."}
        
    if model_type not in llm_manager.models:
        return {"success": False, "error": f"Model {model_type} not available"}
    
    try:
        # FIX: Use await for async state retrieval
        state = await chatbot.aget_state(config={"configurable": {"thread_id": thread_id}})
        
        # Update the model in the state dictionary
        state.values["current_model"] = model_type
        
        # FIX: Save the complete state - CORRECT VERSION
        # LangGraph's checkpointer expects specific parameters
        await checkpointer.aput(
            config={"configurable": {"thread_id": thread_id}},
            # The checkpoint should be a dict with 'config' and 'values'
            checkpoint={
                "config": state.config,
                "values": state.values,
                "next": ("chat",)  # Specify where to continue
            },
            metadata={"source": "manual_switch"}
        )
        
        return {"success": True, "new_model": model_type}
    except Exception as e:
        return {"success": False, "error": f"Could not update state: {str(e)}"}

def switch_model_sync(thread_id: str, model_type: str) -> dict:
    """Synchronous wrapper for switch_model_async."""
    return run_async(switch_model_async(thread_id, model_type))

# Add at the end of the file (before __main__ check)
def cleanup():
    """Cleanup function for database connections."""
    if _ASYNC_LOOP.is_running():
        # Cancel any pending tasks
        for task in asyncio.all_tasks(_ASYNC_LOOP):
            task.cancel()
        _ASYNC_LOOP.call_soon_threadsafe(_ASYNC_LOOP.stop)

# Register cleanup
atexit.register(cleanup)

# ==================== 10. Initialization Check ====================
if __name__ == "__main__":
    print("=" * 50)
    print("🤖 Multi-Model Chatbot Backend")
    print("=" * 50)
    
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key and nvidia_key.startswith("nvapi-"):
        print("✅ NVIDIA_API_KEY is set")
    else:
        print("⚠️ NVIDIA_API_KEY may not be properly configured")
    
    print(f"📊 Available models: {get_available_models()}")
    print(f"🔧 Available tools: {[tool.name for tool in tools]}")
    print("🎯 Backend is ready. Run 'streamlit run frontend.py' to start.")
    print("=" * 50)