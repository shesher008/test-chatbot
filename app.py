import queue
import uuid
import streamlit as st
import sys
import os
import asyncio

# Set environment variables from Streamlit secrets
if hasattr(st, 'secrets'):
    os.environ["NVIDIA_API_KEY"] = st.secrets.get("NVIDIA_API_KEY", "")
    os.environ["DATABASE_URL"] = st.secrets.get("DATABASE_URL", "")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define model descriptions and colors at the TOP
MODEL_DESCRIPTIONS = {
    "coding": "Optimized for programming & technical queries",
    "chatting": "Optimized for general conversation",
    "pentesting": "Optimized for cybersecurity topics",
    "math": "Optimized for mathematical calculations",
    "creative": "Optimized for creative writing",
    "default": "General purpose AI assistant"
}

MODEL_COLORS = {
    "coding": "#00ff88",
    "chatting": "#ffaa00",
    "pentesting": "#ff4444",
    "math": "#4488ff",
    "creative": "#ff44ff",
    "default": "#888888"
}

# =========================== Initialize Session State ===========================
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "current_model" not in st.session_state:
    st.session_state.current_model = "default"

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = [st.session_state.thread_id]

if "backend_loaded" not in st.session_state:
    st.session_state.backend_loaded = False

# =========================== Import Backend (with new sync function) ===========================
try:
    # Updated import list to include the new sync wrapper
    from backend import chatbot, retrieve_all_threads, submit_async_task, get_available_models, switch_model_sync 
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    BACKEND_LOADED = True
    st.session_state.backend_loaded = True
except Exception as e:
    BACKEND_LOADED = False
    st.session_state.backend_loaded = False
    # Don't show error yet - handled in UI

# =========================== Helper Functions ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    st.session_state.message_history = []
    st.session_state.current_model = "default"
    # Only add to list if not already there (new ID)
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)
    st.rerun()

def load_previous_chat(thread_id):
    st.session_state.thread_id = thread_id
    st.session_state.message_history = []
    
    if BACKEND_LOADED:
        try:
            # Use get_state to load the history and current model from the checkpointer
            state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
            messages = state.values.get("messages", [])
            
            # --- Load Messages ---
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    st.session_state.message_history.append({
                        "role": "user",
                        "content": msg.content
                    })
                elif isinstance(msg, AIMessage):
                    st.session_state.message_history.append({
                        "role": "assistant",
                        "content": msg.content
                    })
            
            # --- Load Model ---
            current_model = state.values.get("current_model", "default")
            st.session_state.current_model = current_model
            
            # Ensure the loaded thread is at the top of the chat list for visibility
            if thread_id in st.session_state.chat_threads:
                st.session_state.chat_threads.remove(thread_id)
            st.session_state.chat_threads.append(thread_id)

        except Exception as e:
            st.error(f"Could not load chat: {e}")
    
    st.rerun()

# =========================== Sidebar ===========================
with st.sidebar:
    st.title("🤖 Multi-Model Chatbot")
    
    # Backend Status
    if st.session_state.backend_loaded:
        st.success("✅ Backend Ready")
    else:
        st.error("⚠️ Backend Not Loaded")
        st.info("Make sure backend.py runs without errors")
    
    st.divider()
    
    # Model Selection
    st.subheader("Select Model")
    
    # Use the keys from the descriptions to guarantee order
    for model in list(MODEL_DESCRIPTIONS.keys()):
        is_selected = st.session_state.current_model == model
        button_label = f"✅ {model.capitalize()}" if is_selected else f"🔄 {model.capitalize()}"
        
        if st.button(
            button_label,
            key=f"model_{model}",
            use_container_width=True,
            type="primary" if is_selected else "secondary"
        ):
            # Only switch if not already selected
            if not is_selected:
                st.session_state.current_model = model
                
                if BACKEND_LOADED:
                    try:
                        # --- CRITICAL FIX: Use the synchronous wrapper ---
                        result = switch_model_sync(st.session_state.thread_id, model)
                        # --------------------------------------------------
                        
                        if result.get("success"):
                            st.success(f"✅ Switched to {model} model! (Change persisted in DB)")
                        else:
                            st.warning(f"Note: {result.get('error', 'Could not switch in backend')}")
                    except Exception as e:
                        st.warning(f"Note: Could not update backend: {str(e)}")
                
                st.rerun()
    
    # Current Model Indicator
    current_model = st.session_state.current_model
    model_color = MODEL_COLORS.get(current_model, "#888888")
    st.markdown(f"""
    **Current Model:** <span style='color:{model_color}; font-weight:bold; font-size:18px'>
    {current_model.upper()}
    </span>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Chat Management
    if st.button("🆕 New Chat", use_container_width=True, type="primary"):
        reset_chat()
    
    # Load Previous Conversations
    if BACKEND_LOADED and "chat_threads_loaded" not in st.session_state:
        try:
            # Load threads from the backend checkpointer
            backend_threads = retrieve_all_threads()
            if backend_threads:
                # Use backend threads but ensure the current one is present
                st.session_state.chat_threads = list(set(backend_threads + [st.session_state.thread_id]))
            st.session_state.chat_threads_loaded = True
        except Exception as e:
            st.warning(f"Could not load conversations: {e}")
    
    # Previous Chats
    st.subheader("💬 Previous Chats")
    
    # Sort threads for display: current thread first, then newest to oldest
    current_thread = st.session_state.thread_id
    threads_to_display = sorted(
        [t for t in st.session_state.chat_threads if t != current_thread],
        reverse=True
    )
    # Add current thread to the top
    threads_to_display = [current_thread] + threads_to_display

    for idx, thread_id in enumerate(threads_to_display):
        # Create a more user-friendly label
        display_label = "Current Chat" if thread_id == current_thread else f"Chat {len(threads_to_display) - idx}"
        
        if st.button(
            display_label, 
            key=f"load_{thread_id}", 
            use_container_width=True,
            type="primary" if thread_id == current_thread else "secondary"
        ):
            if thread_id != current_thread:
                load_previous_chat(thread_id)

# =========================== Main Chat Interface ===========================
st.title("💬 Multi-Model Chatbot")

# Backend Status Warning
if not BACKEND_LOADED:
    st.error("""
    ⚠️ **Backend Not Loaded!**
    
    Please ensure:
    1. `backend.py` runs without errors
    2. NVIDIA_API_KEY and DATABASE_URL are set in `.env` file
    3. All dependencies are installed
    
    Running in demo mode...
    """)

# Model Indicator
current_model = st.session_state.current_model
model_color = MODEL_COLORS.get(current_model, "#888888")
description = MODEL_DESCRIPTIONS.get(current_model, "General purpose tasks")

st.markdown(f"""
<div style='background-color:{model_color}20; padding:12px; border-radius:10px; border-left:5px solid {model_color}; margin-bottom:20px'>
    <h4 style='margin:0; color:{model_color}'>
    🤖 Using: <strong>{current_model.upper()}</strong> Model
    </h4>
    <p style='margin:5px 0 0 0; font-size:14px; color:#666'>
    {description}
    </p>
</div>
""", unsafe_allow_html=True)

# Display Chat History
for message in st.session_state.message_history:
    # Filter out empty content messages that can sometimes appear in state
    if message["content"].strip():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat Input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add to history
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Prepare for assistant response
    with st.chat_message("assistant"):
        if not BACKEND_LOADED:
            # Demo mode
            demo_response = f"I received: '{user_input}' (Running in demo mode)"
            st.markdown(demo_response)
            st.session_state.message_history.append({
                "role": "assistant",
                "content": demo_response
            })
        else:
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Configuration for the chatbot
                CONFIG = {
                    "configurable": {"thread_id": st.session_state.thread_id},
                    "metadata": {"thread_id": st.session_state.thread_id},
                    "run_name": "chat_turn",
                }
                
                # Function to stream chatbot response
                def stream_chatbot():
                    event_queue = queue.Queue()
                    
                    async def run_async():
                        try:
                            # Stream the new user message through the LangGraph
                            async for event in chatbot.astream(
                                {"messages": [HumanMessage(content=user_input)]},
                                config=CONFIG,
                                stream_mode="messages",
                            ):
                                event_queue.put(event)
                        except Exception as e:
                            event_queue.put(("error", e))
                        finally:
                            event_queue.put(None)
                    
                    # Start the async task using the backend's dedicated loop
                    submit_async_task(run_async())
                    
                    while True:
                        event = event_queue.get()
                        if event is None:
                            break
                        
                        if isinstance(event, tuple) and event[0] == "error":
                            raise event[1]
                        
                        # Process the streamed event/message chunk
                        if isinstance(event, tuple):
                            message_chunk, _ = event
                            
                            # Update model if the state has changed (e.g., after tool use)
                            if "current_model" in event[1]:
                                st.session_state.current_model = event[1]["current_model"]
                                
                            if isinstance(message_chunk, AIMessage) and message_chunk.content:
                                yield message_chunk.content
                
                # Display the streaming response
                try:
                    for chunk in stream_chatbot():
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    
                    # Final display after streaming is complete
                    message_placeholder.markdown(full_response)
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Handle other errors (removed recursion limit check)
                    st.error(f"Error: {error_msg}")
                    full_response = f"Sorry, an error occurred: {error_msg}"
                    message_placeholder.markdown(full_response)
                
            except Exception as e:
                st.error(f"Chatbot error: {str(e)}")
                full_response = f"Sorry, an error occurred: {str(e)}"
                message_placeholder.markdown(full_response)
            
            # Add assistant response to history
            if full_response:
                st.session_state.message_history.append({
                    "role": "assistant",
                    "content": full_response
                })
    
    # Rerun to clear the input box and update the model indicator if the model was switched
    st.rerun()