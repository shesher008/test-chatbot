# app.py - Main Streamlit app
import streamlit as st
import sys
import os
import uuid
import queue

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

# =========================== Try Import Backend ===========================
try:
    # Load secrets from Streamlit secrets first
    if hasattr(st, 'secrets'):
        os.environ["NVIDIA_API_KEY"] = st.secrets.get("NVIDIA_API_KEY", "")
        os.environ["DATABASE_URL"] = st.secrets.get("DATABASE_URL", "")
    
    # Now import backend
    from backend import chatbot, retrieve_all_threads, submit_async_task, get_available_models, switch_model_sync
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    BACKEND_LOADED = True
    st.session_state.backend_loaded = True
except Exception as e:
    BACKEND_LOADED = False
    st.session_state.backend_loaded = False
    st.warning(f"Backend import warning: {e}")

# =========================== Model Configurations ===========================
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

# =========================== Helper Functions ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    st.session_state.message_history = []
    st.session_state.current_model = "default"
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)
    st.rerun()

def load_previous_chat(thread_id):
    st.session_state.thread_id = thread_id
    st.session_state.message_history = []
    
    if BACKEND_LOADED:
        try:
            state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
            messages = state.values.get("messages", [])
            
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
            
            current_model = state.values.get("current_model", "default")
            st.session_state.current_model = current_model
            
            if thread_id in st.session_state.chat_threads:
                st.session_state.chat_threads.remove(thread_id)
            st.session_state.chat_threads.append(thread_id)

        except Exception as e:
            st.error(f"Could not load chat: {e}")
    
    st.rerun()

# =========================== Sidebar ===========================
with st.sidebar:
    st.title("🤖 Multi-Model Chatbot")
    
    if st.session_state.backend_loaded:
        st.success("✅ Backend Ready")
    else:
        st.error("⚠️ Backend Not Loaded")
        st.info("Check Streamlit secrets configuration")
    
    st.divider()
    
    # Model Selection
    st.subheader("Select Model")
    
    for model in list(MODEL_DESCRIPTIONS.keys()):
        is_selected = st.session_state.current_model == model
        button_label = f"✅ {model.capitalize()}" if is_selected else f"🔄 {model.capitalize()}"
        
        if st.button(
            button_label,
            key=f"model_{model}",
            use_container_width=True,
            type="primary" if is_selected else "secondary"
        ):
            if not is_selected:
                st.session_state.current_model = model
                
                if BACKEND_LOADED:
                    try:
                        result = switch_model_sync(st.session_state.thread_id, model)
                        if result.get("success"):
                            st.success(f"✅ Switched to {model} model!")
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
            backend_threads = retrieve_all_threads()
            if backend_threads:
                st.session_state.chat_threads = list(set(backend_threads + [st.session_state.thread_id]))
            st.session_state.chat_threads_loaded = True
        except Exception as e:
            st.warning(f"Could not load conversations: {e}")
    
    # Previous Chats
    st.subheader("💬 Previous Chats")
    
    current_thread = st.session_state.thread_id
    threads_to_display = sorted(
        [t for t in st.session_state.chat_threads if t != current_thread],
        reverse=True
    )
    threads_to_display = [current_thread] + threads_to_display

    for idx, thread_id in enumerate(threads_to_display):
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
    
    Please check:
    1. Streamlit secrets are properly configured
    2. All dependencies are installed
    3. NVIDIA API key is valid
    
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
    if message["content"].strip():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat Input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Prepare for assistant response
    with st.chat_message("assistant"):
        if not BACKEND_LOADED:
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
                CONFIG = {
                    "configurable": {"thread_id": st.session_state.thread_id},
                    "metadata": {"thread_id": st.session_state.thread_id},
                    "run_name": "chat_turn",
                }
                
                def stream_chatbot():
                    event_queue = queue.Queue()
                    
                    async def run_async():
                        try:
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
                    
                    submit_async_task(run_async())
                    
                    while True:
                        event = event_queue.get()
                        if event is None:
                            break
                        
                        if isinstance(event, tuple) and event[0] == "error":
                            raise event[1]
                        
                        if isinstance(event, tuple):
                            message_chunk, _ = event
                            
                            if "current_model" in event[1]:
                                st.session_state.current_model = event[1]["current_model"]
                            
                            if isinstance(message_chunk, AIMessage) and message_chunk.content:
                                yield message_chunk.content
                
                try:
                    for chunk in stream_chatbot():
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"Error: {error_msg}")
                    full_response = f"Sorry, an error occurred: {error_msg}"
                    message_placeholder.markdown(full_response)
                
            except Exception as e:
                st.error(f"Chatbot error: {str(e)}")
                full_response = f"Sorry, an error occurred: {str(e)}"
                message_placeholder.markdown(full_response)
            
            if full_response:
                st.session_state.message_history.append({
                    "role": "assistant",
                    "content": full_response
                })
    
    st.rerun()