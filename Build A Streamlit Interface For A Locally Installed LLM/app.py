import streamlit as st
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Load external CSS
def load_css(file_path):
    """Load CSS from external file"""
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_path}")

# Load the stylesheet
load_css("style.css")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Default model
DEFAULT_MODEL = 'llama2'

# Query Ollama function
def query_ollama(prompt, model='llama2'):
    """Send query to Ollama and get response"""
    url = 'http://localhost:11434/api/generate'
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False
    }

    try:
        response = requests.post(url, json=data, timeout=120)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Connection Error: Make sure Ollama is running with 'ollama serve'"
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Try a smaller model or simpler query."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Header section
col1, col2 = st.columns([5, 1])

with col1:
    st.markdown("# 🤖 AI Chat Assistant")
    st.caption(f"🦙 Powered by Ollama - Using model: **{DEFAULT_MODEL}**")

with col2:
    if st.button("Clear Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

# Chat display
if st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg['user'])
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg['bot'])
else:
    st.info("👋 Welcome! I'm your AI assistant. Ask me anything!")

# Input Section
with st.container():
    with st.form(key='message_form', clear_on_submit=True):
        user_input = st.text_area(
            "Message",
            placeholder="Type your message here...",
            height=120,
            label_visibility="collapsed",
            key="input_field"
        )

        submitted = st.form_submit_button("Submit", use_container_width=False, type="primary")

    if submitted and user_input.strip():
        with st.spinner(f'🤔 {DEFAULT_MODEL} is thinking...'):
            bot_response = query_ollama(user_input, DEFAULT_MODEL)
            st.session_state.messages.append({
                'user': user_input,
                'bot': bot_response,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()

    elif submitted and not user_input.strip():
        st.warning("⚠️ Please enter a message first!")

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 40px; color: rgba(255,255,255,0.7);'>
        Powered by Ollama 🦙 | Local AI Chat
    </div>
""", unsafe_allow_html=True)