import os
import time
import streamlit as st
from utils.config_loader import load_yaml
from src.chat_service.chat_service import LLMChatbot
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import torch
import google.api_core.exceptions
import requests.exceptions


# Initialize Torch path
if hasattr(torch, '__path__') and hasattr(torch.classes, '__file__'):
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# ---- New: Update your LLMChatbot class to include this method ----
# def load_history_into_bot(bot, history):
#     """Loads Streamlit chat history into LLMChatbot's memory"""
#     bot.memory.chat_memory.messages = history

def inititalise_chatbot():

    # Load config file
    chatbot_config = load_yaml("chatbot_config.yaml")
    model_config = load_yaml("model_config.yaml")

    # --- Configuration ---
    try:
        chatbot_page_title = chatbot_config['chatbot_page_title']
        chatbot_title = chatbot_config['chatbot_title']
        chatbot_page_icon = chatbot_config['chatbot_page_icon']
        chatbot_title_icon = chatbot_config['chatbot_title_icon']
    except Exception as e:
        raise Exception(f"Error while loading config file: {e}")

    # --- Streamlit page configuration ---
    try:
        st.set_page_config(page_title=chatbot_page_title , page_icon=chatbot_page_icon, layout="wide")
    except Exception as e:
        st.error(f"Error loading page icon from '{chatbot_page_icon}': {e}")

    # --- Chatbot title ---
    col1, col2 = st.columns([2, 18])
    with col1:
        st.image(chatbot_title_icon, width=100) # Adjust width as needed
    with col2:
        st.title("AI Assistant ChatBot") # Title text without emoji

    # --- Sidebar for chatbot parameters ---
    with st.sidebar:
        st.title("Chatbot Parameters")
        st.markdown(
            """
            <style>
            .custom-text {
                color: #48A6A7;
                font-size: 15px;
                font-weight: 100;
                text-align: left;
                margin-bottom: 20px;
                margin-top: 70px;
            }
            </style>
            <p class="custom-text">Tweak the parameters for performance change</p>
            """,
            unsafe_allow_html=True,
        )

        # Model selection and parameters
        model_provider = st.selectbox("Select LLM Provider", ("google-gemini", "open-ai", "huggingface-endpoint"))
        st.button("Clear Chat", on_click=lambda: st.session_state.pop("chat_history", None))

    # Chat History container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    chat_container = st.container(border=True, height=500)
    st.markdown("</div>", unsafe_allow_html=True)

    # Create a container for the fixed input field at the bottom
    # with st.container():
    user_prompt = st.chat_input("Ask anything...", key="user_input")

    # Initializing the chatbot and memory in session state if not already present
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = LLMChatbot(model_provider)  # Ensure LLMChatbot is properly imported or defined

    # Initialize chat session in Streamlit if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[AIMessage("Hi, I'm your assistant. How may I help you?")]

    # Load chat history into bot memory
    # load_history_into_bot(st.session_state.chatbot, st.session_state.chat_history)


    with chat_container:
        for message in st.session_state.chat_history:
            if type(message).__name__ != 'SystemMessage':
                role = "assistant" if type(message).__name__ == 'AIMessage' else "user"
                with st.chat_message(role):
                    st.markdown(message.content)

    if user_prompt:
        if user_prompt.strip().lower() == 'exit':
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown("Chat ended. It was indeed nice meeting you :)")
                time.sleep(2)
                # Properly clear chat history and reset the session
                st.session_state.pop("chat_history", None)
                st.rerun() # Force Streamlit to refresh the UI and remove old messages
        
        with chat_container:
            st.chat_message("user").markdown(user_prompt)
            st.session_state.chat_history.append(HumanMessage(content=user_prompt))

            try:
                response= st.session_state.chatbot.invoke(user_prompt)
                st.session_state.chat_history.append(AIMessage(content=response))
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        
            except google.api_core.exceptions.ResourceExhausted as e:
                result= "Exhauted daily limit. You can try me after 10 sec"
                with chat_container:
                    with st.chat_message("assistant"):
                            st.markdown(result)

            except requests.exceptions.SSLError as e:
                result= f"Not able to connect with {model_provider} model"
                with chat_container:
                    with st.chat_message("assistant"):
                            st.markdown(result)
        