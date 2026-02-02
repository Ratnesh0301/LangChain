import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents import initialize_agent, AgentType
import os
from dotenv import load_dotenv

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name='Search')

st.title("LANGCHAIN - CHAT WITH SEARCH")
st.sidebar.title("SETTINGS")
api_key = st.sidebar.text_input("ENTER YOUR GROQ API KEY", type="password")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {
            'role':'assistant',
            'content':"Hi I'm a CHATBOT who can search the web!"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

