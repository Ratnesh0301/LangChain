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

if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({
        'role':'user',
        'content':prompt
    })
    st.chat_message('user').write(prompt)

    llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant",streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append([
            {'role':'assistant',
             'content':response}
        ])
        st.write(response)

