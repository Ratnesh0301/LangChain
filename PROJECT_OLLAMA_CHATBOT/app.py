from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_community.llms import ollama
from ollama import chat
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANCHAIN_TRACING_V2"] = 'true'
os.environ['LANCHAIN_PROJECT'] = "Q&A CHATBOT WITH OLLAMA"

prompt = ChatPromptTemplate(
    [
        ('system','You are a helpful asistant. Please respond to user question'),
        ('user','{question}')
    ]
)

def generate_response(question,engine):
    llm = chat(
        model=engine
    )
    output_parser = StrOutputParser
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

engine = st.sidebar.selectbox("SELECT OLLAMA MODEL",['mistral'])

st.write("<<<GO AHEAD AND ASK ANY QUESTION>>>")
user_input = st.text_input("YOU:")

if user_input:
    response = generate_response(user_input,engine)
    st.write(response)
else:
    st.warning("CHECK THE OLLAMA MODEL")
