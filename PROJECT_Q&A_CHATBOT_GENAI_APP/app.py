import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"

#Prompt Template
prompt = ChatPromptTemplate(
    [
        ("system","You're helpful assistant. Please answer user queries"),
        ("user","Question: {question}")
    ]
)

def generate_response(question, api_key,llm,temperature,max_tokens):
    openai_api_key = api_key
    llm = ChatOpenAI(openai_api_key=openai_api_key, model=llm)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

#Title of the APP
st.title("ENHANCED Q&A CHATBOT WITH OPENAI")

#Sidebar for the Settings
st.sidebar.title("SETTINGS")
api_key = st.sidebar.text_input("ENTER YOUR OPENAI API KEY", type="password")
llm = st.sidebar.selectbox("SELECT AN OPENAI MODEL",['gpt-4o-mini'])
temperature = st.sidebar.slider('TEMPERATURE',min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("MAX TOKENS",min_value=50,max_value=300,value=150)

#MAIN INTERFACE FOR USER
st.write("GO AHEAD AND ASK ANY QUESTION")
user_input = st.text_input("YOU:")

if user_input:
    response = generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("PLEASE PROVIDE YOUR QUESTION!")





