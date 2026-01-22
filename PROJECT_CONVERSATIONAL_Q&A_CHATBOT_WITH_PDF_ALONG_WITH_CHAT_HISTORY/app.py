import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.conversational_retrieval import create
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("CONVERSATIONAL Q&A CHATBOT WITH PDF UPLOADS AND CHAT HISTORY")
st.write("UPLOADS PDFs AND CHAT WITH THE CONTENT")

api_key = st.text_input("ENTER YOUR GROQ API KEY", type="password")
if api_key:
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")
    #CHAT_INTERFACE
    session_id = st.text_input("SESSION ID",value="default")
    #STATEFULLY MANAGED CHAT HISTORY
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("CHOOSE PDF FILES", type="pdf", accept_multiple_files=False)

    if uploaded_files:
        documents = []
        for upload_file in uploaded_files:
            temp_pdf = f'./temp.pdf'
            with open(temp_pdf,'wb') as file:
                file.write(upload_file.getvalue())
                file.name - upload_file.name

        loader = PyMuPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

        #SPLIT AND CREATE EMBEDDINGS FOR THE DOCUMENTS
        text_eplitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
        splits = text_eplitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            """GIVEN A CHAT HISTORY AND THE LATEST USER QUESTION
            WHICH MIGHT REFERENCE CONTEXT IN THE CHAT HISTORY
            FORMULATE A STANDALONE QUESTION WHICH CAN BE UNDERSTOOD
            WITHOUT CHAT HISTORY. DO NOT ANSWER THE QUESTION
            JUST REFORmULATE IT IF NEEDED AND OTHERWISE RETURN AS IT IS
            """
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagePlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

#        history_aware_retriever = 

