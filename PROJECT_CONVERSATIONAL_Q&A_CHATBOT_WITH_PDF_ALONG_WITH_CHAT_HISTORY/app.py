import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

#os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("CONVERSATIONAL Q&A CHATBOT WITH PDF UPLOADS AND CHAT HISTORY")
st.write("UPLOADS PDFs AND CHAT WITH THE CONTENT")

huggingfacehub_api_token = st.text_input("ENTER YOUR HUGGING FACE TOkEN", type="password")
if huggingfacehub_api_token:
    embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=huggingfacehub_api_token,
    model_name="sentence-transformers/all-MiniLM-l6-v2",
)
api_key = st.text_input("ENTER YOUR GROQ API KEY", type="password")
if api_key:
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")
    #CHAT_INTERFACE
    session_id = st.text_input("SESSION ID",value="default")
    #STATEFULLY MANAGED CHAT HISTORY
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("CHOOSE PDF FILES", type="pdf", accept_multiple_files=False)

    uploaded_file = st.file_uploader(
    "CHOOSE PDF FILE",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    temp_pdf = "temp.pdf"

    with open(temp_pdf, "wb") as file:
        file.write(uploaded_file.getvalue())

    st.success("PDF uploaded successfully!")


    loader = PyMuPDFLoader(temp_pdf)
    documents = loader.load()
    #documents.extend(docs)

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
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    #Answer Questions
    system_prompt = (
        """You're are an assistant for question answering tasks.
        Use the following pieces of retrieved context to asnwer the question.
        if you don't know the answer, say that you don't know the answer.
        Use three sentences maximum and keep the answer concise
        \n\n
        {context}
        """
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [('system',system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human','{input}')
    ])

    question_asnwer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_asnwer_chain)

    def get_session_history(session: str)-> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    converstational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history=get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='asnwer'
    )

    user_input = st.text_input("YOUR QUESTION:")

    if user_input:
        session_history = get_session_history(session_id)
        response = converstational_rag_chain.invoke(
            {'input':user_input},
            config={
                "configurable":{'session_id':session_id}
            }, #Construct a key "abc" in store
        )

        st.write(st.session_state.store)
        st.write('ASSISTANT',response['answer'])
        st.write("CHAT_HISTORY",session_history.messages)
