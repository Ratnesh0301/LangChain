from langchain_classic.vectorstores import Cassandra
from langchain_classic.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_classic.llms import openai
from langchain_classic.embeddings import OpenAIEmbeddings
from datasets import load_dataset
import cassio
from pypdf import PdfReader

