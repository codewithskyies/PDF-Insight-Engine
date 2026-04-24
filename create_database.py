#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = splitter.split_documents(docs)

# embedding_model = OpenAIEmbeddings()
# embedding_model = HuggingFaceEmbeddings(...)
embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")




vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
