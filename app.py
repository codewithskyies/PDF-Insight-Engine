import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Load env variables
load_dotenv()

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📚")

st.title("📚 RAG PDF Chatbot")
st.write("Upload a PDF and ask questions!")

# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save file temporarily
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    st.session_state.vectorstore = vectorstore

    st.success("Embeddings created and stored!")

# ---------------- QUERY SECTION ----------------
if st.session_state.vectorstore:

    query = st.text_input("Ask a question from your PDF:")

    if query:
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),
            ("human",
             """Context:
{context}

Question:
{question}
""")
        ])

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        # LLM
        llm = ChatMistralAI(model="mistral-small-2506")

        response = llm.invoke(final_prompt)

        st.subheader("🤖 Answer")
        st.write(response.content)