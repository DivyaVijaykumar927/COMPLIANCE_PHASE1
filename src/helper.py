import os
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Directly assigning Hugging Face API key
HUGGINGFACE_API_KEY = "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"  # Replace with your actual API key

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):                             # Process of chunking 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings()                            # Use Hugging Face embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    # Set up Hugging Face LLM with a pipeline for conversational purposes
    hf_pipeline = pipeline("text-generation", model="gpt-2", api_key=HUGGINGFACE_API_KEY)  # You can replace "gpt-2" with any Hugging Face model
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain


