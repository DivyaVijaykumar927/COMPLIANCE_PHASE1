
# PDF Embedding System

## Overview
The **PDF Embedding System** is a Streamlit-based application that allows users to upload PDF documents, extract text, generate embeddings, and perform conversational retrieval using a vector store. The system utilizes Hugging Face Embeddings, FAISS for vector storage, and LangChain for conversational retrieval. The core functionalities are implemented in `helper.py`, which includes methods for text extraction, chunking, vector storage, and conversational AI integration.

## Features
- Upload multiple PDF files
- Extract text from PDFs using `PyPDF2`
- Chunk extracted text for efficient processing
- Generate embeddings using Hugging Face models
- Store embeddings in a FAISS vector database
- Retrieve relevant document sections using a conversational AI model
- Interactive chatbot interface using Streamlit

## Installation & Setup

### STEP 1: Install dependencies
```bash
pip install -r requirements.txt
```

### STEP 2: Set up environment variables
Create a `.env` file in the root directory and add your Hugging Face API key as follows:
```ini
HUGGINGFACE_API_KEY= "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### STEP 3: Run the application
```bash
streamlit run app.py
```

Now, open the browser and navigate to:
```bash
http://localhost:8501
```



## How It Works
1. Upload PDFs: Users upload PDF files via Streamlit.
2. Extract Text: The system extracts raw text using `PyPDF2`.
3. Chunking: The extracted text is split into smaller chunks using `RecursiveCharacterTextSplitter` for efficient embedding.
4. Generate Embeddings: Hugging Face’s embedding model converts text into vector representations using `HuggingFaceEmbeddings`.
5. Vector Storage: The embeddings are stored using FAISS for fast retrieval.
6. Conversational Retrieval: The chatbot retrieves relevant document sections based on user queries using `ConversationalRetrievalChain`.

#
