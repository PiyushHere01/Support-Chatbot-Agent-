# Support-Chatbot-Agent-

Overview

The CDP Support Agent Chatbot is a Flask-based chatbot that utilizes FAISS for vector search and Hugging Face embeddings for natural language processing. The chatbot is designed to answer CDP-related questions efficiently by leveraging pre-embedded documents for similarity search.

Features

Flask API: A simple REST API for chatbot interactions.

FAISS Vector Search: Fast and scalable document retrieval.

Hugging Face Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2 for document similarity.

Hugging Face QA Pipeline: Utilizes distilbert-base-cased-distilled-squad for answering questions.

Installation & Setup

Prerequisites

Make sure you have the following installed:

Python 3.8+

pip (Python package manager)

Step 1: Clone the Repository

git clone https://github.com/your-username/cdp-chatbot.git
cd cdp-chatbot

Step 2: Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows

Step 3: Install Dependencies

pip install -r requirements.txt

Step 4: Set Environment Variables

Create a .env file in the root directory and add your Hugging Face API Key:

HF_API_KEY=your_huggingface_api_key

Step 5: Create the FAISS Vector Index

If you do not have the FAISS index file (index.faiss), generate it using the following script:

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Sample documents (Replace with actual data)
documents = [
    "Your first document content here.",
    "Your second document content here.",
    "Another document's content here."
]

# Initialize the Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [Document(page_content=doc) for doc in documents]

# Create FAISS vector store
vector_db = FAISS.from_documents(docs, embedding_model)

# Save the FAISS index
vector_db.save_local("faiss_index")

This will create the index.faiss file inside the faiss_index directory.

Step 6: Run the Flask Application

python app.py

API Usage

Endpoint: /chat

Method: POST

Request Body:

{
  "query": "What is CDP?"
}

Response:

{
  "response": "CDP stands for Customer Data Platform."
}

Troubleshooting

Error: No such file or directory: index.faiss

Solution: Ensure you have created the FAISS index using Step 5.

Error: openai_api_key missing

Solution: You are using OpenAIEmbeddings instead of Hugging Face Embeddings. Ensure you have replaced:

from langchain_community.embeddings import OpenAIEmbeddings

with:

from langchain_huggingface import HuggingFaceEmbeddings

Error: allow_dangerous_deserialization=True

Solution: Make sure you trust the FAISS index file before enabling this option.
