# 📄 Hybrid RAG Chatbot with Memory (Text + PDF)

## 🚀 Overview
This project is a **Hybrid Retrieval-Augmented Generation (RAG) chatbot** that answers questions using:
- Uploaded **PDF documents**
- **Direct user-provided text**
- **Conversational memory**

Unlike basic RAG systems that rely only on vector retrieval, this system **injects user text directly into the prompt context**, making it capable of answering even simple or short statements accurately.

---

## 🧠 Features
- 📂 PDF upload and ingestion  
- ✍️ Custom text context injection  
- 🔍 Semantic retrieval using FAISS  
- 🧠 Conversational memory (chat history)  
- 💬 Streamlit chat interface  
- 🚫 Reduced hallucinations (context-grounded answers only)

---

## 🏗️ Architecture
User PDFs + User Text
↓
Text Splitting
↓
Embeddings (MiniLM)
↓
FAISS Vector Store
↓
Retriever
↓
Prompt (Context + Memory)
↓
LLM (Groq – LLaMA 3)
↓
Streamlit UI


## 🛠️ Tech Stack
- **Python**
- **LangChain**
- **Groq API (LLaMA-3.1-8B)**
- **FAISS**
- **HuggingFace Embeddings**
- **Streamlit**

---