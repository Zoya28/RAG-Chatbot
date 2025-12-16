# RAG-Based Company Knowledge Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers **company-specific** and **general** questions by grounding responses strictly in retrieved data from vector databases.  
This project demonstrates an end-to-end RAG pipeline using **LangChain**, **FAISS**, **HuggingFace embeddings**, and a **Groq-hosted LLaMA model**.

---

## Problem Statement

Traditional chatbots and LLM-based assistants often:
- Hallucinate answers
- Cannot reliably answer questions about **private or company-specific data**
- Mix general knowledge with internal information

This makes them unsuitable for answering questions about a company’s services, projects, or internal documentation.

---

## Solution Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that:

- Ingests structured company data (JSON)
- Converts data into semantic embeddings
- Stores embeddings in **FAISS vector databases**
- Retrieves only the most relevant chunks for a user query
- Generates answers **strictly based on retrieved context**
- Returns **“I don’t know”** if the information is not found in the data

To improve relevance, the chatbot routes queries to **different knowledge bases** using intent classification.

---

## Key Features

-  **Document ingestion from JSON**
-  **Recursive text chunking** with overlap
-  **Semantic search using FAISS**
-  **Intent-based routing**
  - Company-specific questions
  - General conversational questions
-  **Context-grounded LLM responses**
-  **Hallucination control**
-  **Persistent FAISS indexes**
-  **Multi-turn conversational support**

---


## Tech Stack

- **Language**: Python
- **LLM**: LLaMA 3.1 (Groq API)
- **Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Environment Management**: python-dotenv

---

## How It Processes Real Data

1. **Data Loading**
   - Company and general data are loaded from JSON files.
   - Each entry is converted into a LangChain `Document`.

2. **Chunking**
   - Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`.

3. **Embedding & Storage**
   - Each chunk is embedded using HuggingFace sentence-transformer embeddings.
   - Embeddings are stored in persistent FAISS vector indexes.

4. **Query Handling**
   - User queries are classified into:
     - `YOUR-COMPANY` (company-specific)
     - `GENERAL`
     - `OTHER`
   - Based on intent, the appropriate FAISS index is queried.

5. **Answer Generation**
   - Retrieved chunks are injected as context into the LLM prompt.
   - The LLM is instructed to answer **only from the provided context**.
   - If no relevant context is found, the chatbot responds with *“I don’t know.”*

---

##  Getting Started

### 1️⃣ Install Dependencies
```bash
pip install langchain langchain_community faiss-cpu langchain-groq dotenv langchain-huggingface
```

2️⃣ Set Environment Variables
Create a .env file:
```bash
GROQ_API_KEY=your_groq_api_key_here
``` 
3️⃣ Run the Chatbot
```bash
python chatbot.py
```
Type exit to quit.

## Example Queries
“What services does THIS COMPANY offer?”

“Tell me about COMPANY'S projects”

“What is artificial intelligence?”

“How are you?”

**If the answer is not present in the data, the chatbot will respond with:**

I don't know.

**Known Limitations**

- This project is not production-optimized
- LLM-based intent classification introduces latency
- Designed for learning and demonstration purposes
- Single-user, command-line interface

**Future Improvements**

- Replace LLM-based intent classification with embedding-based routing
- Add relevance score thresholding
- Convert to FastAPI for API-based usage
- Add async support and caching
- Improve source citation in responses

**Learning Outcomes**

Through this project, I gained hands-on experience with:
- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- Chunking strategies and embeddings
- Prompt engineering for hallucination control
- Designing multi-knowledge-base AI systems
