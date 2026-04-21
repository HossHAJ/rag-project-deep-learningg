# RAG Project – Deep Learning

This project implements a **Retrieval-Augmented Generation (RAG)** system using:

- **ChromaDB** for vector storage  
- **Sentence Transformers** for embeddings  
- **Hugging Face FLAN-T5** for answer generation  

---

## Dataset

The dataset used in this project is:

**The Matrix movie script** named as **documatrix.pdf**

---

## Requirements

Install the required Python packages:

```bash
pip install transformers chromadb sentence-transformers langchain_community langchain-text-splitters pypdf tqdm torch

## then run
python rag_matrix.py
