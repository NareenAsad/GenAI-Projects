# 🚀 Retrieval-Augmented Generation (RAG) with Unsloth Dynamic 4-Bit Quantization

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline leveraging **Unsloth's dynamic 4-bit quantized large language models (LLMs)**.  
The system retrieves relevant information from a domain-specific corpus and generates context-grounded answers — all while using minimal GPU memory.

---

## 📘 Overview

### 🔹 Goal
To build an efficient **RAG system** that can:
- Index and retrieve domain-specific documents.
- Use a **quantized Unsloth model** for low-VRAM inference.
- Generate grounded responses based on retrieved context.
- Demonstrate memory-efficient performance with **dynamic 4-bit quantization**.

### 🔹 Key Features
✅ End-to-end RAG pipeline (retrieval + generation)  
✅ FAISS-based document indexing  
✅ Unsloth dynamic quantization for efficient GPU use  
✅ WikiQA dataset integration for demonstration  
✅ Interactive querying and document expansion  

---

## 🧠 Architecture

```

┌──────────────────────┐
│   Domain Documents   │  ← WikiQA dataset
└─────────┬────────────┘
│
▼
┌──────────────────────┐
│   FAISS Indexing     │  ← Convert docs to embeddings
└─────────┬────────────┘
│
▼
┌──────────────────────┐
│    Retriever (Top-K) │  ← Retrieve most relevant chunks
└─────────┬────────────┘
│
▼
┌──────────────────────────────┐
│  Unsloth Quantized LLM (4-bit) │  ← Generate final answer
└──────────────────────────────┘

````

---

## 🧩 Dataset

The project uses the **WikiQA** dataset from Hugging Face for demonstration:

```python
from datasets import load_dataset
dataset = load_dataset("wiki_qa", split="train[:5%]")
````

Each document includes a Wikipedia title, a question, and a short answer.

---

## ⚙️ Setup & Installation

### 1️⃣ Clone or open in Colab

👉 **[Open in Google Colab](https://colab.research.google.com/drive/1s1wWrYjDLTb46hbLAoR1YoSlHtpoZd7b?usp=sharing)**

### 2️⃣ Install dependencies

```bash
!pip install unsloth faiss-cpu datasets transformers accelerate bitsandbytes
```

### 3️⃣ Run the notebook

Follow the cells in sequence:

* Load dataset
* Initialize and quantize model
* Build FAISS retriever
* Run RAG pipeline

---

## 🧪 Example Queries

After running the notebook, try:

```python
test_custom_query(rag_system, "Who discovered penicillin?")
test_custom_query(rag_system, "When was the Eiffel Tower built?")
test_custom_query(rag_system, "How deep can we drill underwater?")
```

You can also add custom documents:

```python
rag_system.add_documents(["Machine learning is a subfield of AI focused on pattern recognition."])
```

---

## 📊 Results Summary

| Stage        | Description                                       | Output Example                  |
| ------------ | ------------------------------------------------- | ------------------------------- |
| Retrieval    | Finds relevant chunks from indexed corpus         | “Title: Deepwater drilling…”    |
| Generation   | Produces grounded answers using retrieved context | “Deepwater drilling refers to…” |
| Quantization | Uses Unsloth’s dynamic 4-bit model to reduce VRAM | ~3.5GB usage on T4 GPU          |

---

## 🧮 Memory Efficiency

Unsloth’s **dynamic 4-bit quantization** preserves precision for key layers while compressing others — reducing VRAM use by up to **75%** compared to full precision models.

| Model Type    | VRAM Usage | Speed       |
| ------------- | ---------- | ----------- |
| FP16 Model    | ~12GB      | Baseline    |
| Dynamic 4-bit | ~3–4GB     | 1.3× faster |

---

## 🧰 Tech Stack

* 🧠 **Unsloth** – Dynamic quantization and LLM optimization
* 🔍 **FAISS** – Vector similarity search
* 📚 **Transformers** – Model interface
* 📊 **Datasets** – Data loading (WikiQA)
* ⚙️ **PyTorch** – Backend framework

---

## 🧑‍💻 Author

**Nareen Asad**
💼 Student Project — RAG with Quantized LLMs
📅 November 2025

---

## 📎 Reference

* [Unsloth Documentation](https://github.com/unslothai/unsloth)
* [Hugging Face WikiQA Dataset](https://huggingface.co/datasets/wiki_qa)
* [FAISS Library](https://github.com/facebookresearch/faiss)

---

## 💡 Future Improvements

* Add AI/ML domain-specific datasets (e.g., “AI abstracts”)
* Experiment with larger quantized models (7B, 13B)
* Save and reload FAISS index between sessions
* Add evaluation metrics (BLEU, ROUGE, context accuracy)

---