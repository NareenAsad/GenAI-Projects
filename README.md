# GenAI-Projects 🧠  
**A collection of Generative AI projects completed during my internship at Arch Technologies.**

---

## 📂 Repository Overview  
This repository contains multiple hands-on projects exploring generative AI, reasoning pipelines, RAG systems, LLM finetuning, and AI application interfaces. Each project is in its own folder with code, notebooks, and documentation.

### Projects Included:
1. **Build A Speech-to-Reasoning Pipeline With Whisper & Quantized LLM**  
   - Convert audio input → transcribe with Whisper → generate reasoning with a quantized LLM (e.g., 4-bit/NF4).  
   - Key Technologies: `Whisper`, `bitsandbytes`, `transformers`, `PyTorch`, Hugging Face models.  

2. **Build A Streamlit Interface For A Locally Installed LLM**  
   - A user-friendly web interface for interacting with a locally hosted LLM.  
   - Key Technologies: `Streamlit`, `PyTorch`, `transformers`.  

3. **Building LLMs from Scratch**  
   - Fine-tuning and training LLMs on custom datasets from scratch.  
   - Key Technologies: `Hugging Face Transformers`, `PyTorch`, `datasets`.  

4. **Building RAG-Unsloth-4bit**  
   - Implementing a Retrieval-Augmented Generation system using a 4-bit quantized LLM.  
   - Key Technologies: `FAISS`, `bitsandbytes`, `transformers`.  

5. **Medical Finetuning With Qlora Using Unsloth LLM**  
   - Fine-tuning a pre-trained LLM for the medical domain using QLoRA.  
   - Key Technologies: `QLoRA`, `Hugging Face Transformers`, `bitsandbytes`.  

---

## 🗂 Repository Structure  
```

GenAI-Projects/
│
├── Build A Speech-to-Reasoning Pipeline With Whisper & Quantized LLM/
│   ├── README.md
│   └── speech-to-reasoning-whisper.ipynb
│
├── Build A Streamlit Interface For A Locally Installed LLM/
│   └── main.py / app.py
│
├── Building LLMs from Scratch/
│   └── notebooks, scripts, datasets/
│
├── Building RAG-Unsloth-4bit/
│   └── notebooks, scripts, embeddings/
│
├── Medical Finetuning With Qlora Using Unsloth LLM/
│   └── notebooks, scripts, datasets/
│
└── README.md (this file)

```

---

## 🚀 How to Use  
1. Clone the repository:  
   ```bash
   git clone https://github.com/NareenAsad/GenAI-Projects.git
   cd GenAI-Projects
   ```

2. Enter the project folder you want to run:

   ```bash
   cd "Build A Speech-to-Reasoning Pipeline With Whisper & Quantized LLM"
   ```
3. Follow the project-specific README for setup instructions, required libraries, and example runs.
4. Ensure you have GPU access for LLM projects and install dependencies like:

   ```bash
   pip install torch transformers bitsandbytes whisper streamlit faiss-cpu
   ```

---

## 🧰 Prerequisites (Windows-friendly)
- **Python**: 3.10 or 3.11 recommended  
- **GPU (optional but recommended)**: NVIDIA GPU with recent CUDA drivers  
- **Virtual environment**: `conda` or `venv`

Quick setup:
```bash
conda create -n genai python=3.11 -y
conda activate genai
# CPU-only torch (works everywhere)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
# Or for CUDA 12.1 (if you have NVIDIA GPU + matching drivers)
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Then install project deps inside each project as instructed in its README.

---

## 🛠 Technology Stack

* Large Language Models (Hugging Face, OpenAI)
* Quantized inference (4-bit, NF4, bitsandbytes)
* Speech-to-text: `Whisper`
* Streamlit apps and interactive interfaces
* Retrieval-Augmented Generation (RAG) with FAISS
* PyTorch, Transformers, Accelerate, Safetensors
* Fine-tuning with QLoRA

---

## 🎯 Purpose

* Document hands-on experience with GenAI pipelines.
* Explore practical deployment of reasoning LLMs.
* Provide reusable templates for future AI projects.
* Showcase a portfolio of internship projects and experiments.

---

## 🤝 Contributing

* Raise issues for bugs, feature requests, or improvements.
* Fork the repository and submit pull requests.
* Ensure you follow the project folder’s README guidelines.

---

## 📄 License

* Unless otherwise stated, code is for **educational and non-commercial use**. Check each project folder for specific licensing details.

---

## 📬 Connect

* [LinkedIn](https://www.linkedin.com/in/nareen-asad) | [Email](mailto:nareenasad07@gmail.com)
