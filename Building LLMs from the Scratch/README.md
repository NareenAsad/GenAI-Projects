# Pretraining and Finetuning LLMs from the Ground Up

## Overview

This project is part of my learning and implementation journey from the [**LLM Workshop 2024** by Sebastian Raschka](https://github.com/rasbt/LLM-workshop-2024/tree/main).
It focuses on understanding the **foundations of Large Language Models (LLMs)** — how they process data, how their architecture is structured, and how to **pretrain and finetune them** from scratch using **PyTorch** and **LitGPT**.

The project bridges theory and practice: from coding a GPT-like model manually to using modern libraries like **LitGPT** for efficient pretraining, weight loading, and LoRA-based finetuning.

---

## Setup Instructions

### Local Setup

Clone this repository and install dependencies:

```bash
git clone <your_repo_url>
cd <your_repo_folder>
conda create -n LLMs python=3.10
conda activate LLMs
pip install -r requirements.txt
```

> Make sure your environment supports GPU acceleration (CUDA or similar) for pretraining and finetuning steps.

### Cloud Setup

You can also run the code in a ready-to-go cloud GPU environment such as **Kaggle**, **Google Colab**, or **Lightning Studio**, which comes pre-configured with dependencies.

---

## 📂 Project Structure

| Folder             | Description                                                             |
| ------------------ | ----------------------------------------------------------------------- |
| `01_data`          | Implementing the tokenizer and PyTorch DataLoader for model input       |
| `02_architecture`  | Building the GPT-like architecture (attention blocks, embeddings, etc.) |
| `03_pretraining`   | Pretraining a small-scale GPT model on sample text data                 |
| `04_weightloading` | Loading pretrained weights (Phi-2, Gemma, Mistral, etc.) using LitGPT   |
| `05_finetuning`    | Finetuning pretrained LLMs using LoRA adapters in LitGPT                |

---

## My Contributions

In this project, I started from **Sections 2** from the [LLM Workshop 2024](https://github.com/rasbt/LLM-workshop-2024/tree/main), which included:

* **Implementing the Input Data Pipeline**

  * Created a tokenizer and DataLoader to process raw text for model input.
* **Building the Core Model Architecture**

  * Implemented Transformer-based GPT blocks and attention mechanisms in PyTorch.
* **Pretraining the Model**

  * Trained the GPT-like model on a sample text corpus to enable basic text generation.
* **Finetuning with LitGPT + LoRA**

  * Used the LitGPT framework to load **Microsoft Phi-1_5** and apply **LoRA fine-tuning**.
  * Collected model responses and compared **base vs. finetuned** results.

---

## Key Commands

### 🔹 Pretraining

```bash
python pretrain_gpt.py
```

Or using **LitGPT**:

```bash
litgpt train --config configs/phi1_5.yaml --data data/sample.txt
```

---

### 🔹 Finetuning (LoRA)

```bash
litgpt finetune lora --config configs/phi1_5.yaml --out_dir out/finetune/lora/final
```

After finetuning, you can **merge the LoRA weights** back into the model:

```bash
litgpt merge_lora microsoft/phi-1_5 --adapter_path out/finetune/lora/final --out_dir out/finetune/lora/merged
```

---

### 🔹 Evaluation / Generation

Evaluate model performance on a downstream task:

```bash
litgpt evaluate out/finetune/lora/merged --tasks "mmlu_philosophy" --batch_size 4
```

Or test interactively in Python:

```python
from litgpt import LLM

llm = LLM.load("out/finetune/lora/merged")
response = llm.generate("Explain how neural networks learn.")
print(response)
```

---

## Results

| Model            | Task             | Example Output                                                                |
| ---------------- | ---------------- | ----------------------------------------------------------------------------- |
| Base (Phi-2)     | Text Generation  | “The quick brown fox…”                                                        |
| Finetuned (LoRA) | Instruction Task | “Sure! Here’s a concise explanation of how LLMs learn from large-scale data…” |

---

## References

* [**LLM Workshop 2024**](https://github.com/rasbt/LLM-workshop-2024/tree/main) — by Sebastian Raschka
* **LitGPT** Library — [https://github.com/Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt)
* *Build a Large Language Model From Scratch* — Book by Sebastian Raschka
* **Microsoft Phi-2 Model Card** — [https://huggingface.co/microsoft/phi-2](https://huggingface.co/microsoft/phi-2)

---

## Future Work

* Pretrain using a larger, domain-specific dataset
* Experiment with different LoRA rank settings and datasets
* Add quantization and inference optimizations
* Visualize training and attention patterns using TensorBoard
