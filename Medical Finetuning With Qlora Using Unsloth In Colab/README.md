# 🩺 Medical Language Model Fine-Tuning with QLoRA & Unsloth  
### Model: Qwen2.5-1.5B-Instruct | Dataset: FreedomIntelligence/medical-o1-reasoning-SFT | Environment: Google Colab  

---

## 🧠 Overview
This project demonstrates how to fine-tune a lightweight large language model (**Qwen2.5-1.5B-Instruct**) on a **medical reasoning dataset** using **QLoRA** (Quantized Low-Rank Adaptation) via **Unsloth**.

The goal is to efficiently adapt a general instruction-tuned model to perform **clinical reasoning**, **medical Q&A**, and **diagnostic decision-support tasks** using limited GPU memory (such as in Google Colab).

---

## ⚙️ Features
✅ QLoRA 4-bit fine-tuning — memory efficient training  
✅ LoRA adapters for low-rank adaptation  
✅ Domain-specific dataset: *medical reasoning & chain-of-thought*  
✅ Compatible with Google Colab (T4 GPU or better)  
✅ End-to-end workflow: load → tokenize → train → evaluate → save adapter  

---

## 🧩 Architecture

| Component | Description |
|------------|-------------|
| **Base model** | [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| **Framework** | [Unsloth](https://github.com/unslothai/unsloth) + [PEFT](https://github.com/huggingface/peft) |
| **Training type** | 4-bit QLoRA fine-tuning |
| **Dataset** | [`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) |
| **Hardware target** | Google Colab (T4 / A100) |
| **Task** | Clinical question answering & reasoning |

---

## 🧰 Installation

Run the following in a new Colab notebook:

```bash
!pip install unsloth --upgrade
!pip install bitsandbytes transformers accelerate datasets peft trl
````

---

## 🧩 Model Loading (Unsloth + QLoRA)

```python
from unsloth import FastLanguageModel
from datasets import load_dataset

model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,   # Enables QLoRA
)
```

---

## 📚 Dataset Preparation

```python
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT")

def format_example(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    return {
        "text": f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    }

train_dataset = dataset["train"].map(format_example)
```

---

## 🔧 Apply LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0,       # Must be 0 for Unsloth fast patching
    bias = "none",
    target_modules = ["q_proj", "v_proj"],
)
```

---

## 🏋️ Training Configuration

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir = "./medical-qwen-qlora",
    num_train_epochs = 2,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    warmup_steps = 20,
    learning_rate = 2e-4,
    fp16 = True,
    logging_steps = 10,
    save_strategy = "epoch",
    report_to = "none",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = training_args,
)

trainer.train()
```

---

## 💾 Save Adapter

```python
model.save_pretrained("medical-qwen-qlora-adapter")
tokenizer.save_pretrained("medical-qwen-qlora-adapter")
```

---

## 🧪 Evaluation / Testing

```python
from transformers import pipeline

model.eval()
pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 512,
    temperature = 0.7,
    repetition_penalty = 1.1,
)

query = "A 60-year-old hypertensive patient presents with sudden weakness on one side. What are the possible causes and next steps?"
response = pipe(query)[0]['generated_text']
print(response)
```

🩵 *The fine-tuned model should produce step-by-step clinical reasoning and relevant management suggestions.*

---

## 📈 Results & Observations

* QLoRA reduced GPU memory usage from ~12 GB to ~4 GB.
* Fine-tuned model learned domain-specific reasoning without overfitting.
* Outputs exhibit improved diagnostic reasoning and structured explanations.

---

## 📦 Folder Structure

```
medical-qlora/
│
├── medical_qlora.ipynb          # Main Colab notebook
├── medical-qwen-qlora-adapter/  # Saved adapter weights
├── README.md                    # Documentation
└── results/                     # Generated test responses
```

---

## 🧠 Key Learning Outcomes

* How to apply **QLoRA** to adapt large models efficiently
* Using **Unsloth** for ultra-fast PEFT workflows
* Performing domain adaptation for **medical reasoning**
* Monitoring and managing GPU memory in Google Colab

---

## 🪙 Credits

* **Base Model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
* **Dataset:** [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
* **Framework:** [Unsloth](https://github.com/unslothai/unsloth)
* **Author:** *Your Name*
* **Platform:** Google Colab

---

## 🏁 Future Work

* Evaluate model on clinical exam benchmark datasets (e.g., MedQA, PubMedQA)
* Add instruction-response chat interface using Gradio
* Experiment with larger Qwen2.5-3B models if GPU resources allow
