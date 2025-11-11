# Speech-to-Reasoning Pipeline (Kaggle Notebook)

This project demonstrates an end-to-end **speech-to-reasoning** pipeline using Python, OpenAI's Whisper, and a quantized reasoning LLM (4-bit) from Hugging Face. The pipeline takes an audio file as input, transcribes it into text, and performs step-by-step logical reasoning to generate structured answers.

---

## Features

1. **Audio Input**
   - Supports local audio upload or download from a public URL.
   - Common formats: `.wav`, `.mp3`, `.m4a`.

2. **Speech-to-Text**
   - Uses [OpenAI Whisper](https://github.com/openai/whisper) for transcription.
   - Supports multiple model sizes: `tiny`, `small`, `medium`, `large`.
   - Automatic language detection and transcription.

3. **Reasoning with Quantized LLM**
   - Loads a **4-bit quantized LLM** from Hugging Face using `bitsandbytes`.
   - Memory-efficient inference with NF4 quantization.
   - Generates step-by-step reasoning and final answers from the transcription.
   - Compatible with models requiring `trust_remote_code=True`.

4. **Kaggle-ready**
   - Fully compatible with Kaggle GPU runtime (T4/A10/A100).
   - Includes sample audio file and ready-to-use model ID placeholder.

---

## Installation

All dependencies are handled in the notebook:

```bash
!pip install --upgrade pip
!pip install openai-whisper transformers accelerate bitsandbytes sentencepiece safetensors
!apt-get -qq install -y ffmpeg
````

---

## Usage

1. **Upload an audio file** or set a sample URL.
2. **Run Whisper transcription**:

```python
import whisper
whisper_model = whisper.load_model("small")
result = whisper_model.transcribe(audio_path)
transcription_text = result['text']
```

3. **Prepare reasoning prompt**:

```python
prompt_template = """
You are a careful, step-by-step reasoning assistant.
Given the following transcription, restate the problem, reason step-by-step, and provide a final answer.

Transcribed audio:
\"\"\"{transcription}\"\"\"
"""
full_prompt = prompt_template.format(transcription=transcription_text)
```

4. **Load quantized LLM**:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "<YOUR_4BIT_MODEL_REPO_ID>"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

5. **Generate reasoning output**:

```python
answers = generate_reasoning(full_prompt, max_new_tokens=512)
print(answers[0])
```

---

## Notes

* Adjust Whisper model size based on GPU memory.
* Use smaller LLMs (7B vs 13B) if VRAM is limited.
* Clear GPU cache between runs: `torch.cuda.empty_cache()`.
* Ensure your Hugging Face model license allows inference and usage.

---

## References

* [OpenAI Whisper](https://github.com/openai/whisper)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* [BitsAndBytes 4-bit Quantization](https://github.com/TimDettmers/bitsandbytes)
