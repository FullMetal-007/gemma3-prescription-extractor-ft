---
language: en
tags:
- gemma3
- unsloth
- lora
- peft
- multimodal
- ocr
- structured-extraction
- medical-prescription
---

# 💊 Gemma 3 LoRA for Structured Prescription Extraction

This repository contains **LoRA adapters** for Google's **Gemma 3 (4B)** vision-language model, fine-tuned for extracting structured data from handwritten medical prescriptions.

Trained using **Unsloth** for high-performance fine-tuning, these adapters specialize the base model to look at an image of a prescription and accurately extract the `MEDICINE_NAME` and `GENERIC_NAME` into a clean JSON format.



***

## 📜 Model Details

* **Base Model:** This repository contains LoRA adapters only. The base model, [`unsloth/gemma-3-4b-it`](https://huggingface.co/unsloth/gemma-3-4b-it), must be loaded separately.
* **Fine-tuning Library:** [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient 4-bit training.
* **Technique:** Parameter-Efficient Fine-Tuning (PEFT) using **LoRA**.
* **Dataset:** Fine-tuned on the "Doctor’s Handwritten Prescription BD dataset," which contains images of handwritten prescriptions and structured labels for `MEDICINE_NAME` and `GENERIC_NAME`.

***

## 🚀 How to Use

To use these adapters, you must first load the base Gemma 3 model and then apply the adapters from this repository on top of it. This model requires a GPU.

```python
!pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)" --quiet
!pip install --no-deps trl peft accelerate bitsandbytes transformers --quiet

from unsloth import FastLanguageModel
from peft import PeftModel
from PIL import Image
import torch
import requests
from io import BytesIO
import json

# 1. Load the base model and processor
base_model, processor = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Apply the LoRA adapters from this Hub repository
model = PeftModel.from_pretrained(base_model, "FullMetal-007/gemma3-prescription-extractor") # 👈 Change to your repo name

# Prepare the prompt and an example image
instruction = "You are an expert at reading medical prescriptions. Extract the medicine name and generic name from the image and provide the output in a structured JSON format."
prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n<image><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# Load an image (example from a URL)
url = "[https://i.imgur.com/81BC6f6.png](https://i.imgur.com/81BC6f6.png)" # Replace with your image URL or local path
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# Run inference
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
predicted_text = processor.decode(output[0], skip_special_tokens=True)
predicted_json = predicted_text.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()

print(predicted_json)
