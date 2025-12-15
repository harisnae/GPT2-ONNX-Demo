# GPT‑2 ONNX Demo (Colab)

[![Open In Colab][colab-badge]][colab-notebook]  

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-notebook]: https://colab.research.google.com/github/harisnae/GPT2-ONNX-Demo/blob/main/GPT2_ONNX_Demo.ipynb

---

## Overview

This repository hosts a **single‑cell Google Colab notebook** that shows how to:

* Install the required system and Python packages (Git, `transformers`, `optimum`, ONNX‑Runtime, …) automatically.
* Pull the **ONNX‑exported GPT‑2 model** from the 🤗 Hub (`onnx-community/gpt2-ONNX`).
* Load the model with `optimum.onnxruntime.ORTModelForCausalLM`.
* Run fast text generation on CPU **or** GPU (if a CUDA device is present) using the ONNX Runtime execution providers.
* Use a small helper (`generate_text`) that supports temperature, top‑k, stop‑token, and automatic line‑wrapping.

The notebook works **out‑of‑the‑box** on a free Colab GPU (or CPU when no GPU is available) and requires **no manual setup** beyond clicking the badge.

---

## Getting Started

### 1. Open the notebook in Colab

Click the badge at the top of this page:

[![Open In Colab][colab-badge]][colab-notebook]

The notebook will open in a new tab.  

**⚡ Important:** The demo is written as a **single‑cell script**.  
When the notebook loads, it will:

1. Install system utilities (`git`, `wget`) and all Python dependencies.
2. Detect whether a CUDA‑enabled GPU is available and install the matching ONNX‑Runtime build.
3. Download the ONNX‑exported GPT‑2 files from the Hub.
4. Load the tokenizer and the ONNX model.
5. Run a short generation example.

Just press **Runtime → Run all** (or the ▶️ button at the top of the cell) and watch the output appear in the notebook output area.

---

### 2. Run locally (optional)

If you prefer to execute the demo on your own machine:

```bash
# 1 Clone the repository
git clone https://github.com/harisnae/GPT2-ONNX-Demo.git
cd GPT2-ONNX-Demo

# 2 (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3 Install the exact same dependencies
pip install -r requirements.txt   # you can generate this file from the notebook's pip block

# 4 Launch Jupyter / VS Code
jupyter notebook GPT2_ONNX_Demo.ipynb
```

> **Note:**  
> * If you have a CUDA‑capable GPU, `torch.cuda.is_available()` will be `True` and the notebook will automatically install `onnxruntime‑gpu`.  
> * On CPU‑only machines the notebook falls back to `onnxruntime`.

---

## How the Notebook Works (quick walk‑through)

| Section | What it does |
|---------|--------------|
| **System packages** | Installs `git` and `wget` (quietly) – useful for debugging. |
| **Python dependencies** | Upgrades `transformers`, `huggingface_hub`, `optimum[onnxruntime]`, and `sentencepiece`. |
| **ONNX‑Runtime selection** | Detects CUDA availability and installs `onnxruntime‑gpu` or `onnxruntime` accordingly. |
| **Compatibility shim** | Adds `transformers.utils.cached_property` back for older Optimum versions. |
| **Model download** | Uses `huggingface_hub.snapshot_download` to fetch only the required ONNX files, tokenizer files, and generation config. |
| **Model loading** | Loads the tokenizer (`AutoTokenizer`) and the ONNX model (`ORTModelForCausalLM`). |
| **Generation wrapper** | `generate_text()` – a thin wrapper around `model.generate` that handles prompt tokenisation, config overrides, optional stop‑token truncation, and text wrapping. |
| **Demo** | Generates a 500‑token continuation for the prompt *“Once upon a time in a distant galaxy”* and prints the wrapped result. |

Feel free to edit the `generate_text` call at the bottom of the notebook to experiment with different prompts, `max_new_tokens`, `temperature`, `top_k`, or to switch to a quantised model (e.g. `model_q4.onnx`, `model_fp16.onnx`) by changing the `file_name` argument in the `ORTModelForCausalLM.from_pretrained` call.

---

## Acknowledgements

* **[🤗 Transformers](https://github.com/huggingface/transformers)** – for the tokenizer and generation utilities.  
* **[Optimum](https://github.com/huggingface/optimum)** – for the ONNX‑Runtime integration (`ORTModelForCausalLM`).  
* **[ONNX Community](https://huggingface.co/onnx-community/gpt2-ONNX)** – for providing the exported GPT‑2 ONNX models (`onnx-community/gpt2-ONNX`).  
* **[Google Colab](https://colab.research.google.com/)** – for the free GPU/CPU runtime that makes this demo instantly runnable.
---
