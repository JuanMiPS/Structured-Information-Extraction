# Structured Information Extraction with Small Language Models

This repository contains a fully executable **Google Colab notebook** for **structured information extraction** using Small Language Models (SLMs).

Given a recipe written in natural language, the system generates a **strict JSON output** with:
- `ingredients` (list)
- `cooking_time` (minutes or `null`)
- `temperature` (`{"value": int, "unit": "C"|"F"}` or `null`)

The main focus is:
1) valid JSON generation  
2) strict schema compliance  
3) reduced hallucinations (no invented fields/values)

---

## Project Overview

The notebook implements the full experimental pipeline:

1. Install dependencies (pinned versions)
2. Load and subset the **RecipeNLG** dataset
3. Build prompts + strict JSON schema instructions
4. Run **PRE** evaluation (zero-shot)
5. Fine-tune models using **QLoRA (4-bit + LoRA adapters)**
6. Run **POST** evaluation
7. Export results (CSV + examples)

---

## Models

Two Small Language Models are used:

- **Qwen2.5-3B-Instruct**
- **Mistral-7B-Instruct-v0.3**

Both are loaded in **4-bit quantization** and fine-tuned using **QLoRA**.

---

## Dataset

We use the public dataset **RecipeNLG**.  
To keep experiments fast and runnable on limited hardware:

- Train: **300 samples**
- Eval: **40 samples**

A fixed random seed is used for reproducibility.

---

## Metrics

The evaluation includes both format correctness and semantic quality:

- `valid_json_rate` — output can be parsed as JSON
- `schema_ok_rate` — output contains exactly the required keys (no extra / no missing)
- `ingredient_f1` — F1 score on ingredient list (set-based)
- `cooking_time_acc` — exact match accuracy on cooking time
- `temperature_acc` — exact match accuracy on temperature (value + unit)

---

## Results (PRE vs POST)

| Model | Phase | Valid JSON | Schema OK | Ingredient F1 | Cooking Time Acc | Temperature Acc |
|------|------|-----------:|----------:|--------------:|-----------------:|----------------:|
| qwen2.5-3b-instruct | pre  | 1.000 | 1.000 | 0.7159 | 0.600 | 0.575 |
| qwen2.5-3b-instruct | post | 1.000 | 1.000 | 0.9583 | 0.650 | 0.925 |
| mistral-7b-instruct-v0.3 | pre  | 1.000 | 1.000 | 0.9079 | 0.500 | 0.675 |
| mistral-7b-instruct-v0.3 | post | 0.975 | 0.975 | 0.9750 | 0.875 | 0.925 |

**Key takeaways:**
- Fine-tuning improves extraction quality for both models.
- Qwen shows a large gain in `ingredient_f1` and `temperature_acc`.
- Mistral improves strongly on `cooking_time_acc`, with a small drop in strict JSON/schema rates.

---

## How to Run (Colab)

### 1) Open the notebook
Run everything from:

- `structured_extraction.ipynb` (Google Colab)

### 2) Add HuggingFace token (required)
Before running the notebook, set your token:

```python
import os
os.environ["HF_TOKEN"] = "YOUR_TOKEN_HERE"
