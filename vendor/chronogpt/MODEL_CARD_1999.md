---
library_name: pytorch
license: mit
language:
- en
tags:
- chronologically consistent
- instruction following
- modded-nanogpt
- large language model
- lookahead-bias-free
pipeline_tag: text-generation
inference: false
---

# ChronoGPT-Instruct

ChronoGPT-Instruct is a family of **chronologically consistent, instruction-following large language models (LLMs)** that eliminate lookahead bias by training exclusively on time-stamped data available **before a fixed knowledge-cutoff date τ**.  
Each `ChronoGPT-Instruct-τ` extends the `ChronoGPT-τ` base models through supervised instruction fine-tuning while strictly maintaining temporal separation from all post-τ information.

These models provide the research community with a transparent, replicable benchmark for testing **lookahead-bias-free prediction** in economics, finance, and other time-sensitive domains.

---

## 🔍 Model Overview

| Property | Description |
|:--|:--|
| **Architecture** | Transformer-decoder |
| **Parameters** | ≈ 1.55 B |
| **Layers** | 52 layers |
| **Embedding dim** | 1,536 |
| **Context length** | 1,792 tokens |
| **Tokenizer** | `GPT2Tokenizer` (Hugging Face) |
| **Training stage** | Pretraining + Instruction Fine-tuning (SFT) |
| **License** | MIT |
| **Languages** | English |

---

## 🧠 Training & Data

### Chronological Consistency
Each model’s corpus satisfies chronologically consistency in both pretraining and instruction-finetuning phases. Texts dated after the model year are excluded, ensuring zero overlap with evaluation data. A GPT-4.1 classifier screens every instruction-response pair.

### Instruction-Finetuning Corpus
| Stage | Source | # Examples | Avg Length |
|:--|:--|:--:|:--:|
| 1 | LLMs-from-Scratch | 1 097 | 102 |
| 2 | GPT-3 Self-Instruct | 67 136 | 183 |
| 3 | AllenAI Tulu-3 Mixture | 356 886 | 2 513 |

Only English, non-code entries with pre-2000 content (classifier label = 0 & confidence = 10) are retained.

We release the SFT dataset at https://huggingface.co/datasets/manelalab/ChronoInstruct-SFT.

---

## 🚀 Usage Examples

You can try ChronoGPT-instruct directly in your browser via Google Colab:

<p align="left">
  <a href="https://colab.research.google.com/github/LinyingLyu/ChronoGPT/blob/main/ChronoGPT_instruct_tutorial.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
  </a>
</p>

---

## 👩‍💻 Citation

```
@article{He_Lv_Manela_Wu_chronogpt_2025,
  title={Instruction Tuning Chronologically Consistent Language Models},
  author={He, Songrun and Lv, Linying and Manela, Asaf and Wu, Jimmy},
  journal={Working Paper},
  year={2025}
}
```
