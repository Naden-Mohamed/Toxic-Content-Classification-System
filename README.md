# Toxic Content Classification System

A comprehensive AI system for **toxic content classification, image captioning, transformer fine-tuning, and content moderation**, combining deep learning and modern LLM-based techniques into a single production-ready pipeline to detect different types of toxicity in user-generated tex.

---

##  Overview
This project consists of four major components:

1. **Toxic Text Classification (LSTM)**
2. **Multimodal Image Captioning (BLIP)**
3. **Parameter-Efficient Fine-Tuning (LoRA + DistilBERT)**
4. **Content Moderation using Llama Guard (Zero-Shot)**

The system is deployed using **Streamlit**, uses a **Mongodb database** for logging inputs and outputs and **W&B** to track models training.

---
## This project was divided into 3 related tasks:

### Task 1: Toxic Content Classification (LSTM)

#### Features:
- Text preprocessing (cleaning, tokenization, padding)
- Word embeddings
- LSTM-based sequence modeling
- Optional Bidirectional LSTM + Dropout

#### Pipeline:
1. Input text
2. Preprocessing
3. Tokenization & encoding
4. Model inference
5. Prediction output

#### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

### Task 2: Image Captioning System

#### Requirements:
- Implemented in `imagecaption.py`
- Uses BLIP model
- Integrated into main application

#### Features:
- Accepts image input
- Generates captions
- Stores results in Mongodb Atlas database

#### Database Stores:
- User text OR image caption
- Classification result

---

### Task 3: Fine-Tuning DistilBERT with LoRA

#### Key Concepts:
- Parameter-Efficient Fine-Tuning (PEFT)
- Low-Rank Adaptation (LoRA)
- Reduced training cost and memory usage

#### Implementation:
- Tokenization using DistilBERT tokenizer
- LoRA applied to attention layers
- Training + validation pipeline

#### Outputs:
- Fine-tuned LoRA weights
- Model evaluation results
- Inference on custom inputs

#### Metrics:
- Accuracy
- Precision
- Recall
- F1-score

---

### Task 4: Content Moderation using Llama Guard

#### Key Idea:
- Zero-shot moderation (no fine-tuning)
- Prompt-engineered classification

#### Features:
- Detects:
  - Toxic content
  - Harmful language
  - Policy violations
- Structured prompt design
- Output parsing

#### Focus:
- Prompt engineering
- Safety-aware AI systems
- Real-world moderation workflows

---
##  Tech Stack

- **Programming Language:** Python
- **Libraries:**
  - Scikit-learn
  - Pandas
  - NumPy
  - PyTorch
  - NLTK
- **NLP Techniques:**
  - Text preprocessing
  - Tokenization
  - Padding
  - Embeddings
---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Naden-Mohamed/Toxic-Content-Classification-System.git
cd Toxic-Content-Classification-System
```
2. Install dependencies:
```bash
 cd src
 pip install -r requirements.txt
 ```
3. Run streamlit
```bash
 streamlit run src/main.py
 ```
