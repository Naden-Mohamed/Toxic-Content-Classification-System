# Toxic Content Classification System

A machine learning-based system for detecting and classifying toxic text.  
This project aims to improve digital conversations by automatically identifying harmful or offensive language.

---

##  Overview

Online platforms often struggle with toxic comments such as insults, threats, and hate speech.  
This project builds a **multi-label classification model** that detects different types of toxicity in user-generated text.

The system classifies content into the following categories:

- Toxic
- Safe
- Violent Crimes
- Elections
- Sex-Related Crimes
- unsafe
- Non-Violent Crimes
- Child Sexual Exploitation
- Unknown S-Type

---

## 🚀 Features

-  Detects multiple types of toxic content
- Built using Natural Language Processing (NLP)
-  Multi-label classification model using BiLSTM
-  Scalable for real-world moderation systems
-  Clean and modular project structure

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

## 📂 Project Structure
Toxic-Content-Classification-System/
│
├── dataset/ 
├── results/
├── models/ # Saved models
├── src/  
│ ├── text_preprocessing.py
│ ├── toxicity_classifier.py
│ ├── config.py
│ ├── requirements.txt
│ └── main.py
│
└── README.md


## Installation

1. Clone the repository:

```bash
git clone https://github.com/Naden-Mohamed/Toxic-Content-Classification-System.git
cd Toxic-Content-Classification-System
```
2. Install dependencies:
``` pip install -r requirements.txt ```
3. Train & Evaluate the model
``` python src/main.py ```

## Model Performance

# Evaluation metrics:
- Accuracy
- Precision / Recall
- F1 Score
