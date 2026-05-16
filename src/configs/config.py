import torch
import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    MAX_VOCAB_SIZE   = 20_000   
    MAX_SEQ_LEN      = 200      
    MIN_WORD_FREQ    = 2        
    TEST_SIZE    = 0.20
    VAL_SIZE     = 0.10   
    RANDOM_SEED  = 42
 
    # BiLSTM Architecture
    EMBEDDING_DIM = 128
    HIDDEN_DIM    = 256
    NUM_LAYERS    = 2
    DROPOUT       = 0.40
    BIDIRECTIONAL = True 

    # Training
    BATCH_SIZE    = 64
    EPOCHS        = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-5

    # LoRA / DistilBERT
    DISTILBERT_MODEL      = "distilbert-base-uncased"
    LORA_WEIGHTS_PATH     = "task3/lora_weights"
    LORA_R                = 8
    LORA_ALPHA            = 16
    LORA_DROPOUT          = 0.1
    LORA_TARGET_MODULES   = ["q_lin", "v_lin"]   
    DISTILBERT_MAX_LEN    = 128
    DISTILBERT_BATCH_SIZE = 32
    DISTILBERT_EPOCHS     = 5
    DISTILBERT_LR         = 2e-4

    # Llama Gaurd
    # Fallback: use a smaller instruction-tuned model for zero-shot moderation
    LLAMA_GUARD_MODEL = "meta-llama/Llama-Guard-3-1B"
    LLAMA_HF_TOKEN    = os.getenv("HF_TOKEN", "")          
    LLAMA_RESULTS_PATH = "task3/llama_guard_results.json"


    DATASET_PATH = "src/data/raw/NLP_Neurova_toxic_content_classification.xlsx"
    MODEL_PATH       = "src/saved_models/saved_model.pt"
    VOCAB_PATH       = "src/saved_models/vocab.pkl"
    RESULTS_DIR      = "src/data/logs"
    CSV_PATH = "src/data/logs/toxic_logs.csv"


    PRETRAINED_CAPTIONING_MODEL = "norwoodsystems/image-caption"
 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    MONGODB_URI = os.getenv("MONGODB_CONNECTION_STRING")
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
 
    # ── Label mapping (from LabelEncoder order) ───────────────────────────────
    # Populated at runtime; stored here for inference reuse
    LABEL_NAMES: list[str] = []
