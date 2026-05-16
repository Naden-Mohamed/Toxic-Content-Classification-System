import torch
import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    MAX_VOCAB_SIZE   = 20_000   # keep only the N most-frequent tokens
    MAX_SEQ_LEN      = 200      
    MIN_WORD_FREQ    = 2        
    TEST_SIZE        = 0.20     
    RANDOM_SEED      = 42
 
    EMBEDDING_DIM    = 128      
    HIDDEN_DIM       = 256      
    NUM_LAYERS       = 2        
    DROPOUT          = 0.40     
    BIDIRECTIONAL    = True     
 
    BATCH_SIZE       = 64
    EPOCHS           = 15
    LEARNING_RATE    = 1e-3
    WEIGHT_DECAY     = 1e-5     # L2 regularisation on weights
 
    DATASET_PATH = "../dataset/NLP_Neurova_toxic_content_classification.xlsx"
    MODEL_PATH       = "saved_model.pt"
    VOCAB_PATH       = "vocab.pkl"
    RESULTS_DIR      = "results"

    PRETRAINED_CAPTIONING_MODEL = "norwoodsystems/image-caption"
 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    MONGODB_URI = os.getenv("MONGODB_CONNECTION_STRING")
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")

    CSV_PATH = "data/toxic_logs.csv"
 
 