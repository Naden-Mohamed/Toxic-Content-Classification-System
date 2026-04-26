import torch

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
    EPOCHS           = 10
    LEARNING_RATE    = 1e-3
    WEIGHT_DECAY     = 1e-5     # L2 regularisation on weights
 
    MODEL_PATH       = "saved_model.pt"
    VOCAB_PATH       = "vocab.pkl"
    RESULTS_DIR      = "results"
 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
 