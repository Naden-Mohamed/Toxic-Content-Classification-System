import pandas as pd
from config import Config
from text_preprocessing import TextPreprocessor
from sklearn.model_selection import train_test_split
from toxicity_classifier import Vocabulary, ToxicDataset, BRNN, Trainer, Evaluator, ToxicityPipeline
from pathlib import Path
 
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
def main():
    config = Config()
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    print(f"[Config] Device: {config.DEVICE}")

    print("\nLoading & preprocessing data …")
    df = pd.read_excel("/content/NLP_Neurova_toxic_content_classification.xlsx")
    print(f"  Dataset size : {len(df):,}")
    print(f"  Label distribution:\n{df['Toxic Category'].value_counts().to_string()}")
    le = LabelEncoder()
    df["Toxic Category"] = le.fit_transform(df["Toxic Category"])
    preprocessor = TextPreprocessor()
    df["clean_text"] = preprocessor.clean_batch(df["query"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].tolist(),
        df["Toxic Category"].tolist(),
        test_size   = config.TEST_SIZE,
        random_state= config.RANDOM_SEED,
        stratify    = df["Toxic Category"],
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("\n Building vocabulary …")
    vocab = Vocabulary(max_size=config.MAX_VOCAB_SIZE, min_freq=config.MIN_WORD_FREQ)
    vocab.build([vocab.tokenization(t) for t in X_train])
    vocab.save(config.VOCAB_PATH)

    print("\n Creating datasets …")
    train_ds = ToxicDataset(X_train, y_train, vocab, config.MAX_SEQ_LEN, preprocessor)
    test_ds  = ToxicDataset(X_test,  y_test,  vocab, config.MAX_SEQ_LEN, preprocessor)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False)

    NUM_CLASSES = df["Toxic Category"].nunique()

    print("\n Building model …")
    model = BRNN(
        vocab_size    = len(vocab),
        embedding_dim = config.EMBEDDING_DIM,
        hidden_dim    = config.HIDDEN_DIM,
        output_dim    = NUM_CLASSES,
        num_layers    = config.NUM_LAYERS,
        dropout       = config.DROPOUT,
        bidirectional = config.BIDIRECTIONAL,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    print(model)

    print("\nTraining …")
    trainer = Trainer(model, config)
    trainer.train(train_loader, test_loader)
    ToxicityPipeline.plot_training_history(trainer.history, config.RESULTS_DIR)

    print("\n Evaluating …")
    evaluator = Evaluator(model, config)
    results   = evaluator.evaluate(test_loader)

    results_path = f"{config.RESULTS_DIR}/metrics.json"
    with open(results_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "confusion_matrix"}, f, indent=2)
    print(f"[Eval] Metrics saved to {results_path}")


    print("\n" + "="*50)
    print("  INFERENCE DEMO")
    print("="*50)
    pipeline = ToxicityPipeline.load(config.MODEL_PATH, config.VOCAB_PATH, config)

    demo_texts = [
        "I absolutely love this community, everyone is so kind!",
        "You are a terrible and disgusting person, I hate you",
        "Great work on the project, keep it up!",
        "Go away you awful idiot, nobody wants you here",
        "The weather is beautiful today, perfect for a walk",
    ]
    for text in demo_texts:
        pred = pipeline.predict(text)
        flag = "TOXIC" if pred["label"] == "TOXIC" else "SAFE"
        print(f"{flag} [{pred['label']:10s}] (conf={pred['confidence']:.2f})  →  {text[:60]}")

    print("\n Pipeline complete. Check the 'results/' folder for plots and metrics.")


if __name__ == "__main__":
    main()