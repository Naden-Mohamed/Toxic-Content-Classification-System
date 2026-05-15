import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd

from collections import Counter
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Any, cast

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from collections import Counter
import nltk
from nltk.corpus import stopwords
from config import Config
from text_preprocessing import TextPreprocessor
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Vocabulary:

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    def __init__(self, max_size: int = 20_000, min_freq: int = 2):
        self.stopwords  = set(stopwords.words('english'))
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self._built = False

    def build(self, tokenised_texts: list[list[str]]):
        counter = Counter()
        for tokens in tokenised_texts:
            counter.update(tokens)

        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}

        for word, freq in counter.most_common(self.max_size):
            if freq < self.min_freq:
                break
            self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word,idx in self.word2idx.items()}

        self._built   = True
        print(f"Built vocabulary: {len(self.word2idx):,} tokens")

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a list of tokens to a list of integer indices."""
        unk = self.word2idx[self.UNK_TOKEN]
        return [self.word2idx.get(t, unk) for t in tokens]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)
        print(f"Saved vocab to {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = data["idx2word"]
        self._built   = True
        print(f"Loaded vocab from {path} — {len(self.word2idx):,} tokens")

    def __len__(self) -> int:
        return len(self.word2idx)

    def tokenization(self, text):
        if not isinstance(text, str):
            return []
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w not in self.stopwords]

    def pad_sequences(self, indices: list[int], max_len: int, pad_idx: int = 0) -> list[int]:

        if len(indices) >= max_len:
            return indices[:max_len]
        return indices + [pad_idx] * (max_len - len(indices))

class ToxicDataset(Dataset):

    def __init__(self, texts:list[str], labels, vocab: Vocabulary, max_len:int , preprocessor: TextPreprocessor):
        self.vocab = vocab
        self.max_len = max_len
        self.preprocessor = preprocessor
        
        self.encoded = []
        for text in texts:
            cleaned_text = preprocessor.clean_text(text)
            tokens = vocab.tokenization(cleaned_text)
            ids = self.vocab.encode(tokens)
            padded_text = self.vocab.pad_sequences(indices= ids, max_len = self.max_len)
            self.encoded.append(padded_text)

        self.labels = list(labels)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx],  dtype=torch.float)
        return x, y



class BRNN(nn.Module):
    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int,
        hidden_dim:    int,
        output_dim:    int,
        num_layers:    int,
        dropout:       float,
        bidirectional: bool,
        pad_idx:       int = 0,
        ):
        super().__init__()
        self.bidirectional = bidirectional


        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embedding_dim,
            padding_idx    = pad_idx,     # padding_idx=pad_idx ensures <PAD> tokens have a zero gradient

        )


        self.lstm = nn.LSTM(
            input_size    = embedding_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        self.dropout = nn.Dropout(dropout)


        directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * directions, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        embedded = self.dropout(self.embedding(x))


        output, (hidden, cell) = self.lstm(embedded)

        if self.bidirectional:
            # Concatenate the last forward (-2) and last backward (-1) layers
            hidden_cat = torch.cat(
                (hidden[-2, :, :], hidden[-1, :, :]), dim=1
            )
        else:
            hidden_cat = hidden[-1, :, :]

        out = self.fc(self.dropout(hidden_cat))
        return out

class Trainer:

    def __init__(self, model, config: Config):
        self.model   = model.to(config.DEVICE)
        self.config  = config
        self.device  = config.DEVICE

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            model.parameters(),
            lr           = config.LEARNING_RATE,
            weight_decay = config.WEIGHT_DECAY,
        )
        # Halve LR if val-loss doesn't improve for 2 epochs
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, factor=0.5
        )

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device).long()

            with torch.set_grad_enabled(train):
                logits = self.model(x)
                loss   = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Clip gradient norm to 1.0 — crucial for LSTM stability
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total      += x.size(0)

        return total_loss / total, correct / total

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        print(f"\n{'='*60}")
        print(f"  Training on {self.device.upper()}")
        print(f"  Epochs={self.config.EPOCHS} | Batch={self.config.BATCH_SIZE} | LR={self.config.LEARNING_RATE}")
        print(f"{'='*60}\n")

        best_val_loss = float("inf")

        for epoch in range(1, self.config.EPOCHS + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            vl_loss, vl_acc = self._run_epoch(val_loader,   train=False)
            self.scheduler.step(vl_loss)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(vl_acc)

            print(
                f"Epoch {epoch:02d}/{self.config.EPOCHS}  |  "
                f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  |  "
                f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}"
            )

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                print(f"Best model saved (val_loss={best_val_loss:.4f})")

        print("\nTraining Completed.")



class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE

    def evaluate(self, loader: DataLoader) -> dict:
        self.model.load_state_dict(
            torch.load(self.config.MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(y.numpy().astype(int).tolist())

        results = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        }

        print("\n" + "="*50)
        print("  EVALUATION RESULTS")
        print("="*50)
        for k, v in results.items():
            print(f"  {k.capitalize():12s}: {v:.4f}")
        
        report = classification_report(
            all_labels, all_preds, 
            target_names=["Safe","Violent Crimes","Elections","Sex-Related Crimes","unsafe","Non-Violent Crimes","Child Sexual Exploitation","Unknown S-Type", "Toxic"]
        )
        print(report)

        # Save Confusion Matrix as a Text File instead of an image
        cm = confusion_matrix(all_labels, all_preds)
        cm_path = f"{self.config.RESULTS_DIR}/confusion_matrix.txt"
        np.savetxt(cm_path, cm, fmt='%d', delimiter='\t')
        print(f"Evaluation done. Confusion matrix saved as text to {cm_path}")

        results["confusion_matrix"] = cm.tolist()
        return results

    # @staticmethod
    # def _plot_cm(cm: np.ndarray) -> None:
    #     Path(Config.RESULTS_DIR).mkdir(exist_ok=True)
    #     fig, ax = plt.subplots(figsize=(6, 5))
    #     sns.heatmap(
    #         cm, annot=True, fmt="d", cmap="Blues", ax=ax,
    #         xticklabels=["Non-Toxic", "Toxic"],
    #         yticklabels=["Non-Toxic", "Toxic"],
    #     )
    #     ax.set_xlabel("Predicted")
    #     ax.set_ylabel("Actual")
    #     ax.set_title("Confusion Matrix")
    #     plt.tight_layout()
    #     path = f"{Config.RESULTS_DIR}/confusion_matrix.png"
    #     plt.savefig(path)
    #     plt.close()
    #     print(f"Evalation done, Confusion matrix saved to {path}")



class ToxicityPipeline:
    """
    End-to-end reusable inference pipeline.

    Usage:
        pipeline = ToxicityPipeline.load("saved_model.pt", "vocab.pkl", config)
        result   = pipeline.predict("You are absolutely terrible!")
        print(result)
        # → {"label": "TOXIC", "confidence": 0.94, "probability": 0.94}
    """

    def __init__(self, model, vocab: Vocabulary, config: Config):
        self.model        = model.to(config.DEVICE)
        self.vocab        = vocab
        self.config       = config
        self.device       = config.DEVICE
        self.preprocessor = TextPreprocessor(dataset_path=self.config.DATASET_PATH)

    def predict(self, text: str) -> dict:
        self.model.eval()
        with torch.no_grad():
            clean  = self.preprocessor.clean_text(text)
            tokens = self.vocab.tokenization(clean)
            ids    = self.vocab.encode(tokens)
            padded = self.vocab.pad_sequences(ids, self.config.MAX_SEQ_LEN)
            x      = torch.tensor([padded], dtype=torch.long).to(self.device)
            prob   = torch.sigmoid(self.model(x).squeeze()).item()

        return {
            "text":        text,
            "label":       "TOXIC" if prob >= 0.5 else "NON-TOXIC",
            "probability": round(prob, 4),
            "confidence":  round(max(prob, 1 - prob), 4),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    @classmethod
    def load(
        cls,
        model_path: str,
        vocab_path: str,
        config: Config,
    ) -> "ToxicityPipeline":
        vocab = Vocabulary()
        vocab.load(vocab_path)

        model = BRNN(
            vocab_size    = len(vocab),
            embedding_dim = config.EMBEDDING_DIM,
            hidden_dim    = config.HIDDEN_DIM,
            output_dim    = 9,
            num_layers    = config.NUM_LAYERS,
            dropout       = config.DROPOUT,
            bidirectional = config.BIDIRECTIONAL,
        )
        model.load_state_dict(
            torch.load(model_path, map_location=config.DEVICE)
        )
        print(f"Model loaded from {model_path}")
        return cls(model, vocab, config)


    @staticmethod
    def plot_training_history(history: dict, results_dir: str = "results") -> None:
        """Saves training history to CSV instead of plotting to avoid matplotlib issues."""
        Path(results_dir).mkdir(exist_ok=True)
        
        df_history = pd.DataFrame(history)
        csv_path = f"{results_dir}/training_history.csv"
        df_history.to_csv(csv_path, index_label="epoch")
        
        print(f"[Log] Training history saved to {csv_path}")
        print("You can open this file in Excel to visualize your Loss and Accuracy.")

class FullPipeline:
    def __init__(self) -> None:
        self.config = Config()
        self.preprocessor = TextPreprocessor(dataset_path=self.config.DATASET_PATH)
        self.df = self.preprocessor .handle_dataset()
        Path(self.config.RESULTS_DIR).mkdir(exist_ok=True)
        print(f"[Config] Device: {self.config.DEVICE}")

    def run_pipeline(self):
        df = self.df
        le = LabelEncoder()
        label_array = cast(np.ndarray, le.fit_transform(df["label"]))
        df["label"] = label_array.tolist()

        clean_texts = cast(np.ndarray, self.preprocessor.clean_batch())
        df["clean_text"] = clean_texts.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            df["clean_text"].tolist(),
            df["label"].tolist(),
            test_size   = self.config.TEST_SIZE,
            random_state= self.config.RANDOM_SEED,
            stratify    = df["label"],
        )
        print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

        print("\n Building vocabulary …")
        vocab = Vocabulary(max_size=self.config.MAX_VOCAB_SIZE, min_freq=self.config.MIN_WORD_FREQ)
        vocab.build([vocab.tokenization(t) for t in X_train])
        vocab.save(self.config.VOCAB_PATH)

        print("\n Creating datasets …")
        train_ds = ToxicDataset(X_train, y_train, vocab, self.config.MAX_SEQ_LEN, self.preprocessor )
        test_ds  = ToxicDataset(X_test,  y_test,  vocab, self.config.MAX_SEQ_LEN, self.preprocessor )

        train_loader = DataLoader(train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True,  drop_last=True)
        test_loader  = DataLoader(test_ds,  batch_size=self.config.BATCH_SIZE, shuffle=False)

        NUM_CLASSES = df["label"].nunique()

        print("\n Building model …")
        model = BRNN(
            vocab_size    = len(vocab),
            embedding_dim = self.config.EMBEDDING_DIM,
            hidden_dim    = self.config.HIDDEN_DIM,
            output_dim    = NUM_CLASSES,
            num_layers    = self.config.NUM_LAYERS,
            dropout       = self.config.DROPOUT,
            bidirectional = self.config.BIDIRECTIONAL,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")
        print(model)

        print("\nTraining …")
        trainer = Trainer(model, self.config)
        trainer.train(train_loader, test_loader)
        ToxicityPipeline.plot_training_history(trainer.history, self.config.RESULTS_DIR)

        print("\n Evaluating …")
        evaluator = Evaluator(model, self.config)
        results   = evaluator.evaluate(test_loader)

        results_path = f"{self.config.RESULTS_DIR}/metrics.json"
        with open(results_path, "w") as f:
            json.dump({k: v for k, v in results.items() if k != "confusion_matrix"}, f, indent=2)
        print(f"[Eval] Metrics saved to {results_path}")