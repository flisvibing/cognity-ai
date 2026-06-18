"""Train Cognity 0.1 locally on CPU and save cognity_0.1.pt."""

from __future__ import annotations

import argparse
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data import MODEL_FILE, MODEL_NAME, MODEL_VERSION, build_vocab, encode, get_corpus
from model import CognityLSTM


class CharacterDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, encoded_text: list[int], seq_length: int) -> None:
        self.encoded_text = encoded_text
        self.seq_length = seq_length

    def __len__(self) -> int:
        return max(0, len(self.encoded_text) - self.seq_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.encoded_text[index : index + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    text = get_corpus()
    vocab = build_vocab(text)
    encoded = encode(text, vocab)
    dataset = CharacterDataset(encoded, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = CognityLSTM(
        vocab_size=vocab.size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Training {MODEL_NAME} {MODEL_VERSION} on CPU")
    print(f"Characters: {len(text)} | Vocabulary: {vocab.size} | Sequences: {len(dataset)}")

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, vocab.size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | loss {avg_loss:.4f}")

    checkpoint = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "model_state_dict": model.state_dict(),
        "stoi": vocab.stoi,
        "itos": vocab.itos,
        "config": {
            "vocab_size": vocab.size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.layers,
            "dropout": args.dropout,
        },
    }
    torch.save(checkpoint, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Cognity 0.1 from scratch on CPU.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
