"""Inference helpers for chatting with Cognity 0.1."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from data import MODEL_FILE, MODEL_NAME, MODEL_VERSION, Vocabulary, decode, encode
from model import CognityLSTM


class CognityChat:
    def __init__(self, model_path: str | Path = MODEL_FILE) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing {self.model_path}. Train first with: python train.py"
            )

        checkpoint = torch.load(self.model_path, map_location="cpu")
        config = checkpoint["config"]
        self.vocab = Vocabulary(stoi=checkpoint["stoi"], itos=checkpoint["itos"])
        self.model = CognityLSTM(**config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_chars: int = 160,
        temperature: float = 0.8,
        top_k: int | None = 8,
    ) -> str:
        clean_prompt = self._clean_prompt(prompt)
        seed = f"user: {clean_prompt}\ncognity: "
        indices = encode(seed, self.vocab)
        if not indices:
            indices = encode("hello\n", self.vocab)

        x = torch.tensor([indices], dtype=torch.long)
        hidden = None
        generated: list[int] = []

        logits, hidden = self.model(x, hidden)
        next_logits = logits[:, -1, :]

        for _ in range(max_new_chars):
            next_index = self._sample(next_logits, temperature, top_k)
            generated.append(next_index)
            if self.vocab.itos[next_index] == "\n" and len(generated) > 8:
                break
            x = torch.tensor([[next_index]], dtype=torch.long)
            logits, hidden = self.model(x, hidden)
            next_logits = logits[:, -1, :]

        response = decode(generated, self.vocab).strip()
        response = self._postprocess(response)
        return response or "hello i am cognity"

    def reply(self, message: str, history: list[dict[str, str]] | None = None) -> str:
        context = ""
        if history:
            last_messages = history[-6:]
            context = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}" for item in last_messages
            )
        prompt = f"{context}\nuser: {message}" if context else message
        return self.generate(prompt)

    def _sample(self, logits: torch.Tensor, temperature: float, top_k: int | None) -> int:
        temperature = max(0.1, float(temperature))
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
            probs = F.softmax(values, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return int(indices.gather(-1, sampled).item())
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _clean_prompt(self, prompt: str) -> str:
        allowed = set(self.vocab.stoi)
        prompt = prompt.lower().strip()
        return "".join(ch for ch in prompt if ch in allowed)[:500]

    def _postprocess(self, response: str) -> str:
        response = response.replace("user:", "").replace("cognity:", "")
        response = " ".join(response.split())
        return response[:240]


def load_or_stub() -> CognityChat | None:
    try:
        return CognityChat()
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    bot = CognityChat()
    print(f"{MODEL_NAME} {MODEL_VERSION} ready. Type 'quit' to exit.")
    history: list[dict[str, str]] = []
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        answer = bot.reply(user_text, history)
        history.extend([
            {"role": "user", "content": user_text},
            {"role": "cognity", "content": answer},
        ])
        print(f"Cognity: {answer}")
