"""Inference helpers for chatting with Cognity 0.1.

Cognity can run with either:
1. the original local character-level LSTM checkpoint (`cognity_0.1.pt`), or
2. the user's locally trained tiny GPT-2 style checkpoint stored in
   `tiny_gpt2_chatbot_model/` with its BPE tokenizer in `tokenizer_bpe/`.

Both backends are local-only and use CPU by default.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from data import MODEL_FILE, MODEL_NAME, MODEL_VERSION, Vocabulary, decode, encode
from model import CognityLSTM

TINY_GPT2_MODEL_DIR = Path("tiny_gpt2_chatbot_model")
TOKENIZER_BPE_DIR = Path("tokenizer_bpe")


class ChatBackend(Protocol):
    backend_name: str

    def reply(self, message: str, history: list[dict[str, str]] | None = None) -> str:
        """Return a local model response."""


class CognityTransformerChat:
    """Local inference for the user's tiny GPT-2 checkpoint."""

    backend_name = "tiny-gpt2-bpe"

    def __init__(
        self,
        model_dir: str | Path = TINY_GPT2_MODEL_DIR,
        tokenizer_dir: str | Path = TOKENIZER_BPE_DIR,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self._validate_files()
        self._validate_dependencies()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model.to("cpu")
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def reply(self, message: str, history: list[dict[str, str]] | None = None) -> str:
        prompt = self._build_prompt(message, history)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        max_new_tokens = 80
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=40,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._postprocess(text)

    def _validate_files(self) -> None:
        required = [
            self.model_dir / "config.json",
            self.model_dir / "model.safetensors",
            self.tokenizer_dir / "tokenizer.json",
            self.tokenizer_dir / "tokenizer_config.json",
            self.tokenizer_dir / "vocab.json",
            self.tokenizer_dir / "merges.txt",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing tiny GPT-2 model files: " + ", ".join(missing))

    def _validate_dependencies(self) -> None:
        missing = [
            package
            for package in ("transformers", "safetensors", "tokenizers")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            raise RuntimeError(
                "Install tiny GPT-2 runtime dependencies first: pip install -r requirements.txt. "
                f"Missing: {', '.join(missing)}"
            )

    def _build_prompt(self, message: str, history: list[dict[str, str]] | None) -> str:
        turns = history[-6:] if history else []
        lines = ["You are Cognity 0.1, a local English-only chatbot."]
        for item in turns:
            role = "Cognity" if item.get("role") == "cognity" else "User"
            lines.append(f"{role}: {item.get('content', '')}")
        lines.append(f"User: {message.strip()}")
        lines.append("Cognity:")
        return "\n".join(lines)

    def _postprocess(self, text: str) -> str:
        stop_markers = ["\nUser:", "\nCognity:", "User:", "Cognity:"]
        for marker in stop_markers:
            if marker in text:
                text = text.split(marker, 1)[0]
        text = " ".join(text.strip().split())
        return text[:400] or "hello i am cognity"


class CognityChat:
    """Character-level LSTM inference for the original Cognity checkpoint."""

    backend_name = "char-lstm"

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


def _tiny_gpt2_files_exist() -> bool:
    required = [
        TINY_GPT2_MODEL_DIR / "config.json",
        TINY_GPT2_MODEL_DIR / "model.safetensors",
        TOKENIZER_BPE_DIR / "tokenizer.json",
        TOKENIZER_BPE_DIR / "tokenizer_config.json",
        TOKENIZER_BPE_DIR / "vocab.json",
        TOKENIZER_BPE_DIR / "merges.txt",
    ]
    return all(path.exists() for path in required)


def load_or_stub() -> ChatBackend | None:
    """Prefer the uploaded tiny GPT-2 model, then fall back to the LSTM checkpoint."""
    if _tiny_gpt2_files_exist():
        try:
            return CognityTransformerChat()
        except RuntimeError as exc:
            print(f"Tiny GPT-2 model found but could not be loaded: {exc}")
    try:
        return CognityChat()
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    bot = load_or_stub()
    if bot is None:
        raise SystemExit(
            "No local model found. Add tiny_gpt2_chatbot_model/ + tokenizer_bpe/ "
            "or train the LSTM with: python train.py"
        )
    print(f"{MODEL_NAME} {MODEL_VERSION} ready with {bot.backend_name}. Type 'quit' to exit.")
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
