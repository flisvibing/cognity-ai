"""Flask web server for the Cognity 0.1 local chatbot."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from chat import load_or_stub
from data import MODEL_NAME, MODEL_VERSION

app = Flask(__name__)
bot = load_or_stub()
chat_history: list[dict[str, str]] = []


@app.get("/")
def index() -> str:
    return render_template("index.html", model_name=MODEL_NAME, version=MODEL_VERSION)


@app.post("/chat")
def chat() -> tuple[dict[str, str], int] | dict[str, str]:
    global bot
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    if bot is None:
        bot = load_or_stub()
    if bot is None:
        return jsonify({
            "response": "Cognity has no local model loaded yet. Add tiny_gpt2_chatbot_model/ and tokenizer_bpe/, or run python train.py, then restart python app.py."
        })

    response = bot.reply(message, chat_history)
    chat_history.extend([
        {"role": "user", "content": message},
        {"role": "cognity", "content": response},
    ])
    del chat_history[:-12]
    return jsonify({"response": response, "backend": bot.backend_name})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
