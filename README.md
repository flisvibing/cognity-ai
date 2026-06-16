# Cognity 0.1

Cognity is a from-scratch local AI chatbot prototype. It uses a small character-level LSTM language model trained with PyTorch on a built-in English conversation dataset, then serves a simple Flask web UI for chatting.

## Features

- **Model name:** Cognity
- **Version:** 0.1
- **Type:** Character-level LSTM language model
- **Runs locally:** CPU only
- **No external AI APIs** and no pretrained models
- **Built-in dataset:** 200 English conversation lines
- **Sampling:** temperature and optional top-k sampling
- **Web UI:** minimal ChatGPT-like single page with timestamps, loading state, and typing effect

## File structure

```text
.
├── app.py                 # Flask web server and /chat endpoint
├── chat.py                # Model loading and text generation
├── data.py                # Built-in dataset and preprocessing
├── model.py               # Cognity LSTM architecture
├── train.py               # CPU training script
├── requirements.txt       # Python dependencies
├── static/
│   └── style.css          # Web UI styling
└── templates/
    └── index.html         # Chat page
```

Training creates this checkpoint:

```text
cognity_0.1.pt
```

## Setup

Use Python 3.10+ if possible.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Cognity

```bash
python train.py
```

The script prints loss after every epoch and saves the trained model as `cognity_0.1.pt`.

For a faster smoke test, run fewer epochs:

```bash
python train.py --epochs 2 --batch-size 32
```

For better responses, train longer:

```bash
python train.py --epochs 100
```

## Chat in the terminal

After training:

```bash
python chat.py
```

Type `quit` or `exit` to stop.

## Run the web app

After training:

```bash
python app.py
```

Open your browser at:

```text
http://127.0.0.1:5000
```

The web backend exposes:

```text
POST /chat
```

with JSON:

```json
{ "message": "hello" }
```

and returns:

```json
{ "response": "hello i am cognity" }
```

## Notes for Cognity 0.1

This is intentionally a small first prototype. Because it is a character-level model trained from a tiny dataset, generated replies can be repetitive or strange. Improve it by adding more English conversation lines in `data.py`, increasing hidden size, and training for more epochs.
