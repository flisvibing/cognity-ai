# External Cognity model files

Cognity now looks for the user's locally trained tiny GPT-2 style model before falling back to the original character-level LSTM checkpoint.

## Preferred local model layout

Place the model folders at the repository root:

```text
.
├── tiny_gpt2_chatbot_model/
│   ├── config.json
│   ├── generation_config.json
│   └── model.safetensors
└── tokenizer_bpe/
    ├── merges.txt
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.json
```

The current source folders provided by the user are:

- `https://github.com/flisvibing/cognity-ai/tree/main/tiny_gpt2_chatbot_model`
- `https://github.com/flisvibing/cognity-ai/tree/main/tokenizer_bpe`

## Runtime behavior

`chat.py` loads models in this order:

1. If all tiny GPT-2 model and tokenizer files are present, load them locally on CPU with Hugging Face Transformers.
2. Otherwise, fall back to the original `cognity_0.1.pt` character-level LSTM checkpoint.
3. If neither model exists, the Flask UI returns a setup message instead of crashing.

## Updating the model

When a new trained model is available, replace the contents of these two folders and restart the Flask app:

```bash
python app.py
```

No external AI service is used at inference time. The model and tokenizer are loaded from local files only.
