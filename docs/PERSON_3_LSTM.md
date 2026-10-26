# Person C — LSTM, Inference, and Demo

Person C owns the LSTM experiment and leads inference, the API, the Streamlit demo, and application packaging. The LSTM follows the existing RNN contract: a fixed-length integer context in, one probability per vocabulary ID out.

## Train the LSTM

Run from the repository root:

```bash
python -m src.training.train_lstm
```

The defaults match the current RNN baseline: context length 3, embedding size 128, 256 recurrent units, dropout 0.2, Adam, learning rate 0.001, batch size 256, seed 42, and early stopping on validation loss. The trainer uses the shared vocabulary, split data, and evaluator. It writes `artifacts/models/lstm_best.keras`, `artifacts/models/lstm_final.keras`, and LSTM history/configuration/metrics under `results/`.

For a short pipeline run, limit all three splits:

```bash
python -m src.training.train_lstm --epochs 1 --max-train-samples 2000 --max-validation-samples 500 --max-test-samples 500
```

Run the context experiments separately. Non-default lengths use distinct checkpoint and result names so they do not overwrite the length-3 baseline:

```bash
python -m src.training.train_lstm --sequence-length 3
python -m src.training.train_lstm --sequence-length 5
python -m src.training.train_lstm --sequence-length 10
```

Metrics include cross-entropy, Top-1 and Top-3 accuracy, perplexity, MSE, MAE, RMSE, parameter count, training time, and inference latency. The dataset files and vocabulary must be shared with the other model runs for a fair comparison.

## Export examples for error analysis

After training, export deterministic Top-1 correct, Top-1 incorrect, and Top-3 hit examples from the held-out test set:

```bash
python -m src.training.analyze_predictions --model artifacts/models/lstm_best.keras
```

The output defaults to `results/lstm_prediction_examples.json`. Use these examples to discuss model behavior, including plausible alternatives that do not match the single reference target.

## Run inference

The shared predictor is `src.inference.predictor.NextWordPredictor`. It lowercases and whitespace-tokenizes the sentence, maps unknown words to `<unk>`, keeps the most recent words required by the selected model, and left-pads shorter contexts. Model files are loaded on first use and then cached.

Default checkpoint paths:

```text
artifacts/models/gru_best.keras
artifacts/models/rnn_best.keras
artifacts/models/lstm_best.keras
```

Override a checkpoint path with `GRU_MODEL_PATH`, `RNN_MODEL_PATH`, or `LSTM_MODEL_PATH`. The vocabulary defaults to `data/processed/vocab.json`.

## Run the API and UI locally

Install `requirements.txt`, then start the API:

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API reference. Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sentence":"I would like to","model":"lstm","top_k":5}'
```

Start the Streamlit UI in another terminal:

```bash
streamlit run app/streamlit_app.py
```

Set `CONTEXTNEXT_API_URL` if the API is running at a different address. The UI shows only models whose checkpoint files are available. The comparison view reads metric files from `results/` and reports only metrics that have actually been saved.

## Run with Docker Compose

Build and start the API and UI from the repository root:

```bash
docker compose up --build
```

The UI is served at `http://localhost:8501`; the API is at `http://localhost:8000`. Model checkpoints are mounted read-only from `./artifacts`, so train or place the model files there before expecting predictions. Result JSON files are mounted from `./results` for the comparison screen. The vocabulary is copied into the image at build time; rebuild after changing it.

## Limitations and integration notes

- The current project contains the vanilla RNN track; GRU and LSTM checkpoints are optional and become selectable when present.
- The inference tokenizer intentionally matches current preprocessing's lowercase and whitespace split. It does not strip punctuation.
- Test metrics use one reference next word per example. A linguistically plausible alternative may still count as incorrect.
- Compare models only when they use the same processed splits, vocabulary, and experiment conditions. Record any deviation with the results.
