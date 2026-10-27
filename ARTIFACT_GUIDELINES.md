# Artifact and Results Guidelines

To ensure fair comparison between GRU (A), RNN (B), and LSTM (C), all artifacts and results must follow this contract.

## Directory layout

```
artifacts/
  models/
    gru_best.keras           # default best checkpoint for context-3
    gru_final.keras
    gru_ctx5_best.keras      # optional for other context lengths
    gru_ctx5_final.keras
    rnn_best.keras
    rnn_final.keras
    lstm_best.keras
    lstm_final.keras
    lstm_ctx5_best.keras
    lstm_ctx5_final.keras
    lstm_ctx10_best.keras
    lstm_ctx10_final.keras
results/
  gru_test_metrics.json     # metrics for the run used in comparison
  gru_history.json
  gru_config.json
  rnn_test_metrics.json
  rnn_history.json
  rnn_config.json
  lstm_test_metrics.json
  lstm_history.json
  lstm_config.json
  lstm_ctx5_test_metrics.json
  ...
  lstm_prediction_examples.json
```

## Model checkpoint naming

- **Baseline (context length = 3):** use `<model>_best.keras` and `<model>_final.keras` (e.g., `lstm_best.keras`, `rnn_best.keras`, `gru_best.keras`). This is what the shared predictor and API look for by default.
- **Other context lengths:** use suffix `_ctx{N}` (e.g., `lstm_ctx5_best.keras`, `lstm_ctx10_best.keras`). Do not overwrite the context-3 baseline.
- **Models:** `gru`, `rnn`, `lstm` (lowercase).

## Results naming

- Metrics: `<model>[_{suffix}]_test_metrics.json`
- History: `<model>[_{suffix}]_history.json`
- Config: `<model>[_{suffix}]_config.json`
- Examples: `<model>_prediction_examples.json` (typically for the baseline/checkpoint used in demo)

## Metrics to record (minimum)

All required for fair comparison:
- `samples`, `vocabulary_size`, `parameter_count`, `inference_latency_ms`, `training_seconds`, `epochs_completed`
- `cross_entropy`, `perplexity`
- `top_1_accuracy`, `top_3_accuracy`
- `mse`, `mae`, `rmse`
- `model` (e.g., `lstm`, `rnn`, `gru`), `sequence_length`, `best_checkpoint`, `checkpoint_monitor`, `choices` (hyperparams)

## Data & fairness

- For context length 3, use `data/processed/train.json`, `validation.json`, and `test.json`. For another length, generate matching splits first with `python -m src.data.preprocess --context-size N`; the trainer then reads `data/processed_ctxN/`.
- Use the same `data/processed/vocab.json` (shared vocabulary)
- Keep seed, context length, embedding size, hidden units, dropout, learning rate, batch size aligned when doing a like-for-like comparison. If you deviate, record it in `choices` and in the config.
- Use the shared evaluator (`src/training/evaluate.py`) for test metrics.
- Don't fabricate or hardcode metrics — write them from the actual run.

## Predictor/API/Streamlit expectations

- Shared predictor (`src/inference/predictor.NextWordPredictor`) looks for default checkpoints in `artifacts/models/` and respects env vars: `GRU_MODEL_PATH`, `RNN_MODEL_PATH`, `LSTM_MODEL_PATH`.
- FastAPI serves `POST /predict` and `GET /health`; health reports which model checkpoints are available.
- Streamlit reads all `*_test_metrics.json` in `results/` and displays them; only models with available checkpoints are selectable in prediction.
- Docker Compose mounts `./artifacts:/app/artifacts:ro` and `./results:/app/results:ro`. Train models before expecting prediction endpoints to serve them.

## Commands (reference)

- Train LSTM ctx3: `python -m src.training.train_unified --model lstm`
- Prepare ctx5/10: `python -m src.data.preprocess --context-size 5` / `--context-size 10`
- Train LSTM ctx5/10: `python -m src.training.train_unified --model lstm --sequence-length 5` / `--sequence-length 10`
- Analyze examples: `python -m src.training.analyze_predictions --model artifacts/models/lstm_best.keras`
- API: `uvicorn api.main:app --reload`
- UI: `streamlit run app/streamlit_app.py`
