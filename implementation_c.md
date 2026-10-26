# Person C Implementation Plan

## Role and objective

Person C owns the **LSTM experiment** and leads the **inference application and deployment work**. The objective is to add an LSTM that can be compared fairly with Person A's GRU and Person B's vanilla RNN, then expose all available models through one prediction flow and a usable demo.

The work should preserve the project's central contract: given a fixed context of previous words, predict a probability distribution over the shared vocabulary for the next word.

## Current project state

The repository currently contains the Person 2 vanilla RNN and its training/evaluation path:

- `src/models/rnn.py` builds an embedding → `SimpleRNN` → softmax model.
- `src/training/data.py` loads processed examples and maps tokens to IDs, reserving IDs for padding and unknown words.
- `src/training/train.py` trains the RNN and saves a checkpoint, history, configuration, and test metrics.
- `src/training/evaluate.py` calculates the project's classification and probability-distribution metrics.
- `docs/PERSON_2_RNN.md` documents the existing experiment choices.

There is not yet an LSTM implementation, inference module, API, Streamlit app, or Docker setup in the source tree. The project README describes these as planned work. This plan adds Person C's pieces while keeping the RNN track working.

## Scope

### Person C leads

- LSTM model implementation and experiment results.
- LSTM experiments at context lengths 3, 5, and 10.
- Shared model-loading and next-word prediction functionality.
- FastAPI prediction endpoint.
- Streamlit demonstration app.
- Application containerization and deployment preparation.

### Coordinate with the team

- Person A owns GRU and preprocessing; agree on the shared vocabulary, split files, and saved-model conventions.
- Person B owns the vanilla RNN and evaluation/experiment tracking; agree on the model interface, training settings, metrics, and result schema before running final comparisons.
- Docker is shared with Person A in the project work split. Person C should lead packaging the application and coordinate any shared container files with A.

## Target files

Proposed files for Person C's work:

```text
src/
  models/
    lstm.py
  inference/
    __init__.py
    predictor.py
api/
  __init__.py
  main.py
app/
  streamlit_app.py
docs/
  PERSON_3_LSTM.md
Dockerfile
docker-compose.yml          # if API and UI are separate services
```

Use the existing `src/training/data.py` and evaluation utilities where practical. Coordinate with Person B before adding shared training files: the long-term target is one consistent training/evaluation path with a model selection option, not an LSTM-only path with different defaults.

## Phase 1 — Agree on interfaces and experiment rules

Before implementation, confirm these choices with A and B and record them in `docs/PERSON_3_LSTM.md`:

1. **Data:** use the existing `train.json`, `validation.json`, `test.json`, and `vocab.json` unless the team explicitly regenerates all splits together.
2. **Token IDs:** use `Vocabulary` from `src/training/data.py`; preserve its `<pad>` ID 0 and `<unk>` ID 1 conventions.
3. **Model contract:** accept an integer tensor shaped `(batch, sequence_length)` and return softmax probabilities shaped `(batch, vocabulary_size)`.
4. **Shared settings:** align seed, optimizer, learning rate, batch size, epoch limit, early-stopping rule, embedding size, and context length with the other model tracks where possible.
5. **Result format:** agree on artifact names and JSON keys for configuration, training history, and metrics.
6. **Runtime model selection:** determine how the predictor finds the GRU, RNN, and LSTM artifacts, including how it behaves when one model has not been trained yet.

**Completion criteria:** A, B, and C can describe the same input/output contract and final comparison conditions; Person C's checkpoint and result naming are agreed before training.

## Phase 2 — Implement the LSTM architecture

Add `src/models/lstm.py`, following the organization of `src/models/rnn.py`:

1. Define an `LSTMConfig` dataclass with embedding size, hidden units, dropout, and recurrent dropout.
2. Validate positive dimensions and dropout ranges.
3. Build a Keras model with:
   - integer context input,
   - embedding with masking for padding,
   - one `LSTM` layer returning the final sequence representation,
   - dense softmax output over the complete vocabulary.
4. Add a compile helper using the agreed optimizer and sparse categorical cross-entropy. Keep any training metrics consistent with the shared pipeline.
5. Use clear names for layers and the model so saved model inspection is understandable.

The initial LSTM should use a capacity comparable to the RNN/GRU configurations. Do not tune it independently before the shared baseline comparison is complete.

**Completion criteria:** the model accepts the agreed input shape, produces a normalized vocabulary probability distribution per example, supports padding masks, and can be saved and loaded through the project's Keras workflow.

## Phase 3 — Train and evaluate the baseline LSTM

Use the shared training pipeline if it supports architecture selection. If it does not yet, coordinate with B to add a small model factory or agreed `--model lstm` option rather than copying the entire training script. Reuse `load_examples`, the same split data, and the common evaluator.

Baseline settings should match the project experiment configuration, currently centered on a three-word context. Keep the chosen embedding dimension, hidden size, dropout, learning rate, batch size, seed, and early stopping visible in the saved configuration.

Save artifacts using the naming convention agreed with the team. Suggested names if no convention exists:

```text
artifacts/models/lstm_best.keras
artifacts/models/lstm_final.keras
results/lstm_test_metrics.json
results/lstm_history.json
results/lstm_config.json
```

Record at least:

- Cross-entropy, Top-1 accuracy, Top-3 accuracy, and perplexity.
- MSE, MAE, and RMSE as additional required measures.
- Parameter count, training duration, inference latency, and epochs completed.

**Completion criteria:** a fresh run can produce a loadable best checkpoint, config, history, and complete test metrics without changing the data or evaluation definitions used by the other models.

## Phase 4 — Run context-length experiments

Run the LSTM with context lengths **3, 5, and 10**. Keep other settings fixed where practical. Use the same fixed vocabulary and train/validation/test dialogue split for each run so differences can be attributed to context length rather than a changed dataset.

For each run:

1. Save the configuration with its context length.
2. Save the best checkpoint and test metrics with an unambiguous context suffix or run directory.
3. Add its metrics to a summary table for comparison with the GRU and RNN runs.
4. Note any runtime or memory constraints and document any deviations from the agreed settings.

**Completion criteria:** results for all three context lengths are reproducible and can be compared side by side; the selected checkpoint used by the demo is explicitly identified.

## Phase 5 — Analyze LSTM predictions

Use the held-out test examples to collect examples from the LSTM. Include roughly ten correct and ten incorrect predictions, plus a few informative cases where the top-ranked alternatives are linguistically plausible. For each example, record:

- Input context and ground-truth next word.
- Top predictions with probabilities.
- Whether the correct word appears in Top-1 or Top-3.
- A brief interpretation of the failure or ambiguity.

Avoid presenting one reference word as the only possible valid continuation. The dataset supplies one target even when several next words could make sense.

**Completion criteria:** the examples are reproducible from the test split and are suitable for the report or demo.

## Phase 6 — Build shared inference

Add `src/inference/predictor.py` and keep preprocessing consistent with training. The predictor should:

1. Accept a sentence, model name, `top_k`, and the model/context configuration needed for inference.
2. Normalize and tokenize text using the project's agreed tokenizer behavior. Do not remove stopwords.
3. Encode tokens with the shared vocabulary; unseen tokens map to `<unk>`.
4. Select the most recent context words when input exceeds the configured sequence length and left-pad shorter contexts with the padding ID.
5. Load or reuse the selected model checkpoint and run inference with training behavior disabled.
6. Return the top words and probabilities in descending probability order.
7. Validate empty input, unsupported model names, invalid `top_k`, missing artifacts, and model/vocabulary dimension mismatches with useful errors.

Keep model loading out of the per-request hot path where practical: load models at application startup or cache them so a request does not repeatedly deserialize a checkpoint.

Suggested result shape:

```json
{
  "model": "lstm",
  "context": ["i", "would", "like"],
  "predictions": [
    {"word": "to", "probability": 0.42},
    {"word": "a", "probability": 0.17}
  ]
}
```

**Completion criteria:** the same predictor call shape works with each available model; predictions are ordered, probabilities are finite and between 0 and 1, and invalid requests fail clearly.

## Phase 7 — Implement the FastAPI service

Add `api/main.py` with a `POST /predict` endpoint. Use a request schema with `sentence`, `model`, and `top_k`. Keep the endpoint thin: validate the request, call the shared predictor, and serialize its result.

Also provide:

- A health/readiness route that reports whether the service can load its configured resources.
- Clear startup configuration for vocabulary and model artifact paths.
- A documented response and error format.
- Startup loading or caching for model artifacts.

Example request:

```json
{
  "sentence": "I would like to",
  "model": "lstm",
  "top_k": 5
}
```

**Completion criteria:** the API can start from the repository instructions, return predictions for a valid request, and return a client-readable validation or availability error for invalid requests or absent checkpoints.

## Phase 8 — Build the Streamlit application

Add `app/streamlit_app.py`. The first version should prioritize a reliable demo over decorative features:

1. **Next-word prediction:** sentence input, model selector, top-k control, and predictions with probabilities.
2. **Model comparison:** show the final shared test metrics for GRU, RNN, and LSTM. Clearly identify missing results rather than displaying fabricated or stale values.
3. **How it works:** explain tokenization → word IDs → recurrent model → softmax → top-k output.
4. **Useful states:** loading, API unavailable, missing model artifact, and invalid input messages.

Use the API as the UI's prediction backend so the web app and direct API calls share the same inference behavior.

**Completion criteria:** a user can select a model, enter a sentence, view its predictions, and inspect the saved comparison metrics through the UI.

## Phase 9 — Containerize and prepare deployment

Coordinate Docker ownership with Person A. Choose and document one local run path:

- A single image that runs the app/API arrangement, or
- `docker-compose.yml` with separate API and Streamlit services.

Ensure the image or compose setup accounts for Python dependencies, application code, processed vocabulary, and model artifact availability. Avoid assuming large checkpoints are present in a fresh clone unless the repository actually tracks or downloads them. Document the expected artifact placement and any required environment variables.

Provide commands for building and starting the application, and verify the documented flow on a clean environment when feasible. Deployment to a public host should follow once local container startup and model loading work reliably.

**Completion criteria:** the documented container command starts the service and UI, and the app can make a prediction using a clearly specified checkpoint source.

## Phase 10 — Documentation, handoff, and demo readiness

Write `docs/PERSON_3_LSTM.md` with:

- LSTM architecture and experiment choices.
- Training and evaluation commands.
- Context experiment results and checkpoint locations.
- Prediction function/API request and response examples.
- Streamlit and Docker run instructions.
- Known limitations, including vocabulary coverage and one-reference-target evaluation.

Update the root README only for verified capabilities and actual measured results. Do not describe deployment, metrics, or model comparisons as complete until their artifacts exist.

Prepare a short demo sequence: select a model, enter a sentence, inspect top-k predictions, then show the comparison view and how to run the container.

**Completion criteria:** another teammate can follow the docs to load the LSTM, obtain predictions, run the app, and explain the measured comparison.

## Recommended implementation order

| Order | Work item | Main dependency |
|---:|---|---|
| 1 | Agree on model contract, experiment settings, and artifact naming | A and B coordination |
| 2 | Implement `src/models/lstm.py` | Shared model contract |
| 3 | Integrate LSTM into shared training/evaluation | Person B's pipeline decision |
| 4 | Train baseline and run contexts 3, 5, and 10 | LSTM integration |
| 5 | Analyze predictions and save examples | Trained checkpoints |
| 6 | Implement `src/inference/predictor.py` | Vocabulary and checkpoint conventions |
| 7 | Implement FastAPI endpoint | Shared predictor |
| 8 | Implement Streamlit UI | API contract and metric artifacts |
| 9 | Containerize and document local run | Stable API/UI and artifact strategy |
| 10 | Finalize docs and demo | Verified results and working app |

## Overall acceptance checklist

- [ ] LSTM architecture matches the team's input/output contract.
- [ ] LSTM training uses the agreed splits, vocabulary, seed, and comparable settings.
- [ ] Required metrics and model/training timing are saved in a documented format.
- [ ] Context-length results exist for 3, 5, and 10 words.
- [ ] Prediction examples include correct, incorrect, and ambiguous cases.
- [ ] One predictor supports the available GRU, RNN, and LSTM artifacts.
- [ ] API validates requests and returns ordered top-k predictions.
- [ ] Streamlit supports prediction and a results comparison view.
- [ ] Docker run instructions specify where models and vocabulary come from.
- [ ] Documentation reflects implemented behavior and actual measured results.
