# ContextNext

ContextNext is an educational next-word prediction project comparing a vanilla RNN, GRU, and LSTM on DailyDialog. Each model receives a fixed context of recent words and returns a probability distribution over the shared vocabulary. The Streamlit app exposes the tokens, IDs, model scores, dataset reference, and differences between the three architectures.

## Data and training pipeline

- Source data: [DailyDialog](https://huggingface.co/datasets/ConvLab/dailydialog).
- `src/data/preprocess.py` lowercases utterances, collapses whitespace, and creates sliding examples of three context words followed by one target word. Punctuation remains attached to tokens.
- Conversations are shuffled with seed 42 and split by dialogue into 80% train, 10% validation, and 10% test before examples are generated. This keeps utterances from one dialogue in one split; the same three-word context can still occur in different conversations with different targets.
- `src/data/build_vocab.py` builds the vocabulary from the training split. `Vocabulary` reserves ID 0 for `<pad>` and ID 1 for `<unk>`; known vocabulary IDs begin at 2.
- Training and inference both use the same lowercasing, whitespace tokenization, vocabulary encoder, left-padding, and most-recent-context truncation.
- For context lengths other than three, the trainer prepares matching `data/processed_ctxN/` splits from the raw dialogues if needed. It rejects examples with the wrong context length. The vocabulary remains the shared `data/processed/vocab.json`.
- The default configuration uses an embedding of 128 dimensions, one recurrent layer with 256 units, dropout 0.2, Adam at 0.001, sparse categorical cross-entropy, batch size 256, and seed 42. Training allows up to 20 epochs with early stopping on validation loss (patience 3). Check each saved run's config for its actual settings.
- Best-validation-loss and final checkpoints are saved under `artifacts/models/`. The shared evaluator reports Top-1, Top-3, Top-5, cross-entropy, perplexity, MSE, MAE, RMSE, parameter count, and inference latency.

Run an experiment from the repository root:

```bash
python -m src.training.train_unified --model gru
python -m src.training.train_unified --model rnn
python -m src.training.train_unified --model lstm
```

For full-data GPU training outside this computer, use [the Colab guide](COLAB_TRAINING.md) and `notebooks/train_all_colab.ipynb`. There are three architectures and nine runs across context lengths 3, 5, and 10. The three-word runs use all 678,869 training, 89,517 validation, and 85,945 test examples; longer contexts use their complete generated splits. The test split is evaluated after training.

For a short diagnostic run, pass `--epochs 1 --max-train-samples 10000 --max-validation-samples 2000 --max-test-samples 2000`. These limits produce preliminary results, not full-data scores. On Windows, `train_all.bat` runs all three architectures with the shared trainer. The current local `artifacts/` and `results/` directories are ignored by Git; check `results/*_config.json` for the settings of each local checkpoint.

## Educational simulation

Start the API and app in separate terminals:

```bash
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

In the **Predict** tab, choose context length 3, 5, or 10, then an available model or Compare All. Run custom text or select a row from the matching train, validation, or test split. The simulation shows:

1. The input text and its real token IDs.
2. The selected recurrent architecture and the state or gates it uses.
3. The model's actual ranked probability distribution.
4. The stored target, its probability and rank, and whether it was top-ranked.
5. A side-by-side model summary when Compare All is selected.

Dataset examples are identified by split and row so repeated contexts with conflicting labels remain distinguishable. For custom text, ground truth is shown only when the final three-word context has one unambiguous label in the processed data. Unknown or ambiguous contexts still show predictions. If a target is absent from the training vocabulary, the app labels the reported score as the `<unk>` bucket probability rather than claiming it is the exact word's probability.

The API also provides `GET /health`, `GET /examples`, and `POST /predict`. Pass `context_size` (3, 5, or 10) to select matching checkpoints and dataset rows; the dashboard provides this selector. A prediction response includes the checkpoint path, model context length, tokenization, IDs, ranked words, and ground-truth probability/rank when a label is available. Set `GRU_MODEL_PATH`, `RNN_MODEL_PATH`, or `LSTM_MODEL_PATH` to override a context-3 checkpoint; use names such as `GRU_CTX5_MODEL_PATH` for longer contexts.

Set `include_trace` to `true` in `POST /predict` to inspect the checkpoint's actual embedding values, recurrent hidden state at each input position, and output scores before softmax. The dashboard requests this trace automatically. When a custom three-word context has multiple different targets in DailyDialog, choose an exact dataset row to score its stored next word; the input alone cannot identify which row was intended.

## Results and interpretation

The Model comparison tab reads actual metric files in `results/`; it shows no score for a missing run. Local checkpoints and results are not committed. Check each run's config and sample count before comparing models. Top-1 accuracy is the fraction of rows where the single highest-ranked word matches the stored target; Top-3 and Top-5 check whether it appears among the first three or five. Cross-entropy and perplexity assess the probability assigned to the reference target.

One exact dataset row is `vegetarian supreme and → one` in the training split. A row being present in training does not guarantee that a short training run will memorize it. Also, `and one double` occurs with different targets in more than one split, so a context without a selected row does not always have one unique ground truth.

## Project files

- `src/data/`: preprocessing, tokenization, vocabulary, and labelled example lookup.
- `src/models/`: GRU, vanilla RNN, and LSTM architectures.
- `src/training/`: shared data encoding, training, and evaluation.
- `src/inference/predictor.py`: checkpoint loading, token encoding, prediction, and target scoring.
- `api/main.py`: FastAPI health, examples, and prediction endpoints.
- `app/streamlit_app.py`: interactive comparative simulation and experiment metrics.
- `tests/`: vocabulary, data, evaluation, model, and inference checks.

Run the test suite with:

```bash
python -m unittest discover tests -v
```

The synthetic training sanity test is opt-in: set `CONTEXTNEXT_RUN_TRAINING_TESTS=1` before running the suite if you want that test to fit tiny models locally. The default suite performs no training.
