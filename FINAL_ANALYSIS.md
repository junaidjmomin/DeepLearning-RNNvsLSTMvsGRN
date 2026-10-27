# ContextNext — Pipeline Audit and Current Model Analysis

**Task:** predict the next word from a three-word context.
**Dataset:** DailyDialog, split by dialogue into train, validation, and test.
**Vocabulary:** 35,844 tokens from the training split, plus `<pad>` (ID 0) and `<unk>` (ID 1), for 35,846 model outputs.

## Findings from the pipeline audit

### Data and model training

`src/data/preprocess.py` lowercases each utterance, collapses whitespace, and creates one example for each three-word context and its next word. Punctuation is kept as part of a token. Dialogues are shuffled with seed 42 and split 80/10/10 before examples are created. This prevents dialogue-level leakage between splits, though identical short contexts can occur in different dialogues and may have different next words.

`src/data/build_vocab.py` builds the vocabulary from training examples. `Vocabulary.from_json` orders those words by their stored IDs and reserves IDs 0 and 1 for padding and unknown words. The shared `load_examples` function encodes both context words and targets through `Vocabulary.encode`, left-pads short inputs, and keeps the most recent words when an input exceeds the model context length.

All three saved architectures use an embedding of 128 dimensions, one recurrent layer with 256 units, dropout 0.2, and a 35,846-class softmax. The shared trainer uses Adam at 0.001, sparse categorical cross-entropy, batch size 256, seed 42, and early stopping on validation loss with patience 3. It saves both the best-validation and final checkpoints.

The current result configurations show **one epoch**, at most **10,000 training examples**, and **2,000 validation and test examples** for each model. These are short diagnostic runs, not full-data training. In the current environment TensorFlow 2.22.0-rc0 detects no GPU.

### Inference bug

The old predictor added 2 to each known token ID after loading `Vocabulary.token_to_id`. Those IDs already include the padding and unknown-token offset used by training. It also added 2 again when building the output ID-to-word map. As a result, inference supplied different IDs from training and decoded model output IDs as different words.

The predictor now calls the same `Vocabulary.encode` used by training and decodes with its inverse mapping. For the diagnostic input, the corrected IDs are:

```text
vegetarian → 33868
supreme    → 30927
and        → 2461
```

All three loaded checkpoints accept a context of length 3 and return 35,846 probabilities, matching the shared vocabulary.

### Labelled examples and duplicate contexts

The processed data contains:

| Context | Split and row | Stored target |
|---|---|---|
| `vegetarian supreme and` | train, row 7 | `one` |
| `supreme and one` | train, row 8 | `double` |
| `and one double` | train, row 9 | `pepperoni.` |
| `and one double` | validation, row 50,011 | `room` |

The last context has two ground-truth labels, so the app lets a learner select the exact split row. For custom input, it reports ground truth only when there is one distinct matching label. Otherwise, it shows the candidate labels and still displays predictions.

## Current saved-run metrics

Top-1, Top-3, and Top-5 are the share of test rows where the reference target appears in the first 1, 3, or 5 predictions. Cross-entropy and perplexity use the model probability assigned to the one stored target. Top-5 was added by the shared evaluator and recalculated from the saved checkpoints on the same 2,000 test rows.

| Metric | GRU | Vanilla RNN | LSTM |
|---|---:|---:|---:|
| Top-1 accuracy | 3.65% | 3.65% | 3.65% |
| Top-3 accuracy | 9.55% | 8.65% | **10.10%** |
| Top-5 accuracy | 15.10% | 15.10% | 15.10% |
| Cross-entropy ↓ | 8.1008 | 8.0098 | **7.9414** |
| Perplexity ↓ | 3,296.96 | 3,010.33 | **2,811.19** |
| RMSE ↓ | 0.005279 | **0.005267** | 0.005282 |
| Parameters | 14,097,158 | **13,899,270** | 14,194,950 |
| Training time | 11.18 s | **11.05 s** | 11.73 s |
| Warmed CPU inference latency | 21.666 ms | **16.409 ms** | 22.364 ms |

The LSTM has the lowest cross-entropy and perplexity and the best Top-3 score in these runs. Top-1 and Top-5 are tied. The RNN has the lowest parameter count and measured latency in this local run. These small differences are preliminary because each checkpoint completed only one epoch on a capped training subset; the latency numbers are also specific to this CPU environment.

## Exact-input model outputs

For `vegetarian supreme and`, the dataset target is `one`. The corrected predictor produces these actual checkpoint outputs:

| Model | Top prediction | Probability | `one` probability | Rank of `one` | Correct? |
|---|---|---:|---:|---:|---|
| Vanilla RNN | `the` | 4.1431% | 0.4924% | 41 | No |
| GRU | `the` | 3.3585% | 0.3528% | 49 | No |
| LSTM | `the` | 0.2758% | 0.0569% | 58 | No |

The word `one` is present in the training example, but it is not the top prediction from these saved checkpoints. The exact row appears at index 7, is included in the first 10,000 training samples, and was seen during one epoch. Its presence in the training data is not a guarantee that a short run memorizes it.

### Controlled training sanity check

To distinguish an inference/vocabulary issue from a basic training issue, a temporary three-row dataset used these mappings:

```text
vegetarian supreme and → one
supreme and one        → double
and one double         → pepperoni.
```

The shared vocabulary loader, example encoder, model builders, and evaluator were used with small 8-dimensional embeddings and 16 recurrent units. Each architecture trained on repeated copies of the rows for 40 epochs. Training loss decreased for all three, and each reached 100% Top-1 on the three controlled examples. The check wrote no weights or metrics into the project artifacts. This establishes that the shared training path can learn the small mapping; it does not make claims about full DailyDialog quality.

## Application behavior

The Streamlit Predict tab is an educational simulation. It supports custom text, split-specific dataset examples, an individual architecture or Compare All, and displays actual token IDs, architecture steps, ranked probabilities, target probability/rank, and correctness. `<unk>` probability is labelled as the unknown-token bucket when the reference target is absent from the training vocabulary.

The FastAPI service exposes `GET /health`, `GET /examples`, and `POST /predict`. Health identifies the checkpoint file and model context length actually loaded. The prediction response includes the tokenization, encoded input, ranked words, and ground-truth result when a label can be determined.

## Metric artifacts

The current files under `results/` contain the saved run configuration, history, and recalculated test metrics. The inference audit does not alter model weights. To obtain representative final scores, retrain all three models with the same complete train/validation/test splits and aligned settings, then compare the resulting saved metrics.
