# Person 2 — Vanilla RNN and Evaluation

Person 2 owns only the vanilla RNN experiment and its evaluation results. The GRU, LSTM, API, and deployment work are deliberately outside this scope.

## Choices made in this work

- **Three context words:** the existing preprocessing script uses a context of three words, so the RNN starts with the same fair input size.
- **Vanilla `SimpleRNN`:** this is the Person 2 baseline. GRU and LSTM remain work for the other two team members.
- **128-dimensional embeddings and 256 RNN units:** large enough for a meaningful college project while still practical to train on a normal laptop.
- **0.2 dropout:** a simple regularisation choice to reduce overfitting.
- **Cross-entropy, Top-1 accuracy, Top-3 accuracy and perplexity:** these are the main next-word prediction measures. MSE, MAE and RMSE are also reported because the project brief requires them.

## Train

Install the project dependencies, then run from the repository root:

```powershell
python -m src.training.train
```

The command uses the existing processed train, validation, test, and vocabulary files. It saves the best validation-loss checkpoint to `artifacts/models/rnn_best.keras`, the final checkpoint to `artifacts/models/rnn_final.keras`, and reproducible configuration/history/metrics JSON to `results/`.

For a quick pipeline check before a full run:

```powershell
python -m src.training.train --epochs 1 --max-train-samples 2000 --max-validation-samples 500 --max-test-samples 500
```

## Evaluate a saved checkpoint

```powershell
python -m src.training.evaluate --model artifacts/models/rnn_best.keras
```

`results/rnn_test_metrics.json` reports MSE, MAE, RMSE, cross-entropy, Top-1 accuracy, Top-3 accuracy, perplexity, parameter count, and warmed single-example inference latency. Training time and completed epochs are also added when evaluation is launched by the training command.

MSE, MAE, and RMSE compare the predicted softmax distribution with the one-hot target distribution. Cross-entropy, accuracy, and perplexity are the primary next-word prediction metrics.
