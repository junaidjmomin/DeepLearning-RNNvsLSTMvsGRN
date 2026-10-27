# Person A — GRU Presentation Script

**Estimated speaking time:** about 1 minute 35 seconds

## Script

“I’ll present the GRU, or Gated Recurrent Unit. ContextNext predicts a word from recent DailyDialog words, such as ‘I would like’ → ‘to.’ Under the default settings, word IDs become learned 128-number embeddings. The GRU reads them in order and updates a hidden state summarizing the context. Padding is masked.

Two learned gates control each update. Sigmoid gives values between zero and one. The reset gate controls how much old state informs a candidate; the update gate balances it against the old state. This retains useful context without an LSTM’s separate cell state.

The default GRU configuration has 256 units and dropout 0.2. A dense layer scores the shared vocabulary; softmax turns scores into probabilities. Adam and sparse categorical cross-entropy train it, while validation loss selects the best checkpoint and controls early stopping.

For the live demonstration, I’ll read the GRU’s actual checkpoint settings and test metrics from the Model comparison tab. The Predict tab then shows its real next-word probabilities and hidden-state trace for the selected input.”

## Key points to remember

- Reset gate: controls how much past state informs the candidate update.
- Update gate: controls the balance between previous and candidate state.
- The final softmax is over the shared vocabulary; its values represent predicted probabilities.
- Read current scores from the dashboard; checkpoints and result files are local and may change after retraining.
