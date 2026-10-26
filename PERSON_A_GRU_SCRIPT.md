# Person A — GRU Presentation Script

**Estimated speaking time:** about 1 minute 35 seconds

## Script

“I’ll present the GRU, or Gated Recurrent Unit. ContextNext predicts a word from three DailyDialog words, such as ‘I would like’ → ‘to.’ Word IDs become learned 128-number embeddings. The GRU reads them in order and updates a hidden state summarizing the context. Padding is masked.

Two learned gates control each update. Sigmoid gives values between zero and one. The reset gate controls how much old state informs a candidate; the update gate balances it against the old state. This retains useful context without an LSTM’s separate cell state.

Our GRU has 256 units and dropout 0.2. A dense layer scores 35,846 words; softmax turns scores into probabilities. Adam and sparse categorical cross-entropy train it, while validation loss selects the best checkpoint and controls early stopping.

The final analysis reports 14,097,158 parameters, 3.3 percent Top-1 and 10 percent Top-3 accuracy, cross-entropy 8.24, and perplexity 3,777. Top-1 checks if the target ranks first; Top-3 checks if it appears in the first three. The report’s metric files and checkpoints are absent here, so these figures cannot be independently verified.”

## Key points to remember

- Reset gate: controls how much past state informs the candidate update.
- Update gate: controls the balance between previous and candidate state.
- The final softmax is over the shared vocabulary; its values represent predicted probabilities.
- Project-reported GRU figures: 3.30% Top-1, 10.00% Top-3, cross-entropy 8.24, perplexity 3,777.
