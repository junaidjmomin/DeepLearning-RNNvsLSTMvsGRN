# Person A — GRU Presentation Script

**Estimated speaking time:** about 1 minute 35 seconds

## Script

“I’ll present the GRU, or Gated Recurrent Unit. ContextNext predicts a word from three DailyDialog words, such as ‘I would like’ → ‘to.’ Word IDs become learned 128-number embeddings. The GRU reads them in order and updates a hidden state summarizing the context. Padding is masked.

Two learned gates control each update. Sigmoid gives values between zero and one. The reset gate controls how much old state informs a candidate; the update gate balances it against the old state. This retains useful context without an LSTM’s separate cell state.

Our GRU has 256 units and dropout 0.2. A dense layer scores 35,846 words; softmax turns scores into probabilities. Adam and sparse categorical cross-entropy train it, while validation loss selects the best checkpoint and controls early stopping.

In the current saved diagnostic run, the GRU has 14,097,158 parameters. It reached 3.65 percent Top-1, 9.55 percent Top-3, and 15.10 percent Top-5 accuracy, with cross-entropy 8.101 and perplexity 3,296.96. These checkpoints completed one epoch using at most 10,000 training examples, so the figures are preliminary rather than full-training results.”

## Key points to remember

- Reset gate: controls how much past state informs the candidate update.
- Update gate: controls the balance between previous and candidate state.
- The final softmax is over the shared vocabulary; its values represent predicted probabilities.
- Current saved GRU run: 3.65% Top-1, 9.55% Top-3, 15.10% Top-5, cross-entropy 8.1008, perplexity 3,296.96.
