# Person C — LSTM Presentation Script

**Estimated speaking time:** about 1 minute 35 seconds

## Script

“I’ll explain the LSTM, or Long Short-Term Memory network. It predicts the next word from three DailyDialog words. Their IDs become learned 128-number embeddings, read in sequence.

An LSTM carries a hidden state and a cell state. The hidden state moves to the next word and supports prediction; the cell state carries selected memory. Three learned gates manage it. Sigmoid gives gate values between zero and one: the forget gate scales old memory, the input gate controls what candidate information to add, and the output gate controls what becomes the new hidden state. The new cell combines retained and selected information. These decisions are learned, not hand-written.

Our model uses 256 units, dropout 0.2, and masks padding. Its final state is projected to 35,846 scores; softmax gives probabilities. Adam and sparse categorical cross-entropy train it, while validation loss selects the best checkpoint and controls early stopping.

The current saved diagnostic run has 14,194,950 parameters. It reached 3.65 percent Top-1, 10.10 percent Top-3, and 15.10 percent Top-5 accuracy, with cross-entropy 7.941 and perplexity 2,811.19. It leads these runs on Top-3, cross-entropy, and perplexity, while Top-1 and Top-5 are tied. The checkpoint completed one epoch on at most 10,000 training examples, so these results are preliminary. DailyDialog evaluation also uses one reference word even when other continuations are plausible.”

## Key points to remember

- Cell state carries selected memory; hidden state is the step output and final prediction representation.
- Forget, input, and output gates regulate the cell-state and hidden-state flow.
- Gates are learned during training, rather than manually specifying which words matter.
- Current saved LSTM run: 3.65% Top-1, 10.10% Top-3, 15.10% Top-5, cross-entropy 7.9414, perplexity 2,811.19.
