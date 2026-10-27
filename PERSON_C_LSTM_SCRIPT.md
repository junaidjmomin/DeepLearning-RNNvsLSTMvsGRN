# Person C — LSTM Presentation Script

**Estimated speaking time:** about 1 minute 35 seconds

## Script

“I’ll explain the LSTM, or Long Short-Term Memory network. It predicts the next word from recent DailyDialog words. Under the default settings, their IDs become learned 128-number embeddings, read in sequence.

An LSTM carries a hidden state and a cell state. The hidden state moves to the next word and supports prediction; the cell state carries selected memory. Three learned gates manage it. Sigmoid gives gate values between zero and one: the forget gate scales old memory, the input gate controls what candidate information to add, and the output gate controls what becomes the new hidden state. The new cell combines retained and selected information. These decisions are learned, not hand-written.

The default model uses 256 units, dropout 0.2, and masks padding. Its final state is projected to the shared vocabulary; softmax gives probabilities. Adam and sparse categorical cross-entropy train it, while validation loss selects the best checkpoint and controls early stopping.

For the live demonstration, I’ll read the LSTM’s actual checkpoint settings and test metrics from the Model comparison tab. The Predict tab shows its hidden state, cell state, real word probabilities, and the stored next word for a selected dataset row. DailyDialog uses one reference word even when other continuations are plausible.”

## Key points to remember

- Cell state carries selected memory; hidden state is the step output and final prediction representation.
- Forget, input, and output gates regulate the cell-state and hidden-state flow.
- Gates are learned during training, rather than manually specifying which words matter.
- Read current scores from the dashboard; checkpoints and result files are local and may change after retraining.
