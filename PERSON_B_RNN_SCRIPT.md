# Person B — Vanilla RNN Presentation Script

**Estimated speaking time:** about 1 minute 35 seconds

## Script

“I’ll explain the vanilla recurrent neural network, our baseline for predicting a next word from three DailyDialog words. Each word ID becomes a learned 128-number embedding. The SimpleRNN reads these in order. At step t, it combines the current input with the previous hidden state, applies tanh, and produces a new state. This repeats for all three words; the final state represents the context.

The vanilla RNN has no gates or separate cell state, so context passes through its hidden state. During backpropagation, gradients are multiplied through recurrent steps and can shrink across long sequences. This vanishing-gradient problem makes distant information harder to learn. Our short context makes it a useful baseline for gated models.

The implementation uses 256 units and dropout 0.2. A dense layer scores 35,846 words; softmax turns scores into probabilities. Adam and sparse categorical cross-entropy train it. Validation loss selects the best checkpoint and guides early stopping.

The analysis reports 13,899,270 parameters, 3.9 percent Top-1, 10 percent Top-3, cross-entropy 8.08, and perplexity 3,223. It reports fastest training at 35.4 seconds and inference at 13.1 milliseconds. The cited files are absent, so these figures cannot be independently verified.”

## Key points to remember

- The hidden state is updated once for each input word, in sequence order.
- The model has no gates or separate cell state; its simplicity is useful as a baseline.
- Vanishing gradients are a possible limitation, especially for longer sequences.
- Project-reported RNN figures: 3.90% Top-1, 10.00% Top-3, cross-entropy 8.08, perplexity 3,223.
