# Person B — Vanilla RNN Presentation Script

**Estimated speaking time:** about 1 minute 35 seconds

## Script

“I’ll explain the vanilla recurrent neural network, our baseline for predicting a next word from three DailyDialog words. Each word ID becomes a learned 128-number embedding. The SimpleRNN reads these in order. At step t, it combines the current input with the previous hidden state, applies tanh, and produces a new state. This repeats for all three words; the final state represents the context.

The vanilla RNN has no gates or separate cell state, so context passes through its hidden state. During backpropagation, gradients are multiplied through recurrent steps and can shrink across long sequences. This vanishing-gradient problem makes distant information harder to learn. Our short context makes it a useful baseline for gated models.

The implementation uses 256 units and dropout 0.2. A dense layer scores 35,846 words; softmax turns scores into probabilities. Adam and sparse categorical cross-entropy train it. Validation loss selects the best checkpoint and guides early stopping.

The current saved diagnostic run has 13,899,270 parameters. It reached 3.65 percent Top-1, 8.65 percent Top-3, and 15.10 percent Top-5 accuracy, with cross-entropy 8.010 and perplexity 3,010.33. In this CPU run it also had the lowest measured training time, 11.05 seconds, and inference latency, 16.409 milliseconds. Each checkpoint completed one epoch on at most 10,000 training examples, so these scores are preliminary.”

## Key points to remember

- The hidden state is updated once for each input word, in sequence order.
- The model has no gates or separate cell state; its simplicity is useful as a baseline.
- Vanishing gradients are a possible limitation, especially for longer sequences.
- Current saved RNN run: 3.65% Top-1, 8.65% Top-3, 15.10% Top-5, cross-entropy 8.0098, perplexity 3,010.33.
