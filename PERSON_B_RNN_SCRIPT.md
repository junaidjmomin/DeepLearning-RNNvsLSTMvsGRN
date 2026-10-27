# Person B — Vanilla RNN Presentation Script

**Estimated speaking time:** about 1 minute 35 seconds

## Script

“I’ll explain the vanilla recurrent neural network, our baseline for predicting a next word from recent DailyDialog words. Under the default settings, each word ID becomes a learned 128-number embedding. The SimpleRNN reads these in order. At step t, it combines the current input with the previous hidden state, applies tanh, and produces a new state. This repeats for every context word; the final state represents the context.

The vanilla RNN has no gates or separate cell state, so context passes through its hidden state. During backpropagation, gradients are multiplied through recurrent steps and can shrink across long sequences. This vanishing-gradient problem makes distant information harder to learn. Our short context makes it a useful baseline for gated models.

The default configuration uses 256 units and dropout 0.2. A dense layer scores the shared vocabulary; softmax turns scores into probabilities. Adam and sparse categorical cross-entropy train it. Validation loss selects the best checkpoint and guides early stopping.

For the live demonstration, I’ll read the RNN’s actual checkpoint settings and test metrics from the Model comparison tab. The Predict tab shows its real hidden-state updates and whether its top word matches the selected dataset row.”

## Key points to remember

- The hidden state is updated once for each input word, in sequence order.
- The model has no gates or separate cell state; its simplicity is useful as a baseline.
- Vanishing gradients are a possible limitation, especially for longer sequences.
- Read current scores from the dashboard; checkpoints and result files are local and may change after retraining.
