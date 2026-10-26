# ContextNext: Three-Model Presentation Notes

## Quick summary

All three models solve the same task: use the previous three words from a DailyDialog utterance to predict the next word. Each receives integer word IDs, looks up a learned 128-dimensional embedding, processes the sequence with one recurrent layer of 256 units, and uses a dense softmax layer to score the 35,846 vocabulary entries. Padding is masked. The shared training setup uses dropout 0.2, Adam with learning rate 0.001, sparse categorical cross-entropy, Top-1 and Top-3 accuracy, batch size 256, seed 42, and early stopping on validation loss (patience 3; maximum 20 epochs).

| Presenter | Model | Main mechanism | Reported parameters |
|---|---|---|---:|
| Person A | GRU | Reset and update gates regulate hidden-state information | 14,097,158 |
| Person B | Vanilla RNN | Simple recurrent hidden state, with no gates | 13,899,270 |
| Person C | LSTM | Forget, input, and output gates plus a cell state | 14,194,950 |

The training scripts save a best-validation checkpoint and a final checkpoint, then evaluate on the test split. The shared evaluator reports Top-1/Top-3 accuracy, cross-entropy, perplexity, MSE, MAE, RMSE, parameter count, and inference latency; training scripts also record duration and experiment settings.

## Spoken script (about 4 minutes; under 5 minutes)

### Person A — GRU (about 1 minute)

“Our project, ContextNext, compares three recurrent neural networks on next-word prediction using DailyDialog conversations. For every example, the model receives the previous three words and predicts the next word. All three models use the same vocabulary and training setup so the main difference is the recurrent layer.

I’ll cover the GRU, or Gated Recurrent Unit. In our code, the input word IDs first pass through a learned embedding layer. Each word becomes a vector of 128 numbers. The GRU then processes those vectors in order using 256 hidden units. Its reset and update gates regulate which information from the past to use and what to carry forward. A dense softmax layer turns the final state into probabilities over the 35,846-word vocabulary.

The GRU training script uses Adam, sparse categorical cross-entropy, dropout, and early stopping based on validation loss. The final analysis reports 14.1 million parameters. It reports 3.3 percent Top-1 accuracy and 10 percent Top-3 accuracy. These are the reported results for this run, not a claim that the GRU is best for every dataset.”

### Person B — Vanilla RNN (about 1 minute)

“The vanilla RNN is our simplest recurrent baseline. It uses the same embedding and output layers as the other models, but its sequence-processing layer is a SimpleRNN. It reads each word in order and updates a hidden state. That state is its compact representation of the words seen so far.

Unlike GRU and LSTM, the vanilla RNN has no gates to control information flow. This makes it simpler, but it can have difficulty learning long-range dependencies because gradients can become very small during training. For our three-word context, it provides a useful baseline for seeing what the gated models add.

The RNN uses 256 recurrent units and the same shared training settings. Its reported parameter count is about 13.9 million. The analysis reports 3.9 percent Top-1 accuracy, 10 percent Top-3 accuracy, and perplexity of 3,223. It also reports the fastest training time and inference latency in this comparison. These speed results are specific to the reported hardware and run.”

### Person C — LSTM and comparison (about 1 minute 30 seconds)

“The LSTM, or Long Short-Term Memory network, also processes the embedded words in sequence, but it maintains a cell state alongside its hidden state. Three gates help manage that memory: the forget gate chooses what information to discard, the input gate chooses what new information to store, and the output gate chooses what to expose as the next hidden state. This design is intended to help preserve useful information across longer sequences.

In our implementation, the LSTM has 256 units and uses the same embedding, dropout, optimizer, loss, and evaluation setup as the other models. Its reported parameter count is about 14.2 million.

According to our final analysis, LSTM has the lowest cross-entropy at 7.98 and lowest perplexity at 2,914, and the highest Top-3 accuracy at 10.2 percent. It ties the RNN at 3.9 percent Top-1 accuracy. The RNN is faster in the reported run, so there is a trade-off between prediction scores and speed.

These results use a three-word context and, according to the report, a limited training sample. Dialogue can have several plausible next words even though evaluation uses one reference word. Also, the report’s metric files and model checkpoints are not in the reviewed checkout, so these figures are reported results that we could not independently verify from the saved artifacts.”

