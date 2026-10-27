# ContextNext: Live Prototype Presentation

All three models receive the same token IDs for a three-word context and predict a probability for each word in the shared training vocabulary. The default training configuration uses a 128-dimensional embedding, 256 recurrent units, dropout 0.2, Adam at 0.001, batch size 256, and early stopping. A saved run can use different settings, so read its actual configuration and metrics from the dashboard.

## Demonstration order

1. Open **Model comparison**. Check which checkpoints have result files, their sample counts, and their recorded settings. Explain that short diagnostic runs do not establish full-data model quality.
2. Open **Predict** and type a new three-word context. Show lowercasing, token IDs, the actual embedding and hidden-state trace, raw output scores, softmax probabilities, and the top predicted words.
3. Search for a training row and select its exact ID. The dashboard displays the stored next word, each model's top prediction, the stored word's probability and rank, and whether the top prediction was wrong.
4. Try a context with multiple dataset targets. Choose a specific row before calling one word the correct target; identical contexts can have different continuations.

## Architecture explanations

- **Vanilla RNN:** updates one hidden state for each word. It has no gates or separate cell state.
- **GRU:** reset and update gates regulate how new input and prior hidden state contribute to the next hidden state.
- **LSTM:** forget, input, and output gates regulate a cell state and hidden state.

The dashboard's numeric trace shows the saved checkpoint's actual embeddings, hidden states, and output scores. Its architecture text explains the gate design; individual gate activations are not displayed. The current metrics and checkpoints are local files under `results/` and `artifacts/models/`, so quote their values from the live dashboard rather than from a static script.
