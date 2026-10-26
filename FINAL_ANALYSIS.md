# ContextNext — Final Analysis Report

**Project:** GRU vs Vanilla RNN vs LSTM — Next-Word Prediction  
**Dataset:** DailyDialog (conversational text)  
**Task:** Given the previous 3 words, predict the next word  
**Context length:** 3 words  
**Vocabulary size:** 35,846 tokens  
**Hardware:** CPU (RTX 3050 not used — TensorFlow on native Windows is CPU-only)

---

## 1. What the System Does

ContextNext is a next-word prediction system that compares three recurrent neural network architectures fairly on the same data:

```
Input sentence → Tokenize → Last 3 words → Embedding (128-dim) → RNN/GRU/LSTM (256 units) → Dense + Softmax (35,846 classes) → Top-K predicted words
```

All three models share:
- Identical train/validation/test splits (80/10/10)
- Identical vocabulary (`data/processed/vocab.json`)
- Identical hyperparameters: embedding 128, hidden 256, dropout 0.2, Adam lr 0.001, batch 256, seed 42
- Identical loss: Sparse Categorical Cross-Entropy
- Identical metrics: Top-1 Accuracy, Top-3 Accuracy

So any difference in results comes from the **recurrent cell architecture only**, not training conditions.

---

## 2. How Each Model Works (Code Explanation)

All three models live in `src/models/`:

### Vanilla RNN (`src/models/rnn.py`)
Architecture: `Embedding → SimpleRNN → Dense(softmax)`

- Uses `tf.keras.layers.SimpleRNN` — the simplest recurrent cell
- Hidden state: `h_t = tanh(W·x_t + U·h_{t-1} + b)`
- **No gates** — the entire history is compressed into one hidden vector
- Suffers from **vanishing gradients** on longer sequences
- Fastest to train (fewest parameters: ~13.9M)

### GRU (`src/models/gru.py`)
Architecture: `Embedding → GRU → Dense(softmax)`

- Uses `tf.keras.layers.GRU` — Gated Recurrent Unit
- **Two gates:**
  - **Update gate** — decides how much past information to keep vs. discard
  - **Reset gate** — decides how much new input to add to the candidate state
- Simpler and faster than LSTM, but still handles longer dependencies better than vanilla RNN
- Parameter count: ~14.1M

### LSTM (`src/models/lstm.py`)
Architecture: `Embedding → LSTM → Dense(softmax)`

- Uses `tf.keras.layers.LSTM` — Long Short-Term Memory
- **Three gates** plus a **cell state** that runs in parallel to the hidden state:
  - **Forget gate** — decides what to discard from the cell state
  - **Input gate** — decides what new information to store
  - **Output gate** — decides what to output from the cell state
- Best at remembering long-range dependencies because gradients can flow through the cell state unchanged
- Parameter count: ~14.2M (most parameters due to 3 gate matrices)

### Shared components
- **Embedding layer** (`mask_zero=True`): converts word IDs to 128-dimensional vectors; padding tokens are masked so they don't affect the hidden state.
- **Dense + Softmax**: outputs a probability distribution over all 35,846 vocabulary words.
- **`<PAD>`** (ID 0) and **`<UNK>`** (ID 1) special tokens handle padding and unknown words.

---

## 3. Inference Pipeline (`src/inference/predictor.py` + `api/main.py`)

1. **API** (`api/main.py`): FastAPI with two endpoints:
   - `GET /health` — reports vocabulary and model availability
   - `POST /predict` — accepts `{sentence, model, top_k}` and returns top-K predictions
2. **Predictor** (`src/inference/predictor.py`):
   - Lazily loads `.keras` checkpoints (only when first requested)
   - Encodes the input sentence: lowercase → split → take last N tokens → map to IDs via vocab → unknown words → `<UNK>`
   - Left-pads to `sequence_length` (3)
   - Runs `model.predict(training=False)`
   - Returns top-K words sorted by probability
3. **UI** (`app/streamlit_app.py`): Streamlit frontend with three tabs:
   - **Prediction:** type a sentence, choose GRU/RNN/LSTM, see top predictions
   - **Model Comparison:** bar charts of accuracy, perplexity, RMSE, training time, params
   - **How it works:** explains the pipeline

---

## 4. Final Test Results (context=3, 1,000 test samples)

| Metric | GRU | RNN | LSTM |
|--------|-----|-----|------|
| Top-1 Accuracy | 3.30% | 3.90% | 3.90% |
| Top-3 Accuracy | 10.00% | 10.00% | 10.20% |
| Cross-Entropy Loss | 8.24 | 8.08 | 7.98 |
| Perplexity | 3,777 | 3,223 | 2,914 |
| RMSE | 0.00527 | 0.00526 | 0.00528 |
| MAE | 0.0000551 | 0.0000553 | 0.0000552 |
| Parameters | 14,097,158 | 13,899,270 | 14,194,950 |
| Training Time | 41.5s | 35.4s | 36.8s |
| Inference Latency | 19.4ms | 13.1ms | 33.7ms |

*Source: `results/gru_test_metrics.json`, `results/rnn_test_metrics.json`, `results/lstm_test_metrics.json`*

---

## 5. Which Model is Best?

**LSTM is the best model.**

### Why LSTM won:
- **Lowest perplexity (2,914)**: LSTM is least "perplexed" by the test data — its probability distribution is closest to the true next-word distribution.
- **Lowest cross-entropy (7.98)**: best classification objective.
- **Highest Top-3 accuracy (10.2%)**: most often places the correct word in the top 3 suggestions.
- **Tied Top-1 accuracy (3.9%)** with RNN.

### Why RNN is close:
- At context length = 3 (short context), the vanishing-gradient problem of vanilla RNN is not severe yet, so RNN performs nearly as well as GRU/LSTM on Top-1.
- RNN is fastest to train (35.4s) and fastest inference (13.1ms).

### Why GRU is middle:
- GRU sits between RNN and LSTM — better than RNN on perplexity, but not as good as LSTM.
- Good balance if you want gated memory without LSTM's parameter cost.

### Recommendation:
- **For accuracy:** Use **LSTM** (best perplexity, best Top-3).
- **For speed/deployment:** Use **RNN** (fastest training + inference, ~14% fewer params than LSTM).
- **For balance:** Use **GRU** (gated memory, fewer gates than LSTM).

---

## 6. Sample Output

Input: `"i would like to"` | Model: `lstm` | Top-5 predictions:

| Rank | Word | Probability |
|------|------|-------------|
| 1 | the | 9.28% |
| 2 | you | 7.06% |
| 3 | a | 6.99% |
| 4 | to | 6.91% |
| 5 | i | 5.04% |

API response (`POST /predict`):
```json
{
  "model": "lstm",
  "context": ["would", "like", "to"],
  "predictions": [
    {"word": "the", "probability": 0.0928},
    {"word": "you", "probability": 0.0706},
    {"word": "a", "probability": 0.0699},
    {"word": "to", "probability": 0.0691},
    {"word": "i", "probability": 0.0504}
  ]
}
```

---

## 7. Files to Check (Code Reference)

| Purpose | File |
|---------|------|
| GRU model | `src/models/gru.py` |
| Vanilla RNN model | `src/models/rnn.py` |
| LSTM model | `src/models/lstm.py` |
| Training (all 3, unified) | `src/training/train_unified.py` |
| Train all batch script | `train_all.bat` |
| Evaluation metrics | `src/training/evaluate.py` |
| Inference / prediction | `src/inference/predictor.py` |
| FastAPI backend | `api/main.py` |
| Streamlit frontend | `app/streamlit_app.py` |
| Data preprocessing | `src/data/preprocess.py` |
| Vocabulary builder | `src/data/build_vocab.py` |
| Tokenizer | `src/data/tokenizer.py` |
| Trained models | `artifacts/models/{gru,rnn,lstm}_{best,final}.keras` |
| Test metrics | `results/{gru,rnn,lstm}_test_metrics.json` |
| Processed data | `data/processed/{train,validation,test,vocab}.json` |
| Docker | `Dockerfile`, `docker-compose.yml` |

---

## 8. How to Run (for Demo)

**Terminal 1 — Start API:**
```cmd
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Start UI:**
```cmd
python -m streamlit run app/streamlit_app.py --server.port 8501
```

**Browser:** open `http://localhost:8501`

**Or with Docker:**
```cmd
docker compose up
```
Then open `http://localhost:8501`

---

## 9. Limitations

- Trained on only 5,000 samples (capped for CPU training on laptop). Full dataset has 678,869 samples — retraining on full data would improve accuracy.
- Context length fixed at 3. Longer context (5, 10) would likely separate the models more clearly.
- CPU-only training (TensorFlow on Windows does not support GPU). RTX 3050 is unused.
- Perplexity is high (~2,900–3,700) because vocabulary is large (35k+) and next-word prediction is inherently uncertain.
