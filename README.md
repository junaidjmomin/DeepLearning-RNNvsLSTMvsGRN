## Final project definition

dataset: https://huggingface.co/datasets/ConvLab/dailydialog

### **ContextNext: Comparative Next-Word Prediction using GRU, RNN and LSTM**

**Problem statement:**  
Given the previous **N words** of a sentence, predict the next word and compare how GRU, Vanilla RNN and LSTM handle sequential context.

**Primary dataset:** DailyDialog.

**Core comparison:**

- GRU
- Vanilla RNN
- LSTM

**Required metrics:**

- MSE
- MAE
- RMSE
- Cross-entropy loss
- Top-1 Accuracy
- Top-3 Accuracy
- Perplexity
- Training time
- Inference latency
- Parameter count

The addition of proper language-model metrics is important. MSE/MAE/RMSE can be reported because your academic requirement asks for them, but accuracy, cross-entropy and perplexity make the evaluation much more defensible technically.

---

# Team of 3

For now call yourselves:

- **Person A — Data + GRU**
- **Person B — RNN + Evaluation**
- **Person C — LSTM + Deployment**

The workload becomes roughly equal because each person owns **one model + one major engineering responsibility**.

| Person | Model | Main engineering responsibility | Placement talking point |
|---|---|---|---|
| **A** | GRU | Dataset pipeline + preprocessing + testing | Data/ML pipeline |
| **B** | RNN | Experiment tracking + metrics + analysis | ML experimentation |
| **C** | LSTM | API + frontend + Docker/deployment | ML engineering/deployment |

But there is one rule:

> **Nobody should understand only their own part.**

At the end, each member must be capable of explaining GRU, RNN, LSTM, preprocessing, metrics and deployment.

---

# Phase 1 — Set up the project properly

### Day 1

Create **one GitHub repository** rather than three separate projects.

Use a structure like:

```text
contextnext/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_gru_experiments.ipynb
│   ├── 03_rnn_experiments.ipynb
│   └── 04_lstm_experiments.ipynb
│
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── gru.py
│   │   ├── rnn.py
│   │   └── lstm.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── inference/
│       └── predictor.py
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│
├── models/
├── results/
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

Do **not** put your entire project inside one `.ipynb`.

That immediately separates a classroom project from something that looks more professional.

### Work division

**Person A**

Creates repository structure, Python environment and preprocessing skeleton.

**Person B**

Sets up experiment configuration and results structure.

**Person C**

Creates initial application/API skeleton and Docker skeleton.

All three review the repository before continuing.

---

# Phase 2 — Understand the data

### Days 2–3

Download DailyDialog.

First inspect:

- number of conversations
- number of sentences
- average sentence length
- vocabulary size
- most frequent words
- sentence length distribution
- unknown/rare word frequency

### Person A owns this phase.

But Person B and C should participate in deciding preprocessing rules.

Do not perform aggressive NLP preprocessing.

For next-word prediction, you **should not remove stopwords**.

For example:

```text
I am going to the market
```

Removing stopwords could turn it into:

```text
going market
```

which destroys natural language structure.

Use roughly:

```text
lowercase
↓
basic punctuation handling
↓
tokenization
↓
frequency filtering
↓
vocabulary creation
↓
word → integer mapping
```

Keep special tokens:

```text
<PAD>
<UNK>
```

Possibly:

```text
<BOS>
<EOS>
```

---

# Phase 3 — Generate the actual ML problem

### Day 4

Start with:

### **5 previous words → predict word 6**

Example:

```text
Sentence:

I would like to book a hotel room
```

Training samples become:

```text
I would like to book → a

would like to book a → hotel

like to book a hotel → room
```

If vocabulary mapping is:

```text
I      = 14
would  = 73
like   = 27
to     = 5
book   = 321
a      = 2
```

Then the network actually receives:

```text
[14, 73, 27, 5, 321]
```

and target:

```text
2
```

This is the fundamental data representation all three team members need to understand.

### Critical fairness rule

Use the **exact same**:

```text
training examples
validation examples
test examples

vocabulary
tokenizer
embedding dimension
context length
batch size
optimizer
random seed
epochs
```

where possible.

Otherwise somebody can challenge you with:

> "How do you know LSTM performed better because of the architecture rather than different training conditions?"

You want a good answer to that.

---

# Phase 4 — Establish the baseline

Before neural networks, create a tiny baseline.

### Person B owns it.

Build something such as:

### Most Frequent Next Word / Bigram baseline

For:

```text
"I want to"
```

look at which words commonly follow `to`.

This gives you a baseline.

Suppose:

```text
Baseline accuracy: 14%
GRU:               30%
RNN:               31%
LSTM:              36%
```

Now the neural-network numbers actually mean something.

Without a baseline, saying:

> "LSTM got 36% accuracy"

doesn't tell the interviewer whether 36% is impressive.

---

# Phase 5 — Build GRU

### Days 5–6 — Person A

Architecture:

```text
5 input words
     ↓
Embedding
     ↓
GRU
     ↓
Last hidden state
     ↓
Dense
     ↓
Softmax
```

Conceptually:

```python
Embedding(vocab_size, embedding_dim)

GRU(embedding_dim, hidden_size)

Linear(hidden_size, vocab_size)
```

GRU maintains a hidden state and uses **gates** to control what information should be retained or updated.

It provides a simpler gated recurrent architecture compared with LSTM.

Person A must document:

```text
architecture
number of parameters
training loss
validation loss
training time
test metrics
```

---

# Phase 6 — Build Vanilla RNN

### Days 5–6 — Person B

In parallel with GRU.

Architecture:

```text
Words
 ↓
Embedding
 ↓
Vanilla RNN
 ↓
Last hidden state
 ↓
Dense
 ↓
Softmax
```

This architecture now processes:

```text
word1 → word2 → word3 → word4 → word5
```

rather than flattening everything immediately.

Record exactly the same metrics.

---

# Phase 7 — Build LSTM

### Days 5–6 — Person C

Architecture:

```text
Words
 ↓
Embedding
 ↓
LSTM
 ↓
Last hidden state
 ↓
Dense
 ↓
Softmax
```

Try to keep hidden dimensions comparable to RNN and GRU.

That gives you a fair:

```text
GRU
vs
RNN
vs
LSTM
```

experiment.

---

# Phase 8 — Merge everything into one training pipeline

### Days 7–8

This is extremely important.

You don't want:

```text
train_gru.py
train_rnn_randomcode.py
lstm_final_final2.py
```

Instead:

```bash
python train.py --model gru
python train.py --model rnn
python train.py --model lstm
```

Person B leads this.

Person A integrates GRU.

Person C integrates LSTM.

Your configuration might contain:

```python
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
```

Then everything is reproducible.

---

# Phase 9 — Add experiment tracking

### Day 9 — Person B

This is one of the easiest additions that makes the project look substantially more mature.

Use **MLflow**.

Log:

```text
model
learning rate
batch size
epochs
context length
embedding dimension
hidden dimension

training loss
validation loss
accuracy
top-3 accuracy
MSE
MAE
RMSE
perplexity

training duration
```

MLflow is designed specifically to track parameters, metrics, models and artifacts across training runs.

---

# Phase 10 — Evaluate the models correctly

### Days 10–11

Person B builds the common evaluation module.

Everyone verifies their model.

Create one final table:

| Metric | GRU | RNN | LSTM |
|---|---:|---:|---:|
| Cross Entropy | | | |
| Accuracy | | | |
| Top-3 Accuracy | | | |
| Perplexity | | | |
| MSE | | | |
| MAE | | | |
| RMSE | | | |
| Parameters | | | |
| Training Time | | | |
| Inference Time | | | |

### Why MSE/MAE/RMSE?

Your output is a probability distribution.

If:

```text
Target:

[0, 0, 1, 0]
```

and prediction is:

```text
[0.05, 0.10, 0.80, 0.05]
```

you can compute those errors against the one-hot target.

But explain during viva:

> "Because next-word prediction is a multiclass classification problem, cross-entropy, accuracy and perplexity are more directly meaningful. MSE, MAE and RMSE were additionally reported as comparative error measures."

That answer will sound far better than blindly claiming RMSE is the ideal metric.

---

# Phase 11 — Your most interesting experiment

### Days 12–13

Don't stop at:

> "GRU vs RNN vs LSTM."

Run:

```text
Context length = 3
Context length = 5
Context length = 10
```

Now test:

| Context | GRU | RNN | LSTM |
|---|---:|---:|---:|
| 3 words | | | |
| 5 words | | | |
| 10 words | | | |

This lets you investigate:

> **How does increasing contextual information affect each architecture?**

This is much more interesting academically.

### Division

**Person A:** runs GRU context experiments.

**Person B:** runs RNN context experiments.

**Person C:** runs LSTM context experiments.

Perfectly equal.

---

# Phase 12 — Perform error analysis

### Day 14

Each person finds approximately:

```text
10 correct predictions
10 incorrect predictions
5 interesting predictions
```

Then classify failures.

For example:

```text
Input:
"can you bring me a glass of"

Actual:
water

GRU:
milk

RNN:
juice

LSTM:
water
```

Discuss **why**.

Maybe another:

```text
Input:
"I will meet you at the"

Actual:
airport

LSTM:
station
```

That's not necessarily linguistically absurd.

This illustrates a major NLP issue:

> There can be several valid next words even when the dataset provides only one ground-truth target.

That is excellent material for your report and viva.

---

# Phase 13 — Build a real inference engine

### Day 15 — Person C

Create:

```python
predict_next_words(sentence, model, top_k=5)
```

Output:

```text
Sentence:
"I would like to"

LSTM predictions:

go       0.34
know     0.21
have     0.14
see      0.09
make     0.06
```

The function should support all models:

```text
GRU
RNN
LSTM
```

This becomes the foundation for your app.

---

# Phase 14 — Build an API

### Day 16 — Person C leads, A reviews

Use FastAPI.

Endpoint:

```text
POST /predict
```

Request:

```json
{
  "sentence": "I would like to",
  "model": "lstm",
  "top_k": 5
}
```

Response:

```json
{
  "predictions": [
    {"word": "go", "probability": 0.34},
    {"word": "know", "probability": 0.21}
  ]
}
```

FastAPI's current deployment documentation explicitly supports containerized deployment with Docker, so this is a reasonable production-style architecture rather than adding Docker only for show.

---

# Phase 15 — Build your frontend

### Days 17–18

Use **Streamlit**.

Your UI should look roughly like:

```text
╔══════════════════════════════════════════╗
║             ContextNext                  ║
║ GRU vs RNN vs LSTM Next Word Predictor   ║
╚══════════════════════════════════════════╝


Enter sentence:

[ I am planning to go to                         ]


Context length:  [5]

                 PREDICT


           GRU         RNN         LSTM
           ───         ───         ────
1. school  28%     market 41%      market 62%
2. office  19%     school 26%      school 21%
3. market  15%     office 13%      office  8%
```

Then include another tab:

### Model Comparison

Display:

```text
Accuracy
Perplexity
RMSE
Training time
Inference latency
```

with graphs.

Third tab:

### How it works

Show:

```text
Sentence
   ↓
Tokenization
   ↓
Word IDs
   ↓
GRU / RNN / LSTM
   ↓
Softmax
   ↓
Top-K words
```

This will make the project very easy to demonstrate.

---

# Phase 16 — Dockerize it

### Day 19

Person A and C together.

Add:

```text
Dockerfile
```

and ideally:

```text
docker-compose.yml
```

for:

```text
Frontend
+
API
```

Being able to run:

```bash
docker compose up
```

and launch the entire project is a strong engineering improvement.

---

# Phase 17 — Deploy it publicly

Use either:

**Streamlit Community Cloud**

or a container-based platform.

Streamlit Community Cloud can deploy directly from a GitHub repository.

Your GitHub README should then contain:

```text
🚀 Live Demo
```

with your deployed app.

That's important for placements because the recruiter can actually **use the project** instead of only reading about it.

---

# Phase 18 — Make the GitHub repository excellent

### Day 20 — Person A leads

Your README should contain:

```text
Project title

Live demo

Demo GIF

Problem statement

Why GRU vs RNN vs LSTM?

Dataset

Architecture

Preprocessing

Experiment design

Results

Graphs

Example predictions

Error analysis

Installation

Docker usage

API usage

Project structure

Team contributions

Limitations

Future work
```

Put the most impressive things near the top:

```text
Live Demo
GitHub
Results table
Architecture diagram
Demo GIF
```

not 30 paragraphs of theory.

---

# Phase 19 — Write the academic report

Person B leads this.

Divide chapters:

### Person A

```text
Dataset
EDA
Preprocessing
GRU architecture
```

### Person B

```text
RNN
Experimental methodology
Metrics
Results
Statistical analysis
```

### Person C

```text
LSTM
System architecture
Web application
Deployment
Future work
```

Shared:

```text
Abstract
Introduction
Conclusion
References
```

Then everyone reviews the entire document.

---

# Phase 20 — Prepare for viva/interviews

Each team member should be able to answer these without notes:

1. Why next-word prediction?
2. Why DailyDialog?
3. Why GRU?
4. Why does an RNN understand sequence order?
5. What problem does LSTM solve?
6. What are vanishing gradients?
7. How does GRU use gates?
8. What are LSTM's input, forget and output gates?
9. Why softmax?
10. Why cross-entropy?
11. Why isn't RMSE sufficient?
12. What is perplexity?
13. Why Top-3 accuracy?
14. Why fixed context length?
15. How did you prevent data leakage?
16. What happens to unknown words?
17. Why shouldn't stopwords be removed?
18. How does your inference API work?
19. How does Docker help?
20. Why did LSTM outperform/not outperform RNN or GRU?
21. What would happen with Transformers?
22. How would you scale the system?
23. What are your project's limitations?

If all three people can answer those, the project becomes genuinely useful for interviews.

---

# Equal-work matrix

This is what I would actually put in your internal project tracker.

| Work | A | B | C |
|---|---|---|---|
| GitHub/project setup | **Lead** | Review | Review |
| Dataset acquisition | **Lead** | Assist | Assist |
| EDA | **Lead** | Assist | — |
| Preprocessing | **Lead** | Review | Review |
| GRU | **Lead** | Review | Review |
| RNN | Review | **Lead** | Review |
| LSTM | Review | Review | **Lead** |
| Baseline | Assist | **Lead** | — |
| Metrics | Review | **Lead** | Review |
| MLflow | Assist | **Lead** | Assist |
| Context experiments | GRU | RNN | LSTM |
| Error analysis | GRU | RNN | LSTM |
| API | Review | Assist | **Lead** |
| Streamlit | Assist | Review | **Lead** |
| Docker | **Lead** | Review | **Lead** |
| Testing | **Lead** | Assist | Assist |
| Results/report | Dataset/GRU | **Results/RNN** | App/LSTM |
| README | **Lead** | Assist | Assist |
| Demo video | Assist | Assist | **Lead** |
| Final presentation | ⅓ | ⅓ | ⅓ |

That's fairly balanced.

---

# The version I would put on a résumé

Don't write:

> **Next Word Prediction Using LSTM**  
> Made GRU, RNN and LSTM models and compared them.

Write something closer to:

> **ContextNext — Neural Next-Word Prediction System**  
> Built and benchmarked GRU, Vanilla RNN and LSTM language models on conversational text using a reproducible NLP pipeline; evaluated Top-1/Top-3 accuracy, perplexity and error metrics across varying context lengths. Developed an interactive inference application with FastAPI and Streamlit, tracked experiments using MLflow, and containerized the system with Docker.

Once you have your real results, improve it further:

> LSTM improved Top-3 accuracy by **X%** over GRU while maintaining **Y ms** inference latency.

Numbers make résumé bullets considerably stronger.

---

## What turns this from a normal mini-project into a placement project

Don't add Transformers, attention, BERT and ten other models just to increase complexity.

The high-value additions are:

**reproducible experiments → meaningful metrics → baseline → error analysis → API → interactive UI → Docker → live deployment → proper GitHub documentation.**

That's a much stronger story than simply training more neural networks.

Your final system should be:

```text
                   DailyDialog
                       │
                       ▼
              Preprocessing Pipeline
                       │
                 Tokenization
                       │
              Sliding Word Windows
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
          GRU         RNN         LSTM
           │           │           │
           └───────────┼───────────┘
                       ▼
                Evaluation Engine
        Accuracy / Top-K / Perplexity
          MSE / MAE / RMSE / Time
                       │
                       ▼
                 Saved Models
                       │
                       ▼
                   FastAPI
                       │
                       ▼
                  Streamlit
                       │
                       ▼
                    Docker
                       │
                       ▼
                  Live Demo
```
