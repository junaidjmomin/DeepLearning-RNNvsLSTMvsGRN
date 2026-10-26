"""Comparative analysis dashboard and prediction demo for ContextNext."""

from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("CONTEXTNEXT_API_URL", "http://127.0.0.1:8000").rstrip("/")
RESULTS_DIR = Path(os.getenv("CONTEXTNEXT_RESULTS_DIR", "results"))
METRICS = {
    "top_1_accuracy": ("Top-1 accuracy", "higher"),
    "top_3_accuracy": ("Top-3 accuracy", "higher"),
    "cross_entropy": ("Cross-entropy", "lower"),
    "perplexity": ("Perplexity", "lower"),
    "rmse": ("RMSE", "lower"),
    "training_seconds": ("Training time (s)", "lower"),
    "inference_latency_ms": ("Inference latency (ms)", "lower"),
    "parameter_count": ("Parameters", "lower"),
}
MODEL_LABELS = {"gru": "GRU", "rnn": "Vanilla RNN", "vanilla_rnn": "Vanilla RNN", "lstm": "LSTM"}
MODEL_COLORS = {"GRU": "#cc765f", "Vanilla RNN": "#7185c5", "LSTM": "#2f9587"}


def pretty_model(name: str) -> str:
    return MODEL_LABELS.get(name.lower(), name.replace("_", " ").title())


def read_experiments() -> list[dict]:
    """Load metric artifacts, keeping context length and settings with each run."""

    experiments = []
    if not RESULTS_DIR.exists():
        return experiments
    for path in sorted(RESULTS_DIR.glob("*_test_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        raw_model = str(payload.get("model") or path.name.removesuffix("_test_metrics.json"))
        choices = payload.get("choices") if isinstance(payload.get("choices"), dict) else {}
        context = payload.get("sequence_length", choices.get("context_words"))
        experiments.append(
            {
                **payload,
                "_model": pretty_model(raw_model),
                "_raw_model": raw_model,
                "_context": int(context) if context is not None else None,
                "_run": path.stem,
                "_choices": choices,
            }
        )
    return experiments


def api_status() -> tuple[bool, dict]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        response.raise_for_status()
        return True, response.json()
    except (requests.RequestException, ValueError):
        return False, {}


def format_metric(key: str, value):
    if value is None:
        return "—"
    if key in {"top_1_accuracy", "top_3_accuracy"}:
        return f"{float(value):.1%}"
    if key in {"cross_entropy", "perplexity", "rmse"}:
        return f"{float(value):.4f}"
    if key == "training_seconds":
        return f"{float(value):.1f}"
    if key == "inference_latency_ms":
        return f"{float(value):.3f}"
    if key == "parameter_count":
        return f"{int(value):,}"
    return f"{float(value):.4g}"


st.set_page_config(page_title="ContextNext", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --ink:#172522; --muted:#667570; --paper:#f3f6f2; --line:#dfe7e0; --accent:#147b70; --accent-soft:#e4f1ec; }
      .stApp { background:radial-gradient(ellipse at 12% 0%, #e9f2ed 0, transparent 34rem),var(--paper); color:var(--ink); font-family:'Manrope',sans-serif; }
      [data-testid="stHeader"] { background:transparent; }
      .block-container { padding-top:1.65rem; padding-bottom:3rem; max-width:1180px; }
      h1,h2,h3 { letter-spacing:-.035em; color:var(--ink); }
      h2 { margin-top:.5rem; }
      .app-heading { padding:.7rem 1rem 1.15rem; border:1px solid #dce7df; border-radius:16px; background:rgba(255,255,255,.74); margin-bottom:1.15rem; box-shadow:0 8px 24px rgba(29,70,54,.045); }
      .app-heading:before { content:'DAILYDIALOG · LANGUAGE MODEL STUDY'; display:block; margin-bottom:.45rem; font:500 .65rem 'DM Mono',monospace; letter-spacing:.16em; color:var(--accent); }
      .app-heading h1 { font-size:2.45rem; line-height:1.05; margin:0; }
      .app-heading p { margin:.45rem 0 0; color:var(--muted); font-size:.98rem; }
      .section-kicker { font:500 .68rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.13em; color:var(--accent); margin:.1rem 0 .65rem; }
      .metric-note { color:var(--muted); font-size:.83rem; }
      .empty-state { background:linear-gradient(135deg,#fff,#f8fbf8); border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:14px; padding:1.4rem 1.5rem; box-shadow:0 8px 24px rgba(29,70,54,.035); }
      .empty-state code { color:#176e63; }
      div[data-testid="stMetric"] { height:100%; background:linear-gradient(145deg,#fff,#f9fbf9); border:1px solid var(--line); border-radius:14px; padding:1rem 1.05rem; box-shadow:0 5px 18px rgba(29,70,54,.035); }
      div[data-testid="stMetricLabel"] { color:var(--muted); font-size:.8rem; }
      div[data-testid="stMetricValue"] { color:var(--ink); font-family:'DM Mono',monospace; }
      div[data-testid="stTabs"] [role="tablist"] { gap:.35rem; border-bottom:1px solid var(--line); }
      div[data-testid="stTabs"] button[role="tab"],
      div[data-testid="stTabs"] button[role="tab"] * { color:#40534d !important; -webkit-text-fill-color:#40534d !important; font-weight:700; border-radius:9px 9px 0 0; opacity:1 !important; }
      div[data-testid="stTabs"] button[role="tab"] { padding:.65rem 1rem; }
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * { color:#126b61 !important; -webkit-text-fill-color:#126b61 !important; background-color:#e4f1ec; }
      div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--accent); height:3px; }
      div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#fff; }
      div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:rgba(255,255,255,.72); }
      div[data-testid="stAlert"] { border:1px solid #c9ddd4; border-radius:12px; background:#eaf4ef; }
      div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
      div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] span { color:#24463d !important; -webkit-text-fill-color:#24463d !important; }
      div[data-testid="stAlert"] svg { fill:#147b70 !important; color:#147b70 !important; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
      [data-testid="stMarkdownContainer"] p, label, label *,
      [data-testid="stWidgetLabel"] p { color:#40534d; }
      .stButton > button[kind="primary"] { border:0; border-radius:10px; background:var(--accent); font-weight:700; box-shadow:0 5px 12px rgba(20,123,112,.18); }
      .stButton > button[kind="primary"]:hover { background:#106b62; border:0; }
      .stTextInput input { border-radius:10px; background:#fff; }
      .stSelectbox [data-baseweb="select"] > div { border-radius:10px; }
      div[data-testid="stProgress"] > div { background:#e5eee8; border-radius:99px; }
      div[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#45a99a,#147b70); border-radius:99px; }
      .small-label { color:var(--muted); font: .72rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.08em; }
    </style>
    """,
    unsafe_allow_html=True,
)

experiments = read_experiments()
api_online, health = api_status()
available_models = [name for name, info in health.get("models", {}).items() if info.get("available")] if api_online else []

st.markdown(
    """
    <div class="app-heading">
      <h1>ContextNext</h1>
      <p>GRU vs RNN vs LSTM next-word prediction on DailyDialog</p>
    </div>
    """,
    unsafe_allow_html=True,
)

prediction_tab, overview_tab, method_tab = st.tabs(["Predict", "Model comparison", "How it works"])

with overview_tab:
    st.markdown('<div class="section-kicker">Model comparison · DailyDialog test results</div>', unsafe_allow_html=True)
    if not experiments:
        st.markdown(
            """
            <div class="empty-state">
              <strong>No experiment results are available yet.</strong><br><br>
              Run the model training commands to create metric files in <code>results/</code>. The dashboard will populate from those saved runs; it does not fill in placeholder scores.
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("What will be compared?"):
            st.write("Top-1 and Top-3 accuracy, cross-entropy, perplexity, RMSE, parameter count, training time, and inference latency.")
    else:
        contexts = sorted({item["_context"] for item in experiments if item["_context"] is not None})
        context_options = ["All contexts"] + [f"{value} words" for value in contexts]
        selected_context = st.selectbox("Context window", context_options, index=1 if contexts else 0)
        if selected_context == "All contexts":
            visible = experiments
            st.info("Runs with different context lengths are shown separately. For an architecture-only comparison, select one context length.")
        else:
            chosen = int(selected_context.split()[0])
            visible = [item for item in experiments if item["_context"] == chosen]

        # Headline metrics use the selected shared context. Direction is
        # explicit, so runtime and error measures are not mistaken for scores.
        valid_accuracy = [item for item in visible if item.get("top_1_accuracy") is not None]
        valid_perplexity = [item for item in visible if item.get("perplexity") is not None]
        valid_latency = [item for item in visible if item.get("inference_latency_ms") is not None]
        metric_cols = st.columns(4)
        if valid_accuracy:
            winner = max(valid_accuracy, key=lambda item: item["top_1_accuracy"])
            metric_cols[0].metric("Best Top-1", format_metric("top_1_accuracy", winner["top_1_accuracy"]), winner["_model"])
        else:
            metric_cols[0].metric("Best Top-1", "—")
        if valid_perplexity:
            winner = min(valid_perplexity, key=lambda item: item["perplexity"])
            metric_cols[1].metric("Lowest perplexity", format_metric("perplexity", winner["perplexity"]), winner["_model"])
        else:
            metric_cols[1].metric("Lowest perplexity", "—")
        if valid_latency:
            winner = min(valid_latency, key=lambda item: item["inference_latency_ms"])
            metric_cols[2].metric("Fastest inference", f"{float(winner['inference_latency_ms']):.3f} ms", winner["_model"])
        else:
            metric_cols[2].metric("Fastest inference", "—")
        metric_cols[3].metric("Compared runs", str(len(visible)), f"{len({item['_model'] for item in visible})} architectures")

        st.markdown("### Model scorecard")
        st.caption("Higher is better for accuracy. Lower is better for loss, perplexity, error, time, and parameter count.")
        table_rows = []
        for item in visible:
            label = item["_model"] if selected_context != "All contexts" else f"{item['_model']} · {item['_context'] or '?'} words"
            row = {"Model / run": label}
            for key, (title, _) in METRICS.items():
                if key in item:
                    row[title] = format_metric(key, item[key])
            row["Examples"] = item.get("samples", "—")
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

        available_metric_keys = [key for key in METRICS if any(item.get(key) is not None for item in visible)]
        if available_metric_keys:
            st.markdown("### Compare a metric")
            selected_metric = st.selectbox(
                "Metric",
                available_metric_keys,
                format_func=lambda key: f"{METRICS[key][0]} · {'higher' if METRICS[key][1] == 'higher' else 'lower'} is better",
            )
            chart_rows = []
            for item in visible:
                if item.get(selected_metric) is None:
                    continue
                label = item["_model"] if selected_context != "All contexts" else f"{item['_model']} · ctx {item['_context']}"
                chart_rows.append({"Model / run": label, METRICS[selected_metric][0]: float(item[selected_metric])})
            st.bar_chart(pd.DataFrame(chart_rows).set_index("Model / run"), color="#23877b", height=300)
            st.caption(f"{METRICS[selected_metric][1].capitalize()} values indicate a better result for this metric.")

        with st.expander("Experiment settings and comparability"):
            settings_rows = []
            for item in visible:
                choices = item.get("_choices", {})
                settings_rows.append(
                    {
                        "Model": item["_model"],
                        "Context": item.get("_context", "—"),
                        "Embedding": choices.get("embedding_dim", "—"),
                        "Hidden units": choices.get("rnn_units", choices.get("lstm_units", choices.get("gru_units", "—"))),
                        "Dropout": choices.get("dropout", "—"),
                        "Recurrent dropout": choices.get("recurrent_dropout", "—"),
                        "Learning rate": choices.get("learning_rate", "—"),
                        "Seed": choices.get("seed", "—"),
                    }
                )
            st.dataframe(pd.DataFrame(settings_rows), hide_index=True, use_container_width=True)
            st.caption("Check these settings before attributing a score difference to architecture. Missing values mean the run did not record that setting.")

with prediction_tab:
    st.markdown('<div class="section-kicker">Next-word prediction</div>', unsafe_allow_html=True)
    st.subheader("Try the models")
    st.write("Enter a sentence and inspect each model’s most likely next words.")
    if not api_online:
        st.warning("The prediction API is offline. Start it with `uvicorn api.main:app --reload` to use this panel.")
    elif not available_models:
        st.info("No model checkpoints are available. Train a model and place its checkpoint under `artifacts/models/`.")
    else:
        model_choice = st.selectbox("Architecture", available_models, format_func=pretty_model)
        sentence = st.text_input("Sentence context", placeholder="I would like to")
        top_k = st.slider("Suggestions", 1, 10, 5)
        if st.button("Predict next word", type="primary"):
            if not sentence.strip():
                st.error("Enter a sentence to get predictions.")
            else:
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"sentence": sentence, "model": model_choice, "top_k": top_k},
                        timeout=30,
                    )
                    response.raise_for_status()
                    predictions = response.json().get("predictions", [])
                    for rank, item in enumerate(predictions, start=1):
                        with st.container(border=True):
                            col_rank, col_word, col_bar, col_prob = st.columns([0.55, 1.5, 5, 1])
                            col_rank.caption(f"{rank:02d}")
                            col_word.markdown(f"**{item['word']}**")
                            col_bar.progress(min(max(float(item["probability"]), 0.0), 1.0))
                            col_prob.markdown(f"**{float(item['probability']):.1%}**")
                except requests.RequestException as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Prediction failed: {detail}")

with method_tab:
    st.markdown('<div class="section-kicker">Project overview</div>', unsafe_allow_html=True)
    st.subheader("How it works")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Task")
        st.write("Use a fixed window of recent words to predict the next token from the shared DailyDialog vocabulary.")
        st.markdown("#### Models")
        st.write("Vanilla RNN, GRU, and LSTM share an embedding and vocabulary-softmax prediction task. The recurrent layer is the main architectural difference.")
    with col_b:
        st.markdown("#### Fairness checks")
        st.write("Keep data splits, vocabulary, context length, seed, and training settings aligned. Use the context selector to avoid comparing different input windows as if they were the same experiment.")
        st.markdown("#### Metrics")
        st.write("Accuracy and Top-3 accuracy measure ranking quality. Cross-entropy and perplexity measure predictive uncertainty. Timing and parameter count show computational cost; MSE, MAE, and RMSE are included as additional probability-distribution errors.")
    st.info("DailyDialog provides one reference next word per example. A plausible alternative can be marked incorrect, so inspect error examples alongside aggregate scores.")
