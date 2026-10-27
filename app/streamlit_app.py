"""Comparative analysis dashboard and prediction demo for ContextNext."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("CONTEXTNEXT_API_URL", "http://127.0.0.1:8000").rstrip("/")
RESULTS_DIR = Path(os.getenv("CONTEXTNEXT_RESULTS_DIR", "results"))
METRICS = {
    "top_1_accuracy": ("Top-1 accuracy", "higher"),
    "top_3_accuracy": ("Top-3 accuracy", "higher"),
    "top_5_accuracy": ("Top-5 accuracy", "higher"),
    "cross_entropy": ("Cross-entropy", "lower"),
    "perplexity": ("Perplexity", "lower"),
    "rmse": ("RMSE", "lower"),
    "training_seconds": ("Training time (s)", "lower"),
    "inference_latency_ms": ("Inference latency (ms)", "lower"),
    "parameter_count": ("Parameters", "lower"),
}
MODEL_LABELS = {"gru": "GRU", "rnn": "Vanilla RNN", "vanilla_rnn": "Vanilla RNN", "lstm": "LSTM"}
MODEL_COLORS = {"GRU": "#c2705c", "Vanilla RNN": "#6d81c4", "LSTM": "#0f766e"}


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
    if key in {"top_1_accuracy", "top_3_accuracy", "top_5_accuracy"}:
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


def badge(text: str, tone: str = "") -> str:
    return f'<span class="ctx-badge {tone}">{html.escape(str(text))}</span>'


def render_model_card(result: dict) -> str:
    label = pretty_model(result["model"])
    color = MODEL_COLORS.get(label, "#0f766e")
    top = result["predictions"][0]
    truth = result.get("ground_truth") or {}
    note = badge("No scored target")
    if truth.get("available") and truth.get("representable"):
        rank = int(truth["rank"])
        if truth.get("correct"):
            note = badge("Exact target hit", "good")
        elif rank <= 5:
            note = badge(f"Target in top 5 · rank {rank}", "warm")
        else:
            note = badge(f"Target rank {rank:,}")
    return f"""
<div class="model-card" style="border-top-color:{color}">
  <div class="ctx-eyebrow">{html.escape(label)}</div>
  <div class="model-prediction">{html.escape(str(top["word"]))}</div>
  <div class="model-prob">{float(top["probability"]):.2%}<span>top-1 confidence</span></div>
  <div class="model-note">{note}</div>
</div>
"""


def render_token_path(result: dict) -> str:
    tokenization = result["tokenization"]
    chips = []
    for token, token_id, decoded in zip(
        tokenization["context_tokens"], tokenization["token_ids"], tokenization["decoded_tokens"]
    ):
        is_pad = str(token) == "<pad>"
        decoded_line = (
            f'<span class="decoded">{html.escape(str(decoded))}</span>'
            if str(decoded) != str(token)
            else ""
        )
        chips.append(
            f'<span class="ctx-token{" pad" if is_pad else ""}">'
            f'<span class="tok">{html.escape(str(token))}</span>'
            f'<span class="tid">ID {int(token_id)}</span>'
            f"{decoded_line}"
            "</span>"
        )
    return '<div class="token-row">' + "".join(chips) + "</div>"


def render_probability_panel(result: dict, max_probability: float) -> str:
    label = pretty_model(result["model"])
    color = MODEL_COLORS.get(label, "#0f766e")
    rows = []
    for item in result["predictions"]:
        probability = float(item["probability"])
        width = 1.5 if max_probability <= 0 else max(1.5, probability / max_probability * 100)
        rows.append(
            '<div class="prob-row">'
            f'<span class="prob-rank">{int(item["rank"]):02d}</span>'
            f'<span class="prob-word">{html.escape(str(item["word"]))}</span>'
            f'<span class="prob-track"><span class="prob-fill" style="width:{width:.1f}%;background:{color}"></span></span>'
            f'<span class="prob-pct">{probability:.2%}</span>'
            "</div>"
        )
    return (
        '<div class="prob-panel">'
        f'<div class="prob-head"><span class="prob-dot" style="background:{color}"></span>'
        f'{html.escape(label)}<span class="prob-top">top word · {html.escape(str(result["predictions"][0]["word"]))}</span></div>'
        + "".join(rows)
        + "</div>"
    )


def architecture_details(model_name: str) -> tuple[list[str], str]:
    if model_name == "rnn":
        return (
            ["Token embeddings", "SimpleRNN cell updates hidden state for each word", "Final hidden state", "Vocabulary softmax"],
            "hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)",
        )
    if model_name == "gru":
        return (
            ["Token embeddings", "GRU reset and update gates", "Final hidden state", "Vocabulary softmax"],
            "The reset and update gates control how the previous hidden state and current input form the next state.",
        )
    return (
        ["Token embeddings", "LSTM forget, input, and output gates", "Cell state and final hidden state", "Vocabulary softmax"],
        "The cell state carries information through the sequence; the gates control what it forgets, stores, and exposes.",
    )


st.set_page_config(page_title="ContextNext · Next-word simulation", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root {
        --ink:#13231f; --muted:#687771; --paper:#f4f7f3; --card:#ffffff;
        --line:#dde7df; --accent:#147b70; --accent-deep:#0f665d; --accent-soft:#e7f4ef;
      }
      .stApp {
        background:
          radial-gradient(ellipse at 8% -4%, #e3f0e9 0, transparent 32rem),
          radial-gradient(ellipse at 100% 0%, #eef3eb 0, transparent 28rem),
          var(--paper);
        color:var(--ink);
        font-family:'Manrope',sans-serif;
      }
      [data-testid="stHeader"] { background:transparent; }
      .block-container { padding-top:1.5rem; padding-bottom:3.5rem; max-width:1180px; }
      h1,h2,h3 { letter-spacing:-.035em; color:var(--ink); }
      h2 { margin-top:.5rem; }
      .ctx-eyebrow { font:500 .66rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.15em; color:var(--accent); }
      .ctx-hero {
        display:flex; align-items:flex-start; justify-content:space-between; gap:1.25rem;
        padding:1.35rem 1.5rem; border:1px solid #d9e6dd; border-radius:18px;
        background:rgba(255,255,255,.76); margin-bottom:1.15rem;
        box-shadow:0 14px 34px rgba(29,70,54,.055); backdrop-filter:blur(8px);
      }
      .ctx-hero h1 { font-size:2.5rem; line-height:1; margin:.42rem 0 0; }
      .ctx-hero p { margin:.55rem 0 0; color:var(--muted); font-size:.96rem; max-width:44rem; }
      .ctx-status {
        display:inline-flex; align-items:center; gap:.5rem; white-space:nowrap;
        border:1px solid var(--line); border-radius:999px; padding:.46rem .8rem;
        background:#fff; color:#4a5c55; font:500 .72rem 'DM Mono',monospace;
      }
      .ctx-status .dot { width:.48rem; height:.48rem; border-radius:50%; background:#9aa8a1; }
      .ctx-status.online { border-color:#c4e2d5; background:#edf8f2; color:#17695c; }
      .ctx-status.online .dot { background:#1e9b7e; box-shadow:0 0 0 3px rgba(30,155,126,.14); }
      .ctx-status.offline { border-color:#ecd8c7; background:#fdf4ed; color:#8a5a37; }
      .ctx-status.offline .dot { background:#cf8a52; }

      div[data-testid="stTabs"] [role="tablist"] { gap:.35rem; border-bottom:1px solid var(--line); }
      div[data-testid="stTabs"] button[role="tab"],
      div[data-testid="stTabs"] button[role="tab"] * {
        color:#42564f !important; -webkit-text-fill-color:#42564f !important;
        font-weight:700; border-radius:10px 10px 0 0; opacity:1 !important;
      }
      div[data-testid="stTabs"] button[role="tab"] { padding:.65rem 1rem; }
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {
        color:#126b61 !important; -webkit-text-fill-color:#126b61 !important; background-color:var(--accent-soft);
      }
      div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--accent); height:3px; }

      div[data-testid="stVerticalBlockBorderWrapper"] {
        background:rgba(255,255,255,.78); border-color:var(--line) !important; border-radius:16px;
      }
      div[data-testid="stMetric"] {
        height:100%; background:linear-gradient(145deg,#fff,#fafcfa); border:1px solid var(--line);
        border-radius:14px; padding:1rem 1.05rem; box-shadow:0 5px 18px rgba(29,70,54,.035);
      }
      div[data-testid="stMetricLabel"] { color:var(--muted); font-size:.8rem; }
      div[data-testid="stMetricValue"] { color:var(--ink); font-family:'DM Mono',monospace; }
      div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#fff; }
      div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:rgba(255,255,255,.72); }
      div[data-testid="stAlert"] { border:1px solid #c9ddd4; border-radius:12px; background:#eaf4ef; }
      div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
      div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] span { color:#24463d !important; -webkit-text-fill-color:#24463d !important; }
      div[data-testid="stAlert"] svg { fill:#147b70 !important; color:#147b70 !important; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
      [data-testid="stMarkdownContainer"] p, label, label *,
      [data-testid="stWidgetLabel"] p { color:#40534d; }
      .stButton > button[kind="primary"] {
        border:0; border-radius:10px; background:var(--accent); font-weight:700;
        box-shadow:0 5px 12px rgba(20,123,112,.18);
      }
      .stButton > button[kind="primary"]:hover { background:var(--accent-deep); border:0; }
      .stTextInput input, .stTextArea textarea { border-radius:10px; background:#fff; }
      .stSelectbox [data-baseweb="select"] > div { border-radius:10px; }
      div[data-testid="stProgress"] > div { background:#e5eee8; border-radius:99px; }
      div[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#45a99a,#147b70); border-radius:99px; }

      .sim-heading { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin:.2rem 0 .9rem; }
      .sim-heading h2 { margin:0; font-size:1.65rem; }
      .sim-heading p { margin:.3rem 0 0; color:var(--muted); font-size:.92rem; }
      .step-head { display:flex; align-items:center; gap:.6rem; margin:1.35rem 0 .65rem; }
      .step-num {
        display:inline-flex; align-items:center; justify-content:center; width:1.65rem; height:1.65rem;
        border-radius:8px; background:var(--accent-soft); color:var(--accent-deep);
        font:500 .7rem 'DM Mono',monospace;
      }
      .step-title { font-size:1.02rem; font-weight:800; letter-spacing:-.02em; }
      .step-note { color:var(--muted); font-size:.82rem; margin-left:auto; }

      .input-card, .verdict-card, .model-card, .prob-panel, .offline-card, .empty-state {
        background:var(--card); border:1px solid var(--line); border-radius:14px;
        box-shadow:0 6px 20px rgba(29,70,54,.04);
      }
      .input-card { padding:1rem 1.15rem; }
      .input-quote { font-size:1.18rem; font-weight:700; letter-spacing:-.02em; margin:.35rem 0 .55rem; word-break:break-word; }
      .input-meta { color:var(--muted); font: .74rem 'DM Mono',monospace; }
      .token-row { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.7rem; }
      .ctx-token {
        display:inline-flex; flex-direction:column; align-items:center; gap:.1rem; min-width:4.6rem;
        padding:.48rem .65rem; border:1px solid #cfe4da; border-radius:10px; background:#f4fbf7;
      }
      .ctx-token .tok { font-weight:800; font-size:.88rem; color:#174c42; }
      .ctx-token .tid { font: .68rem 'DM Mono',monospace; color:#5b756c; }
      .ctx-token .decoded { font: .63rem 'DM Mono',monospace; color:#93a49d; }
      .ctx-token.pad { background:#f5f6f5; border-color:#e0e5e1; }
      .ctx-token.pad .tok { color:#98a49f; }

      .model-card { padding:.95rem 1rem 1rem; border-top:3px solid var(--accent); height:100%; }
      .model-prediction { font-size:1.62rem; font-weight:800; letter-spacing:-.04em; margin:.28rem 0 0; word-break:break-word; }
      .model-prob { font:500 1.18rem 'DM Mono',monospace; color:#23483f; margin-top:.15rem; }
      .model-prob span { display:block; font:500 .66rem 'DM Mono',monospace; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-top:.18rem; }
      .model-note { margin-top:.7rem; }
      .ctx-badge {
        display:inline-block; padding:.22rem .5rem; border-radius:999px; border:1px solid #e0e7e2;
        background:#f7f9f7; color:#6b7973; font:500 .66rem 'DM Mono',monospace;
      }
      .ctx-badge.good { background:#e7f7ef; border-color:#bfe7d2; color:#167051; }
      .ctx-badge.warm { background:#fdf5e6; border-color:#eeddbb; color:#8a6424; }

      .prob-panel { padding:.85rem .95rem 1rem; height:100%; }
      .prob-head { display:flex; align-items:center; gap:.45rem; font-weight:800; font-size:.9rem; margin-bottom:.72rem; }
      .prob-dot { width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }
      .prob-top { margin-left:auto; color:var(--muted); font:500 .66rem 'DM Mono',monospace; }
      .prob-row { display:flex; align-items:center; gap:.55rem; padding:.24rem 0; }
      .prob-rank { width:1.35rem; color:#95a49e; font: .68rem 'DM Mono',monospace; }
      .prob-word { width:4.4rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:700; font-size:.85rem; }
      .prob-track { flex:1; height:.45rem; border-radius:99px; background:#edf2ee; overflow:hidden; }
      .prob-fill { display:block; height:100%; border-radius:99px; }
      .prob-pct { width:3.4rem; text-align:right; font: .72rem 'DM Mono',monospace; color:#40534d; }

      .verdict-card { padding:1rem 1.15rem; display:flex; align-items:stretch; gap:1.4rem; flex-wrap:wrap; }
      .verdict-word { min-width:11rem; padding-right:1.4rem; border-right:1px solid var(--line); }
      .verdict-target { font-size:1.75rem; font-weight:800; letter-spacing:-.04em; margin:.2rem 0 .25rem; }
      .verdict-source { color:var(--muted); font: .7rem 'DM Mono',monospace; }
      .verdict-list { display:flex; flex-direction:column; gap:.5rem; justify-content:center; flex:1; min-width:16rem; }
      .verdict-item { display:flex; align-items:center; gap:.55rem; flex-wrap:wrap; font-size:.86rem; }
      .verdict-item .name { font-weight:800; min-width:6.2rem; }
      .verdict-item .score { color:var(--muted); font: .72rem 'DM Mono',monospace; }

      .offline-card, .empty-state {
        padding:1.35rem 1.45rem; border-left:4px solid var(--accent);
        background:linear-gradient(135deg,#fff,#f8fbf8);
      }
      .offline-card h3, .empty-state h3 { margin:0 0 .35rem; font-size:1.05rem; }
      .offline-card p, .empty-state p { margin:0 0 .8rem; color:var(--muted); font-size:.92rem; }
      .offline-card code, .empty-state code { color:#176e63; background:#eaf5f0; border-radius:5px; padding:.1rem .3rem; }
      .section-spacer { height:.35rem; }
      @media (max-width: 760px) {
        .ctx-hero { flex-direction:column; }
        .ctx-hero h1 { font-size:2.05rem; }
        .verdict-word { border-right:0; border-bottom:1px solid var(--line); padding:0 0 .85rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

experiments = read_experiments()
api_online, health = api_status()
available_models = [name for name, info in health.get("models", {}).items() if info.get("available")] if api_online else []
status_text = f"API online · {len(available_models)} model{'s' if len(available_models) != 1 else ''}" if api_online else "API offline"
status_class = "online" if api_online else "offline"

st.markdown(
    f"""
<div class="ctx-hero">
  <div>
    <div class="ctx-eyebrow">DAILYDIALOG · LANGUAGE MODEL STUDY</div>
    <h1>ContextNext</h1>
    <p>See how a vanilla RNN, GRU, and LSTM turn the same three-word context into a next-word prediction.</p>
  </div>
  <div class="ctx-status {status_class}"><span class="dot"></span>{html.escape(status_text)}</div>
</div>
""",
    unsafe_allow_html=True,
)

prediction_tab, overview_tab, method_tab = st.tabs(["Simulate", "Compare models", "How it works"])

with overview_tab:
    st.markdown('<div class="ctx-eyebrow">Model comparison · DailyDialog test results</div>', unsafe_allow_html=True)
    if not experiments:
        st.markdown(
            """
            <div class="empty-state">
              <h3>No experiment results are available yet.</h3>
              <p>Run the model training commands to create metric files in <code>results/</code>. The dashboard will populate from those saved runs; it does not fill in placeholder scores.</p>
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
    st.markdown(
        """
<div class="sim-heading">
  <div>
    <div class="ctx-eyebrow">Interactive model experiment</div>
    <h2>Next-word simulation</h2>
    <p>Pick an architecture, give it a context, and compare the real probability distribution with the stored target.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if not api_online:
        st.markdown(
            f"""
<div class="offline-card">
  <h3>The prediction API is offline</h3>
  <p>Start the API in a terminal, then refresh this page to run a live simulation.</p>
  <code>uvicorn api.main:app --reload</code>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption(f"Expected API URL: {API_URL}")
    elif not available_models:
        st.markdown(
            """
<div class="offline-card">
  <h3>No usable model checkpoints were found</h3>
  <p>Check the checkpoint paths reported by the API health endpoint, or train a model first.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        with st.container(border=True):
            model_options = (["all"] if len(available_models) > 1 else []) + available_models
            control_left, control_right = st.columns([1, 1.25])
            with control_left:
                model_choice = st.selectbox(
                    "Architecture",
                    model_options,
                    format_func=lambda name: "Compare all models" if name == "all" else pretty_model(name),
                )
            with control_right:
                experiment_mode = st.radio(
                    "Input source",
                    ["Custom input", "Dataset example"],
                    horizontal=True,
                )

            example_id = None
            selected_example = None
            if experiment_mode == "Dataset example":
                split_col, search_col = st.columns([1, 2.1])
                with split_col:
                    split_choice = st.selectbox("Dataset split", ["train", "validation", "test", "all"], index=0)
                with search_col:
                    search_query = st.text_input("Find a labelled context", placeholder="vegetarian supreme and")
                try:
                    example_response = requests.get(
                        f"{API_URL}/examples",
                        params={"split": split_choice, "q": search_query, "limit": 100},
                        timeout=30,
                    )
                    example_response.raise_for_status()
                    dataset_examples = example_response.json().get("examples", [])
                except requests.RequestException as exc:
                    dataset_examples = []
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Could not load labelled examples: {detail}")

                if dataset_examples:
                    selected_example = st.selectbox(
                        "Select the exact labelled record",
                        dataset_examples,
                        format_func=lambda item: f"{item['input']}  →  {item['target']}  ·  {item['split']} ({item['id']})",
                    )
                    sentence = selected_example["input"]
                    example_id = selected_example["id"]
                    st.caption(f"This record's stored target is **{selected_example['target']}** in the **{selected_example['split']}** split.")
                    if selected_example["split"] != "test":
                        st.caption("This selected row is not a held-out test example; use it to inspect model behavior, not to claim test accuracy.")
                else:
                    sentence = ""
                    st.info("No labelled contexts match that search. Try another term or switch to a different split.")
            else:
                sentence = st.text_area(
                    "Input text",
                    value="vegetarian supreme and",
                    height=92,
                    help="The model uses the last three words for the default checkpoints.",
                )

            top_col, button_col = st.columns([3, 1])
            with top_col:
                top_k = st.slider("Show top predictions", 1, 20, 5)
            with button_col:
                st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
                run_clicked = st.button("Run simulation", type="primary", use_container_width=True)

        run_models = available_models if model_choice == "all" else [model_choice]
        signature = (experiment_mode, sentence.strip(), example_id, model_choice, top_k)

        if run_clicked:
            if not sentence.strip():
                st.error("Choose or enter an input before running the simulation.")
            else:
                try:
                    model_results = []
                    for model_name in run_models:
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={
                                "sentence": sentence,
                                "model": model_name,
                                "top_k": top_k,
                                "example_id": example_id,
                            },
                            timeout=60,
                        )
                        response.raise_for_status()
                        model_results.append(response.json())
                    st.session_state["contextnext_simulation"] = {"signature": signature, "results": model_results}
                except requests.RequestException as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Simulation failed: {detail}")

        simulation = st.session_state.get("contextnext_simulation")
        if not simulation or simulation["signature"] != signature:
            st.caption("Run the simulation to see the model's real output and dataset comparison.")
        else:
            model_results = simulation["results"]
            first_result = model_results[0]
            max_probability = max(float(item["probability"]) for result in model_results for item in result["predictions"])
            model_list = " · ".join(pretty_model(result["model"]) for result in model_results)

            st.markdown(
                f"""
<div class="step-head">
  <span class="step-num">01</span>
  <span class="step-title">Input &amp; tokenization</span>
  <span class="step-note">{html.escape(model_list)}</span>
</div>
<div class="input-card">
  <div class="ctx-eyebrow">Context sent to the model</div>
  <div class="input-quote">{html.escape(sentence)}</div>
  <div class="input-meta">Model context · {first_result["sequence_length"]} words · vocabulary {first_result["vocabulary_size"]:,} IDs</div>
  {render_token_path(first_result)}
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown(
                """
<div class="step-head">
  <span class="step-num">02</span>
  <span class="step-title">Top-1 predictions</span>
  <span class="step-note">higher confidence is better</span>
</div>
""",
                unsafe_allow_html=True,
            )
            card_columns = st.columns(len(model_results))
            for column, result in zip(card_columns, model_results):
                with column:
                    st.markdown(render_model_card(result), unsafe_allow_html=True)

            st.markdown(
                """
<div class="step-head">
  <span class="step-num">03</span>
  <span class="step-title">Probability distribution</span>
  <span class="step-note">bars are scaled to the highest shown probability</span>
</div>
""",
                unsafe_allow_html=True,
            )
            distribution_columns = st.columns(len(model_results))
            for column, result in zip(distribution_columns, model_results):
                with column:
                    st.markdown(render_probability_panel(result, max_probability), unsafe_allow_html=True)

            ground_truth = first_result.get("ground_truth", {"available": False, "reason": "No label was returned."})
            st.markdown(
                """
<div class="step-head">
  <span class="step-num">04</span>
  <span class="step-title">Target check</span>
  <span class="step-note">DailyDialog supplies one reference word per row</span>
</div>
""",
                unsafe_allow_html=True,
            )
            if ground_truth["available"]:
                source = ground_truth.get("source", {})
                if "split" in source:
                    source_text = f"{source['split']} split · {source['id']}"
                elif "matches" in source:
                    source_text = f"{source['matches']} labelled record(s)"
                else:
                    source_text = "custom input"

                if ground_truth["representable"]:
                    verdict_items = []
                    for result in model_results:
                        truth = result["ground_truth"]
                        label = pretty_model(result["model"])
                        color = MODEL_COLORS.get(label, "#0f766e")
                        rank = int(truth["rank"])
                        if truth["correct"]:
                            mark = badge("exact", "good")
                        elif rank <= 5:
                            mark = badge("top 5", "warm")
                        else:
                            mark = badge("missed")
                        verdict_items.append(
                            '<div class="verdict-item">'
                            f'<span class="prob-dot" style="background:{color}"></span>'
                            f'<span class="name">{html.escape(label)}</span>'
                            f'<span class="score">{float(truth["probability"]):.3%} · rank {rank:,}</span>'
                            f"{mark}"
                            "</div>"
                        )
                    st.markdown(
                        f"""
<div class="verdict-card">
  <div class="verdict-word">
    <div class="ctx-eyebrow">Stored next word</div>
    <div class="verdict-target">{html.escape(str(ground_truth["target"]))}</div>
    <div class="verdict-source">{html.escape(source_text)}</div>
  </div>
  <div class="verdict-list">{"".join(verdict_items)}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    with st.expander("Exact scores"):
                        summary_rows = []
                        for result in model_results:
                            truth = result["ground_truth"]
                            prediction = result["predictions"][0]
                            summary_rows.append(
                                {
                                    "Model": pretty_model(result["model"]),
                                    "Top prediction": prediction["word"],
                                    "P(top)": f"{float(prediction['probability']):.3%}",
                                    "Actual": truth["target"],
                                    "P(actual)": f"{float(truth['probability']):.3%}",
                                    "Actual rank": f"{int(truth['rank']):,} / {int(result['vocabulary_size']):,}",
                                    "Top-5 hit": "Yes" if int(truth["rank"]) <= 5 else "No",
                                }
                            )
                        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
                        st.caption("A plausible alternative word can still be marked incorrect because DailyDialog stores only one reference next word.")
                else:
                    st.warning(
                        f"The target word is outside the model vocabulary. Its probability and rank below refer to the `<unk>` bucket "
                        f"({float(ground_truth['probability']):.3%}, rank {int(ground_truth['rank']):,}), not to the exact word."
                    )
            else:
                st.info(f"Ground truth unavailable: {ground_truth['reason']}")
                for candidate in ground_truth.get("candidates", []):
                    sources = ", ".join(f"{item['split']} · {item['id']}" for item in candidate["examples"])
                    st.caption(f"Stored target **{candidate['target']}**: {sources}")
                st.caption("Model predictions are still shown above. Select one exact dataset row to score an ambiguous context.")

            with st.expander("Architecture details"):
                architecture_columns = st.columns(len(model_results))
                for column, result in zip(architecture_columns, model_results):
                    steps, explanation = architecture_details(result["model"])
                    with column:
                        st.markdown(f"**{pretty_model(result['model'])}**")
                        st.markdown(" → ".join(steps))
                        st.caption(explanation)
                        st.caption(f"Checkpoint: `{result['checkpoint']}`")


with method_tab:
    st.markdown('<div class="ctx-eyebrow">Project overview</div>', unsafe_allow_html=True)
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
