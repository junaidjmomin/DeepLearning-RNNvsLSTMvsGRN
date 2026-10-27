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
    "top_5_accuracy": ("Top-5 accuracy", "higher"),
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
        config_path = path.with_name(path.name.replace("_test_metrics.json", "_config.json"))
        try:
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        except (OSError, ValueError):
            config = {}
        history_path = path.with_name(path.name.replace("_test_metrics.json", "_history.json"))
        try:
            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else {}
        except (OSError, ValueError):
            history = {}
        context = payload.get("sequence_length", choices.get("context_words"))
        experiments.append(
            {
                **payload,
                "_model": pretty_model(raw_model),
                "_raw_model": raw_model,
                "_context": int(context) if context is not None else None,
                "_run": path.stem,
                "_choices": choices,
                "_config": config,
                "_history": history,
            }
        )
    return experiments


def api_status(context_size: int = 3) -> tuple[bool, dict]:
    try:
        response = requests.get(f"{API_URL}/health", params={"context_size": context_size}, timeout=30)
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
        contexts = sorted({3, 5, 10} | {item["_context"] for item in experiments if item["_context"] is not None})
        context_options = ["All contexts"] + [f"{value} words" for value in contexts]
        selected_context = st.selectbox("Context window", context_options, index=1 if contexts else 0)
        if selected_context == "All contexts":
            visible = experiments
            st.info("Runs with different context lengths are shown separately. For an architecture-only comparison, select one context length.")
        else:
            chosen = int(selected_context.split()[0])
            visible = [item for item in experiments if item["_context"] == chosen]
            if not visible:
                st.info(f"No saved results for context length {chosen} yet. The full-data Colab notebook writes these after training.")

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
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

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
                        "Train rows used": item.get("_config", {}).get("max_train_samples") or "All",
                        "Test rows used": item.get("samples", "—"),
                        "Epochs completed": item.get("epochs_completed", "—"),
                        "Best checkpoint by": item.get("checkpoint_monitor", choices.get("checkpoint_monitor", "—")),
                        "Embedding": choices.get("embedding_dim", "—"),
                        "Hidden units": choices.get("rnn_units", choices.get("lstm_units", choices.get("gru_units", "—"))),
                        "Dropout": choices.get("dropout", "—"),
                        "Recurrent dropout": choices.get("recurrent_dropout", "—"),
                        "Learning rate": choices.get("learning_rate", "—"),
                        "Seed": choices.get("seed", "—"),
                    }
                )
            st.dataframe(pd.DataFrame(settings_rows), hide_index=True, width="stretch")
            st.caption("A row cap means this is a diagnostic run. Compare architectures only when their data limits, context lengths, and settings match.")

        with st.expander("Training and validation loss by epoch"):
            history_rows = []
            for item in visible:
                history = item.get("_history", {})
                for index, (training_loss, validation_loss) in enumerate(zip(history.get("loss", []), history.get("val_loss", [])), start=1):
                    history_rows.append({"Epoch": index, "Run": f"{item['_model']} · {item['_context']} words", "Training loss": float(training_loss), "Validation loss": float(validation_loss)})
            if history_rows:
                frame = pd.DataFrame(history_rows)
                for run, run_frame in frame.groupby("Run"):
                    st.markdown(f"**{run}**")
                    st.line_chart(run_frame.set_index("Epoch")[["Training loss", "Validation loss"]])
            else:
                st.caption("No saved training history is available for these runs.")

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


with prediction_tab:
    st.markdown('<div class="section-kicker">Interactive model experiment</div>', unsafe_allow_html=True)
    st.subheader("Run a next-word simulation")
    st.write("Choose an architecture and input, then inspect the actual token IDs, model scores, and labelled target.")
    selected_context_length = st.selectbox("Context length", health.get("available_context_lengths", [3, 5, 10]))
    context_online, context_health = api_status(selected_context_length) if api_online else (False, {})
    available_models = [name for name, info in context_health.get("models", {}).items() if info.get("available")]
    if not api_online:
        st.warning("The prediction API is offline. Start it with `uvicorn api.main:app --reload` to run an experiment.")
    elif not context_online:
        st.warning("The API did not respond for this context length.")
    elif not available_models:
        st.info(f"No usable context-{selected_context_length} checkpoints are loaded. Train the three models in Colab, copy their artifacts here, then restart the API.")
    else:
        model_options = (["all"] if len(available_models) > 1 else []) + available_models
        model_choice = st.selectbox(
            "1 · Choose model",
            model_options,
            format_func=lambda name: "Compare all available models" if name == "all" else pretty_model(name),
        )
        experiment_mode = st.radio(
            "2 · Choose experiment",
            ["Custom input", "Dataset example"],
            horizontal=True,
        )

        example_id = None
        selected_example = None
        if experiment_mode == "Dataset example":
            split_choice = st.selectbox("Dataset split", ["train", "validation", "test", "all"], index=0)
            search_query = st.text_input("Find a labelled context", placeholder="vegetarian supreme and")
            try:
                example_response = requests.get(
                    f"{API_URL}/examples",
                    params={"split": split_choice, "q": search_query, "limit": 100, "context_size": selected_context_length},
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
                height=90,
                help=f"This checkpoint uses the last {selected_context_length} words.",
            )
            context = " ".join(sentence.lower().split()[-selected_context_length:])
            if len(context.split()) == selected_context_length:
                try:
                    match_response = requests.get(
                        f"{API_URL}/examples",
                        params={"split": "all", "q": context, "limit": 100, "context_size": selected_context_length},
                        timeout=30,
                    )
                    match_response.raise_for_status()
                    match_payload = match_response.json()
                    exact_rows = [row for row in match_payload.get("examples", []) if row["input"] == context]
                    if exact_rows:
                        split_counts = {split: sum(row["split"] == split for row in exact_rows) for split in ("train", "validation", "test")}
                        st.caption(
                            f"Exact context found in the dataset: {split_counts['train']} train, "
                            f"{split_counts['validation']} validation, {split_counts['test']} test row(s)."
                        )
                        if match_payload.get("has_more"):
                            st.caption(f"Showing the first {len(exact_rows)} of {match_payload['total']} matching rows; choose one exact row for a reliable comparison.")
                        if len({row["target"] for row in exact_rows}) > 1 or match_payload.get("has_more"):
                            selected_id = st.selectbox(
                                "Choose the actual next word for this dataset row",
                                [None] + exact_rows,
                                format_func=lambda row: "Show all possible targets" if row is None else f"{row['target']} · {row['split']} ({row['id']})",
                            )
                            example_id = selected_id["id"] if selected_id else None
                        else:
                            st.caption(f"Stored next word: **{exact_rows[0]['target']}**")
                except requests.RequestException as exc:
                    st.caption(f"Dataset lookup unavailable: {exc}")

        top_k = st.slider("Show top predictions", 1, 20, 5)
        run_clicked = st.button("Run simulation", type="primary")
        run_models = available_models if model_choice == "all" else [model_choice]
        signature = (selected_context_length, experiment_mode, sentence.strip(), example_id, model_choice, top_k)

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
                                "include_trace": True,
                                "context_size": selected_context_length,
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
            st.markdown("### 1 · What did I give the model?")
            st.code(sentence, language=None)

            st.markdown("### 2 · How was it tokenized?")
            tokenization = first_result["tokenization"]
            token_rows = [
                {"Position": i + 1, "Token": token, "Token ID": token_id, "Decoded ID": decoded}
                for i, (token, token_id, decoded) in enumerate(
                    zip(tokenization["context_tokens"], tokenization["token_ids"], tokenization["decoded_tokens"])
                )
            ]
            st.dataframe(pd.DataFrame(token_rows), hide_index=True, width="stretch")
            st.caption(
                f"Model context: {first_result['sequence_length']} words · vocabulary: {first_result['vocabulary_size']:,} IDs · "
                f"checkpoint: `{first_result['checkpoint']}`"
            )

            st.markdown("### 3 · How does the model process it?")
            trace_tabs = st.tabs([pretty_model(result["model"]) for result in model_results])
            for tab, result in zip(trace_tabs, model_results):
                with tab:
                    steps, explanation = architecture_details(result["model"])
                    st.markdown(" → ".join(steps))
                    st.caption(explanation)
                    trace = result.get("trace", {"available": False, "reason": "Trace was not requested."})
                    if trace["available"]:
                        st.caption(f"Actual checkpoint values · embedding width {trace['embedding_dimension']} · hidden units {trace['hidden_units']}")
                        st.dataframe(pd.DataFrame([
                            {
                                "Step": step["position"],
                                "Token": step["token"],
                                "ID": step["token_id"],
                                "Used": "Yes" if step["active"] else "Padding",
                                "Embedding norm": round(step["embedding_norm"], 5),
                                "First 6 embedding values": ", ".join(f"{value:.3f}" for value in step["embedding_preview"]),
                                "Hidden state norm": round(step["hidden_norm"], 5),
                                "First 6 hidden values": ", ".join(f"{value:.3f}" for value in step["hidden_preview"]),
                                "Gate means": ", ".join(f"{name}={value:.3f}" for name, value in step.get("gates", {}).items()) or "—",
                            }
                            for step in trace["steps"]
                        ]), hide_index=True, width="stretch")
                        st.caption(f"Final hidden state, first 8 of {trace['hidden_units']} values: {trace['final_hidden_preview']}")
                        if "final_cell_preview" in trace:
                            st.caption(f"Final LSTM cell state, first 8 values: {trace['final_cell_preview']}")
                        st.caption(f"Softmax probability total: {trace['output']['softmax_sum']:.6f}")
                        if result["model"] != "rnn":
                            st.caption("Gate means average each learned gate's activation across hidden units at that word.")
                        with st.expander("Raw output scores before softmax"):
                            st.dataframe(pd.DataFrame(trace["output"]["top_logits"]), hide_index=True, width="stretch")
                    else:
                        st.warning(trace["reason"])

            st.markdown("### 4 · What did it predict?")
            distribution_tabs = st.tabs([pretty_model(result["model"]) for result in model_results])
            for tab, result in zip(distribution_tabs, model_results):
                with tab:
                    for item in result["predictions"]:
                        rank_col, word_col, bar_col, probability_col = st.columns([0.55, 1.5, 5, 1])
                        rank_col.caption(f"{item['rank']:02d}")
                        word_col.markdown(f"**{item['word']}**")
                        bar_col.progress(min(max(float(item["probability"]), 0.0), 1.0))
                        probability_col.markdown(f"**{float(item['probability']):.2%}**")

            ground_truth = first_result["ground_truth"]
            st.markdown("### 5 · What was the actual next word?")
            if ground_truth["available"]:
                source = ground_truth.get("source", {})
                if "split" in source:
                    st.success(f"Ground truth: **{ground_truth['target']}** · source: **{source['split']}** split · `{source['id']}`")
                else:
                    st.success(f"Ground truth: **{ground_truth['target']}** · found in {source.get('matches', 1)} labelled record(s)")

                if ground_truth["representable"]:
                    st.markdown("### 6 · Was the model correct?")
                    summary_rows = []
                    for result in model_results:
                        truth = result["ground_truth"]
                        prediction = result["predictions"][0]
                        summary_rows.append(
                            {
                                "Model": pretty_model(result["model"]),
                                "Prediction": prediction["word"],
                                "Actual": truth["target"],
                                "Correct": "Yes" if truth["correct"] else "No",
                                "P(prediction)": f"{prediction['probability']:.3%}",
                                "P(actual)": f"{truth['probability']:.3%}",
                                "Probability gap": f"{prediction['probability'] - truth['probability']:.3%}",
                                "Actual rank": f"{truth['rank']:,} / {result['vocabulary_size']:,}",
                                "Top-5 hit": "Yes" if truth["rank"] <= 5 else "No",
                            }
                        )
                    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")
                    for result in model_results:
                        truth = result["ground_truth"]
                        if truth["correct"]:
                            st.success(f"{pretty_model(result['model'])} ranked the labelled word first in this example.")
                        else:
                            st.info(
                                f"{pretty_model(result['model'])} ranked **{result['predictions'][0]['word']}** first; "
                                f"the labelled word **{truth['target']}** was rank {truth['rank']:,} "
                                f"({truth['probability']:.3%}). DailyDialog supplies one reference word for each row."
                            )
                else:
                    st.warning(
                        f"The target word is outside the model vocabulary. Its probability and rank below refer to the `<unk>` bucket ({ground_truth['probability']:.3%}, rank {ground_truth['rank']:,}), not to the exact word."
                    )
            else:
                st.info(f"Ground truth unavailable: {ground_truth['reason']}")
                for candidate in ground_truth.get("candidates", []):
                    sources = ", ".join(f"{item['split']} · {item['id']}" for item in candidate["examples"])
                    st.caption(f"Stored target **{candidate['target']}**: {sources}")
                st.caption("Model predictions are still shown above. Select one exact dataset row to score an ambiguous context.")

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
