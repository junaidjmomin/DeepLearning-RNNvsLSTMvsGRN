"""Unified training entry-point for all three model architectures.

Usage
-----
python -m src.training.train --model gru
python -m src.training.train --model rnn
python -m src.training.train --model lstm
python -m src.training.train --model gru --sequence-length 5
python -m src.training.train --model lstm --sequence-length 10

All three architectures share the same data splits, vocabulary, optimizer,
loss, and evaluation pipeline so results are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GRU, RNN, or LSTM for next-word prediction (fair comparison)."
    )
    parser.add_argument(
        "--model",
        choices=["gru", "rnn", "lstm"],
        required=True,
        help="Architecture to train.",
    )
    parser.add_argument("--train-data", type=Path, default=Path("data/processed/train.json"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/processed/validation.json"))
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.json"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/vocab.json"))

    # Shared experiment settings — keep identical across architectures for a
    # fair comparison (see README Phase 8).
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=3,
        help="Number of previous words used as context (3, 5, or 10).",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-units", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--recurrent-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # Optional sample caps for quick smoke-tests.
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)

    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_and_compile(model_name: str, vocab_size: int, sequence_length: int, args):
    """Return a compiled Keras model for the requested architecture."""

    if model_name == "gru":
        from src.models.gru import GRUConfig, build_gru_model, compile_gru_model

        config = GRUConfig(args.embedding_dim, args.hidden_units, args.dropout, args.recurrent_dropout)
        model = build_gru_model(vocab_size, sequence_length, config)
        compile_gru_model(model, args.learning_rate)
        choices = {
            "context_words": sequence_length,
            "embedding_dim": args.embedding_dim,
            "gru_units": args.hidden_units,
            "dropout": args.dropout,
            "recurrent_dropout": args.recurrent_dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }

    elif model_name == "rnn":
        from src.models.rnn import RNNConfig, build_rnn_model, compile_rnn_model

        config = RNNConfig(args.embedding_dim, args.hidden_units, args.dropout, args.recurrent_dropout)
        model = build_rnn_model(vocab_size, sequence_length, config)
        compile_rnn_model(model, args.learning_rate)
        choices = {
            "context_words": sequence_length,
            "embedding_dim": args.embedding_dim,
            "rnn_units": args.hidden_units,
            "dropout": args.dropout,
            "recurrent_dropout": args.recurrent_dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }

    else:  # lstm
        from src.models.lstm import LSTMConfig, build_lstm_model, compile_lstm_model

        config = LSTMConfig(args.embedding_dim, args.hidden_units, args.dropout, args.recurrent_dropout)
        model = build_lstm_model(vocab_size, sequence_length, config)
        compile_lstm_model(model, args.learning_rate)
        choices = {
            "context_words": sequence_length,
            "embedding_dim": args.embedding_dim,
            "lstm_units": args.hidden_units,
            "dropout": args.dropout,
            "recurrent_dropout": args.recurrent_dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }

    return model, choices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 0 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive; patience cannot be negative.")

    import tensorflow as tf

    from src.training.data import Vocabulary, ensure_non_empty, load_examples
    from src.training.evaluate import evaluate_model, write_metrics

    tf.keras.utils.set_random_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  ContextNext — training {args.model.upper()} (context={args.sequence_length})")
    print(f"{'='*60}\n")

    vocabulary = Vocabulary.from_json(args.vocab)
    train_x, train_y = load_examples(args.train_data, vocabulary, args.sequence_length, args.max_train_samples)
    val_x, val_y = load_examples(args.validation_data, vocabulary, args.sequence_length, args.max_validation_samples)
    test_x, test_y = load_examples(args.test_data, vocabulary, args.sequence_length, args.max_test_samples)
    ensure_non_empty(train_x, val_x, test_x)

    print(f"Train samples   : {len(train_x):,}")
    print(f"Val samples     : {len(val_x):,}")
    print(f"Test samples    : {len(test_x):,}")
    print(f"Vocabulary size : {vocabulary.size:,}")

    model, choices = _build_and_compile(args.model, vocabulary.size, args.sequence_length, args)
    model.summary()

    # Artifact naming: context-length suffix keeps different runs from
    # overwriting each other (baseline=3 has no suffix for backwards compat).
    suffix = "" if args.sequence_length == 3 else f"_ctx{args.sequence_length}"
    model_dir = args.artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / f"{args.model}{suffix}_best.keras"
    final_path = model_dir / f"{args.model}{suffix}_final.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
        ),
    ]

    started = time.perf_counter()
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    training_seconds = time.perf_counter() - started
    model.save(final_path)
    print(f"\nSaved final model -> {final_path}")

    best_model = tf.keras.models.load_model(best_path)
    metrics = evaluate_model(best_model, test_x, test_y, args.batch_size)
    metrics.update(
        {
            "model": args.model,
            "choices": choices,
            "training_seconds": training_seconds,
            "epochs_completed": len(history.history["loss"]),
            "sequence_length": args.sequence_length,
            "best_checkpoint": str(best_path),
        }
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics, args.results_dir / f"{args.model}{suffix}_test_metrics.json")
    with (args.results_dir / f"{args.model}{suffix}_history.json").open("w", encoding="utf-8") as fh:
        json.dump(history.history, fh, indent=2)
    with (args.results_dir / f"{args.model}{suffix}_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  FINAL TEST METRICS")
    print("=" * 60)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
