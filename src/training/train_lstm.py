"""Train Person C's LSTM on the shared next-word data and metric pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.models.lstm import LSTMConfig, build_lstm_model, compile_lstm_model
from src.training.data import Vocabulary, ensure_non_empty, load_examples
from src.training.evaluate import evaluate_model, write_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM for next-word prediction.")
    parser.add_argument("--train-data", type=Path, default=Path("data/processed/train.json"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/processed/validation.json"))
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.json"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/vocab.json"))
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-units", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--recurrent-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequence_length < 1 or args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise ValueError("sequence-length, epochs, and batch-size must be positive; patience cannot be negative.")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    vocabulary = Vocabulary.from_json(args.vocab)
    train_x, train_y = load_examples(args.train_data, vocabulary, args.sequence_length, args.max_train_samples)
    validation_x, validation_y = load_examples(
        args.validation_data, vocabulary, args.sequence_length, args.max_validation_samples
    )
    test_x, test_y = load_examples(args.test_data, vocabulary, args.sequence_length, args.max_test_samples)
    ensure_non_empty(train_x, train_y, validation_x, validation_y, test_x, test_y)

    config = LSTMConfig(args.embedding_dim, args.hidden_units, args.dropout, args.recurrent_dropout)
    model = build_lstm_model(vocabulary.size, args.sequence_length, config)
    compile_lstm_model(model, args.learning_rate)

    # Keep the conventional path for the baseline context and suffix other
    # experiments so running context 5 or 10 cannot overwrite context 3.
    suffix = "" if args.sequence_length == 3 else f"_ctx{args.sequence_length}"
    model_dir = args.artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / f"lstm{suffix}_best.keras"
    final_model_path = model_dir / f"lstm{suffix}_final.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True
        ),
    ]

    started = time.perf_counter()
    history = model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    training_seconds = time.perf_counter() - started
    model.save(final_model_path)

    best_model = tf.keras.models.load_model(best_model_path)
    metrics = evaluate_model(best_model, test_x, test_y, args.batch_size)
    metrics.update(
        {
            "model": "lstm",
            "choices": {
                "context_words": args.sequence_length,
                "embedding_dim": args.embedding_dim,
                "lstm_units": args.hidden_units,
                "dropout": args.dropout,
                "recurrent_dropout": args.recurrent_dropout,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
            "training_seconds": training_seconds,
            "epochs_completed": len(history.history["loss"]),
            "sequence_length": args.sequence_length,
            "best_checkpoint": str(best_model_path),
        }
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics, args.results_dir / f"lstm{suffix}_test_metrics.json")
    with (args.results_dir / f"lstm{suffix}_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)
    with (args.results_dir / f"lstm{suffix}_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
