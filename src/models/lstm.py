"""LSTM model used for Person C's next-word experiment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LSTMConfig:
    embedding_dim: int = 128
    hidden_units: int = 256
    dropout: float = 0.2
    recurrent_dropout: float = 0.0

    def validate(self) -> None:
        if self.embedding_dim < 1 or self.hidden_units < 1:
            raise ValueError("embedding_dim and hidden_units must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.recurrent_dropout < 1.0:
            raise ValueError("recurrent_dropout must be in [0, 1).")


def build_lstm_model(vocab_size: int, sequence_length: int, config: LSTMConfig):
    """Build an embedding → LSTM → vocabulary-softmax next-word model."""

    if vocab_size < 3:
        raise ValueError("vocab_size must include padding, OOV, and at least one token.")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    config.validate()

    import tensorflow as tf

    inputs = tf.keras.Input(shape=(sequence_length,), dtype="int32", name="context")
    embedded = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=True,
        name="token_embedding",
    )(inputs)
    encoded = tf.keras.layers.LSTM(
        config.hidden_units,
        dropout=config.dropout,
        recurrent_dropout=config.recurrent_dropout,
        name="lstm",
    )(embedded)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax", name="next_token")(encoded)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_next_word")


def compile_lstm_model(model, learning_rate: float) -> None:
    """Compile with the same objective and metrics used by the RNN experiment."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top_1_accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )
