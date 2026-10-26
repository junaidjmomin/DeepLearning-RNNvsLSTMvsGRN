"""Vanilla RNN model used for Person 2's experiment track."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RNNConfig:
    """Hyperparameters for the next-word vanilla RNN."""

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


def build_rnn_model(vocab_size: int, sequence_length: int, config: RNNConfig):
    """Build the simple vanilla RNN used as our baseline model."""

    if vocab_size < 3:
        raise ValueError("vocab_size must include at least padding, OOV, and one token.")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    config.validate()

    import tensorflow as tf

    # CHOICE 1: use a fixed number of previous words as the input context.
    inputs = tf.keras.Input(shape=(sequence_length,), dtype="int32", name="context")
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=True,
        name="token_embedding",
    )(inputs)
    # CHOICE 2: SimpleRNN is intentional here. GRU and LSTM belong to the
    # other team members, while Person 2 needs a clear vanilla-RNN baseline.
    x = tf.keras.layers.SimpleRNN(
        config.hidden_units,
        dropout=config.dropout,
        recurrent_dropout=config.recurrent_dropout,
        name="vanilla_rnn",
    )(x)
    # CHOICE 3: softmax gives a probability for every word in the vocabulary.
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax", name="next_token")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vanilla_rnn_next_word")


def compile_rnn_model(model, learning_rate: float) -> None:
    """Compile ``model`` with classification metrics appropriate for next-word prediction."""

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
