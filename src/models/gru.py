"""GRU model used for Person A's next-word experiment.

Architecture mirrors the RNN and LSTM modules so all three can be
compared fairly under identical training conditions.
"""


# Terminal 1
# python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2
# python -m streamlit run app/streamlit_app.py --server.port 8501


from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GRUConfig:
    """Hyperparameters for the next-word GRU model."""

    embedding_dim: int = 128
    hidden_units: int = 256
    dropout: float = 0.2
    recurrent_dropout: float = 0.0

    def validate(self) -> None:
        if self.embedding_dim < 1 or self.hidden_units < 1:
            raise ValueError("embedding_d im and hidden_units must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.recurrent_dropout < 1.0:
            raise ValueError("recurrent_dropout must be in [0, 1).")


def build_gru_model(vocab_size: int, sequence_length: int, config: GRUConfig):
    """Build an embedding → GRU → vocabulary-softmax next-word model.

    The architecture is intentionally comparable to the RNN and LSTM tracks:
    one recurrent layer between the shared embedding and the softmax output so
    differences in test metrics reflect the gating mechanism, not capacity.
    """

    if vocab_size < 3:
        raise ValueError("vocab_size must include padding, OOV, and at least one token.")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    config.validate()

    import tensorflow as tf

    # Integer context IDs shaped (batch, sequence_length).
    inputs = tf.keras.Input(shape=(sequence_length,), dtype="int32", name="context")

    # Shared embedding with mask so padding zeros do not contribute to the
    # hidden state — same choice as the RNN and LSTM experiments.
    embedded = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=True,
        name="token_embedding",
    )(inputs)

    # GRU uses reset and update gates to control memory flow, which lets it
    # handle longer dependencies than a vanilla RNN without the full gate set
    # of LSTM.  Only the last step's output is returned (return_sequences=False).
    encoded = tf.keras.layers.GRU(
        config.hidden_units,
        dropout=config.dropout,
        recurrent_dropout=config.recurrent_dropout,
        name="gru",
    )(embedded)

    # Softmax produces a proper probability distribution over the vocabulary.
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax", name="next_token")(encoded)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="gru_next_word")


def compile_gru_model(model, learning_rate: float) -> None:
    """Compile with the same objective and metrics used by the RNN and LSTM experiments."""

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
