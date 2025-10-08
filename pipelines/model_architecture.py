from __future__ import annotations

import tensorflow as tf

layers = tf.keras.layers


class MatchingModelBuilder:
    """Xây dựng model Deep Learning với Residual Block và Attention."""

    def __init__(self, input_dim: int, dropout_rate: float = 0.2) -> None:
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

    def _residual_block(self, x: tf.Tensor, units: int) -> tf.Tensor:
        shortcut = x
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != units:
            shortcut = layers.Dense(units)(shortcut)
        x = layers.Add()([x, shortcut])
        return layers.Activation("relu")(x)

    def build(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.input_dim,), name="matching_inputs")

        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = self._residual_block(x, 256)
        x = self._residual_block(x, 128)
        x = self._residual_block(x, 64)

        attention_input = layers.Reshape((8, x.shape[-1] // 8))(x)
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=16)(attention_input, attention_input)
        attention_output = layers.Flatten()(attention_output)

        combined = layers.Concatenate()([x, attention_output])
        combined = layers.Dense(128, activation="relu")(combined)
        combined = layers.Dropout(self.dropout_rate)(combined)
        combined = layers.Dense(64, activation="relu")(combined)
        combined = layers.Dropout(self.dropout_rate / 2)(combined)

        output = layers.Dense(1, activation="sigmoid", name="match_probability")(combined)

        model = tf.keras.Model(inputs=inputs, outputs=output, name="gamenect_matching_model")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        return model
