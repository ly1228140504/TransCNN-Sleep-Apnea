import tensorflow as tf
from tensorflow.keras import layers, regularizers

# --- Model Configuration ---
filter_config = [8, 12, 16, 24]


@tf.keras.utils.register_keras_serializable()
class CSA_Channel(layers.Layer):
    """
    CSA_Channel Block for two parallel input streams.
    It learns channel-wise attention weights by considering both streams.
    """

    def __init__(self, channels, reduction=16, **kwargs):
        super(CSA_Channel, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.global_avg_pool = layers.GlobalAvgPool1D()
        self.concat = layers.Concatenate(axis=-1)
        self.fc1 = layers.Dense(2 * channels // reduction, activation='relu', use_bias=False, name="se_fc1")
        self.fc2 = layers.Dense(2 * channels, activation='sigmoid', use_bias=False, name="se_fc2")
        self.reshape = layers.Reshape((1, channels), name="reshape_weights")
        self.dropout = layers.Dropout(rate=0.2)

    def call(self, input1, input2):
        y1 = self.global_avg_pool(input1)
        y2 = self.global_avg_pool(input2)
        y_concat = self.concat([y1, y2])
        y_concat = self.fc1(y_concat)
        y_concat = self.fc2(y_concat)

        y_w1, y_w2 = tf.split(y_concat, num_or_size_splits=2, axis=-1)

        y_w1 = self.reshape(y_w1)
        y_w2 = self.reshape(y_w2)

        output1 = self.dropout(input1 * y_w1)
        output2 = self.dropout(input2 * y_w2)

        return output1, output2

    def get_config(self):
        config = super(CSA_Channel, self).get_config()
        config.update({"channels": self.channels, "reduction": self.reduction})
        return config


@tf.keras.utils.register_keras_serializable()
class CSA_Temporal(layers.Layer):
    """
    CSA_Temporal Block focusing on the sequence dimension for two streams.
    It learns to re-weight the importance of different time steps.
    """

    def __init__(self, seq1, seq2, reduction=16, **kwargs):
        super(CSA_Temporal, self).__init__(**kwargs)
        self.seq1, self.seq2 = seq1, seq2
        self.reduction = reduction
        self.global_avg_pool = layers.GlobalAvgPool1D(data_format="channels_first")
        self.concat = layers.Concatenate(axis=1)
        self.fc1 = layers.Dense((self.seq1 + self.seq2) // reduction, activation='relu', use_bias=False)
        self.fc2 = layers.Dense(self.seq1 + self.seq2, activation='sigmoid', use_bias=False)
        self.gate1 = layers.Dense(1, activation="sigmoid")
        self.gate2 = layers.Dense(1, activation="sigmoid")
        self.reshape1 = layers.Reshape((seq1, 1))
        self.reshape2 = layers.Reshape((seq2, 1))
        self.dropout = layers.Dropout(rate=0.2)

    def call(self, input1, input2):
        y1 = self.global_avg_pool(input1)
        y2 = self.global_avg_pool(input2)

        gate_y1 = self.gate1(y1)
        gate_y2 = self.gate2(y2)

        y_concat = self.concat([y1 * gate_y1, y2 * gate_y2])
        y_concat = self.fc1(y_concat)
        y_concat = self.fc2(y_concat)

        y_w1, y_w2 = tf.split(y_concat, num_or_size_splits=[self.seq1, self.seq2], axis=-1)

        y_w1 = self.reshape1(y_w1)
        y_w2 = self.reshape2(y_w2)

        output1 = self.dropout(input1 * y_w1)
        output2 = self.dropout(input2 * y_w2)

        return output1, output2

    def get_config(self):
        config = super(CSA_Temporal, self).get_config()
        config.update({"seq1": self.seq1, "seq2": self.seq2, "reduction": self.reduction})
        return config


def create_model(input_a_shape, input_b_shape, weight=1e-2):
    """
    Builds the complete MSCNN model (formerly M10).
    This model uses two parallel branches for 5-min and 1-min signals,
    integrates channel and sequence SE blocks, and a final weighted combiner.
    """

    # --- Input Layers ---
    input1 = layers.Input(shape=input_a_shape, name="input_5min")  # 5-min signal
    input2 = layers.Input(shape=input_b_shape, name="input_1min")  # 1-min signal

    # --- Shared Block 1 ---
    x1 = layers.Conv1D(filter_config[0], 11, padding="same", activation="relu", kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.L2(weight))(input1)
    x1 = layers.BatchNormalization()(x1)
    x2 = layers.Conv1D(filter_config[0], 11, padding="same", activation="relu", kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.L2(weight))(input2)
    x2 = layers.BatchNormalization()(x2)
    x1, x2 = CSA_Channel(channels=filter_config[0], reduction=4)(x1, x2)

    # --- Shared Block 2 ---
    x1 = layers.Conv1D(filter_config[1], 11, strides=2, padding="same", activation="relu",
                       kernel_initializer="he_normal", kernel_regularizer=regularizers.L2(weight))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(pool_size=3, padding="same")(x1)
    x2 = layers.Conv1D(filter_config[1], 11, strides=2, padding="same", activation="relu",
                       kernel_initializer="he_normal", kernel_regularizer=regularizers.L2(weight))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling1D(pool_size=3, padding="same")(x2)
    x1, x2 = CSA_Channel(channels=filter_config[1], reduction=4)(x1, x2)

    # --- Shared Block 3 with Sequence Attention ---
    x1 = layers.Conv1D(filter_config[2], 11, padding="same", activation="relu", kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.L2(weight))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(pool_size=5, padding="same")(x1)
    x2 = layers.Conv1D(filter_config[2], 11, padding="same", activation="relu", kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.L2(weight))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling1D(pool_size=2, padding="same")(x2)
    x1, x2 = CSA_Temporal(seq1=x1.shape[1], seq2=x2.shape[1], reduction=4)(x1, x2)

    # --- Shared Block 4 with Sequence Attention ---
    x1 = layers.Conv1D(filter_config[3], 11, padding="same", activation="relu", kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.L2(weight))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(pool_size=5, padding="same")(x1)
    x2 = layers.Conv1D(filter_config[3], 11, padding="same", activation="relu", kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.L2(weight))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling1D(pool_size=3, padding="same")(x2)
    x1, x2 = CSA_Temporal(seq1=x1.shape[1], seq2=x2.shape[1], reduction=2)(x1, x2)

    # --- Feature Aggregation and Combination ---
    x1 = layers.GlobalAveragePooling1D()(x1)
    x2 = layers.GlobalAveragePooling1D()(x2)

    alpha = layers.Dense(1, activation='sigmoid', name="alpha")(x1)
    beta = 1.0 - alpha
    x1_weighted = layers.Multiply(name="weighted_x1")([alpha, x1])
    x2_weighted = layers.Multiply(name="weighted_x2")([beta, x2])

    # --- Classifier Head ---
    x_output = layers.Concatenate(name="combined_features")([x1_weighted, x2_weighted])
    x_output = layers.Dropout(0.6)(x_output)
    x_output = layers.Dense(filter_config[3]*1.5, activation='relu')(x_output)
    x_output = layers.Dropout(0.6)(x_output)
    x_output = layers.Dense(2, activation='softmax')(x_output)

    model = tf.keras.Model(inputs=[input1, input2], outputs=x_output)
    return model