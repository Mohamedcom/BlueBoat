from tensorflow.keras import models, layers, Input

class ThreatModel:
    def build(self) -> models.Model:
        inp = Input(shape=(12,), name='threat_input')
        x = layers.Dense(512, activation='swish')(inp)
        x = layers.BatchNormalization()(x)
        att = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
        x = layers.Dropout(0.3)(att)
        out = layers.Dense(6, activation='sigmoid', name='threat_output')(x)
        return models.Model(inputs=inp, outputs=out)

class CurrentModel:
    def build(self) -> models.Model:
        inp = Input(shape=(8,), name='current_input')
        x = layers.Dense(256, activation='swish')(inp)
        x = layers.Dense(128, activation='swish')(x)
        return models.Model(inputs=inp, outputs=layers.Dense(3, name='current_output')(x))

class VesselBehaviorModel:
    def build(self) -> models.Model:
        inp = Input(shape=(10,), name='vessel_input')
        x = layers.Reshape((1,10))(inp)
        x = layers.LSTM(256, return_sequences=True)(x)
        att = layers.Attention()([x, x])
        flat = layers.Flatten()(att)
        return models.Model(inputs=inp, outputs=layers.Dense(4, name='vessel_output')(flat))
