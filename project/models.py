import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import numpy as np

class BERT:
    def __init__(self, module_url='https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1'):
        self.model = None
        self.bert_layer = hub.KerasLayer(module_url, trainable=True)

    def build_model(self, max_len=512):
        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        _, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        out = Dense(1, activation='sigmoid')(clf_output)

        self.model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        self.model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def train(self, train_input, train_labels, epochs=3, batch_size=16, path='model.h5'):
        train_history = self.model.fit(
            train_input, train_labels,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size
        )

        self.model.save(path)

        return train_history

