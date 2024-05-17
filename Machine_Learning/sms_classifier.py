import pandas as pd
import tensorflow as tf
from keras.layers import TextVectorization, Embedding, Bidirectional, Dense
import matplotlib.pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


# load data
train_df = pd.read_csv('./SMS_data/train-data.tsv', sep='\t', names=['label', 'text'])
valid_df = pd.read_csv('./SMS_data/valid-data.tsv', sep='\t', names=['label', 'text'])

# create text encoder
encoder = TextVectorization()
encoder.adapt(train_df.to_numpy().reshape(-1))

# create model
model = tf.keras.Sequential([
    encoder,
    Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    Bidirectional(tf.keras.layers.LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
              metrics=['accuracy'])

# TODO ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
history = model.fit(train_df.to_numpy(), epochs=10,
                    validation_data=valid_df.to_numpy(),
                    validation_steps=30)

test_loss, test_acc = model.evaluate(valid_df.to_numpy().reshape(-1))

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
