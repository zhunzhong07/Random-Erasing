import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import math
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from random_eraser import RandomErasing


test_file = "fashion-mnist_test.csv"
train_file = "fashion-mnist_train.csv"

test_df = pd.read_csv(test_file)
train_df = pd.read_csv(train_file)

labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

X_test = test_df.drop(columns="label").values
y_test = test_df['label'].values
X_train = train_df.drop(columns='label').values
y_train = train_df['label'].values

X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

fig, axes = plt.subplots(12, 12, figsize=(15, 15))

random_eraser = RandomErasing()

fig, axes = plt.subplots(12, 24, figsize=(30, 15))

for i in range(12 * 12):
    index = np.random.randint(0, len(X_train))
    original_image = X_train[index]
    transformed_image = random_eraser(original_image)

    axes[i // 12, 2 * (i % 12)].imshow(original_image, cmap='gray')
    axes[i // 12, 2 * (i % 12)].set_title(f"Original\n{labels[y_train[index]]}")
    axes[i // 12, 2 * (i % 12)].axis('off')

    axes[i // 12, 2 * (i % 12) + 1].imshow(transformed_image, cmap='gray')
    axes[i // 12, 2 * (i % 12) + 1].set_title(f"Random Erasing\n{labels[y_train[index]]}")
    axes[i // 12, 2 * (i % 12) + 1].axis('off')

plt.tight_layout()
plt.show()


# Preprocess the data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, kernel_size=3, activation="relu", padding="same", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Define the learning rate schedule
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
)

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Train the model
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Plot the training loss and accuracy
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make predictions on the test data
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Show sample predictions
fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"True: {labels[y_test[i]]}\nPred: {labels[predicted_labels[i]]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
