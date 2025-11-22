
# train_model.py
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescale for testing
test_datagen = ImageDataGenerator(rescale=1/255.0)

# Load data from folders
train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)

# Build CNN
model = models.Sequential([
    layers.Input(shape=(64,64,3)),  # Input layer
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# Save model
model.save("my_model.h5")
print("Model saved as my_model.h5")

