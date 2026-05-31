"""
CNN Model Training Script for Drowsiness Detection
=================================================
This script defines and trains a Convolutional Neural Network (CNN) model
for classifying eye states (open/closed) in drowsiness detection.

The model uses a simple CNN architecture with multiple convolutional layers,
pooling, dropout for regularization, and dense layers for classification.

Note: This script requires training data in 'data/train' and 'data/valid' directories
with subdirectories for each class (e.g., 'Open' and 'Closed').

Author: Aditya Sreekumar Achary
Date: May 2026
"""

import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.keras.utils import to_categorical
import random, shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import load_model

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    """
    Create a data generator for training/validation data.
    
    Args:
        dir (str): Directory containing training/validation data
        gen: ImageDataGenerator instance with preprocessing options
        shuffle (bool): Whether to shuffle the data
        batch_size (int): Number of samples per batch
        target_size (tuple): Target size for images (height, width)
        class_mode (str): Type of classification ('categorical', 'binary', etc.)
    
    Returns:
        DirectoryIterator: Iterator that yields batches of images and labels
    """
    return gen.flow_from_directory(
        dir, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        color_mode='grayscale', 
        class_mode=class_mode, 
        target_size=target_size
    )

# Training configuration parameters
BS = 32  # Batch size for training
TS = (24, 24)  # Target size for input images (height, width)

# Create data generators for training and validation sets
# Note: These directories should contain subdirectories for each class
train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)

# Calculate steps per epoch for training and validation
SPE = len(train_batch.classes) // BS  # Steps per epoch for training
VS = len(valid_batch.classes) // BS   # Validation steps
print(f"Training steps per epoch: {SPE}, Validation steps: {VS}")

# Optional: Uncomment to inspect batch data
# img, labels = next(train_batch)
# print(f"Image shape: {img.shape}")

# Define the CNN model architecture
model = Sequential([
    # First convolutional block
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),  # Minimal pooling to preserve spatial information
    
    # Second convolutional block
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    
    # Third convolutional block with more filters
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    
    # Regularization to prevent overfitting
    Dropout(0.25),  # Randomly turn off 25% of neurons during training
    
    # Flatten the 2D feature maps into a 1D vector
    Flatten(),
    
    # Fully connected layer for feature combination
    Dense(128, activation='relu'),
    
    # Additional dropout for better generalization
    Dropout(0.5),  # Randomly turn off 50% of neurons during training
    
    # Output layer with softmax activation for binary classification
    Dense(2, activation='softmax')  # 2 classes: Open (1) and Closed (0)
])

# Compile the model with optimizer, loss function, and metrics
model.compile(
    optimizer='adam',  # Adaptive learning rate optimizer
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

# Train the model using the data generators
model.fit_generator(
    train_batch, 
    validation_data=valid_batch, 
    epochs=15,  # Number of complete passes through the training data
    steps_per_epoch=SPE,  # Number of batches per epoch
    validation_steps=VS   # Number of validation batches
)

# Save the trained model for later use
model.save('models/cnnCat2.h5', overwrite=True)
print("Model training completed and saved successfully!")