"""
Drowsiness Detection System
==========================
A real-time drowsiness detection system that uses computer vision and deep learning
to monitor eye states and alert users when they appear to be falling asleep.

This system captures video from the webcam, detects faces and eyes using Haar cascades,
and uses a trained CNN model to classify whether eyes are open or closed.
When drowsiness is detected (both eyes closed for extended periods), an alarm sounds.

Author: [Your Name]
Date: [Current Date]
"""

import cv2
import os
import numpy as np
from pygame import mixer
import time
from tensorflow.keras.models import load_model

# Initialize the audio system for alarm notifications
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascade classifiers for face and eye detection
# These XML files contain pre-trained models for detecting facial features
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Labels for eye state classification (0 = Closed, 1 = Open)
lbl = ['Close', 'Open']

# Load the pre-trained CNN model for eye state classification
# This model was trained to distinguish between open and closed eyes
model = load_model('models/cnncat2_new.h5')

# Initialize video capture and other variables
path = os.getcwd()  # Current working directory for saving images
cap = cv2.VideoCapture(0)  # Initialize webcam (0 for default camera)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # Font for text overlay
count = 0  # Counter for processed frames
score = 0  # Drowsiness score (increases when eyes are closed)
thicc = 2  # Thickness for warning rectangle
rpred = [99]  # Right eye prediction (99 = no detection yet)
lpred = [99]  # Left eye prediction (99 = no detection yet)

# Main processing loop - runs continuously until 'q' is pressed
while(True):
    # Capture frame from webcam
    ret, frame = cap.read()
    height, width = frame.shape[:2] 

    # Convert frame to grayscale for better face/eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces, left eyes, and right eyes using Haar cascades
    # These functions return rectangles (x, y, width, height) for each detected feature
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw a black rectangle at the bottom for status display
    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Draw rectangles around detected faces (for visualization)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    # Process right eye detection and classification
    for (x, y, w, h) in right_eye:
        # Extract the eye region from the frame
        r_eye = frame[y:y+h, x:x+w]
        count += 1
        
        # Preprocess the eye image for the CNN model
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        r_eye = cv2.resize(r_eye, (24, 24))  # Resize to model input size
        r_eye = r_eye / 255  # Normalize pixel values to [0,1]
        r_eye = r_eye.reshape(24, 24, -1)  # Reshape for model input
        r_eye = np.expand_dims(r_eye, axis=0)  # Add batch dimension
        
        # Use the CNN model to predict eye state (0=closed, 1=open)
        rpred = np.argmax(model.predict(r_eye, verbose=0), axis=1)
        
        # Update label based on prediction
        if(rpred[0] == 1):
            lbl = 'Open' 
        if(rpred[0] == 0):
            lbl = 'Closed'
        break  # Process only the first detected eye

    # Process left eye detection and classification (same as right eye)
    for (x, y, w, h) in left_eye:
        # Extract the eye region from the frame
        l_eye = frame[y:y+h, x:x+w]
        count += 1
        
        # Preprocess the eye image for the CNN model
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        l_eye = cv2.resize(l_eye, (24, 24))  # Resize to model input size
        l_eye = l_eye / 255  # Normalize pixel values to [0,1]
        l_eye = l_eye.reshape(24, 24, -1)  # Reshape for model input
        l_eye = np.expand_dims(l_eye, axis=0)  # Add batch dimension
        
        # Use the CNN model to predict eye state (0=closed, 1=open)
        lpred = np.argmax(model.predict(l_eye, verbose=0), axis=1)
        
        # Update label based on prediction
        if(lpred[0] == 1):
            lbl = 'Open'   
        if(lpred[0] == 0):
            lbl = 'Closed'
        break  # Process only the first detected eye

    # Drowsiness detection logic
    if(rpred[0] == 0 and lpred[0] == 0):
        # Both eyes are closed - increase drowsiness score
        score += 1
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # At least one eye is open - decrease drowsiness score
        score -= 1
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Ensure score doesn't go below zero
    if(score < 0):
        score = 0   
    
    # Display the current drowsiness score
    cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Alert system - trigger when drowsiness score exceeds threshold
    if(score > 15):
        # Person appears to be drowsy - activate alarm system
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)  # Save current frame
        
        # Play alarm sound (with error handling)
        try:
            sound.play()
        except:  # Handle cases where sound might not play
            pass
        
        # Create pulsing red border effect for visual alert
        if(thicc < 16):
            thicc += 2  # Increase border thickness
        else:
            thicc -= 2  # Decrease border thickness
            if(thicc < 2):
                thicc = 2  # Reset to minimum thickness
        
        # Draw red warning rectangle around the entire frame
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc) 
    
    # Display the processed frame
    cv2.imshow('Drowsiness Detection System', frame)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources when exiting
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
