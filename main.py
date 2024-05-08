import cv2
import numpy as np
import tensorflow as tf
import cvzone
import keras
from cvzone.FaceDetectionModule import FaceDetector
# Load the pre-trained model
model = keras.models.load_model('emotion_model.h5', compile=True)
#model.compile(loss=keras.losses.CategoricalCrossentropy(),
              #optimizer=keras.optimizers.Adam(learning_rate=0.001),
              #metrics=['accuracy'])

#Face detector
detector = FaceDetector(minDetectionCon=0.5,modelSelection=0)


# Function to preprocess the frame
def preprocess_frame(frame, x, y, w, h):
    # Crop the face region
    face_roi = frame[y:y+h, x:x+w]
    # Resize to match model input size
    resized_face = cv2.resize(face_roi, (224, 224))

    return resized_face

# Access the camera feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read frame from the camera
    success, img = cap.read()

    # Detect faces
    img,bboxs = detector.findFaces(img,draw=False)
    
    # Process each detected face
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'

            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)
    
            # Preprocess the face region
            processed_face = preprocess_frame(img, x, y, w, h)
        
            # Make predictions
            predictions = model.predict(np.expand_dims(processed_face, axis=0))
        
            labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

            # Assign the label based on predictions
            label = labels[np.argmax(predictions)]  

            # ---- Draw Data  ---- #
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, label , (x, y - 15),border=5)
            cvzone.cornerRect(img, (x, y, w, h))

    # Display the result
    cv2.imshow('Emotion Detection', img)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

