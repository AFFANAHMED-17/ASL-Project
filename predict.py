import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('asl_alphabet_model.h5')

# Load the label map
asl_label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}  # Complete this mapping based on your dataset

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to 64x64
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    
    # Get the corresponding label
    label = asl_label_map[predicted_class]
    
    # Display the label on the frame
    cv2.putText(frame, f'Predicted: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('ASL Alphabet Detection', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
