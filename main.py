import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def load_asl_alphabet(data_dir):
    images = []
    labels = []
    label_map = {}
    for label, letter_dir in enumerate(os.listdir(data_dir)):
        letter_path = os.path.join(data_dir, letter_dir)
        if os.path.isdir(letter_path):
            label_map[label] = letter_dir  # Store label mapping
            for img_file in os.listdir(letter_path):
                img_path = os.path.join(letter_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:  # Check if image is loaded
                    img = cv2.resize(img, (64, 64))  # Resize image to 64x64
                    img = img / 255.0  # Normalize the image
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels), label_map
asl_train_directory = r"D:\Sign-language-detection\sign_language_dataset\ASL_Alphabet\asl_alphabet_train\asl_alphabet_train"
X_train_asl, y_train_asl, asl_label_map = load_asl_alphabet(asl_train_directory)
asl_test_directory = r"D:\Sign-language-detection\sign_language_dataset\ASL_Alphabet\asl_alphabet_test\asl_alphabet_test"
X_test_asl, y_test_asl, _ = load_asl_alphabet(asl_test_directory)
if X_test_asl.size == 0 or y_test_asl.size == 0:
    print("Warning: Test dataset is empty. Check the directory and ensure images are available.")
else:
    X_train, X_val, y_train, y_val = train_test_split(X_train_asl, y_train_asl, test_size=0.2, random_state=42)
    print(f"Shape of X_test_asl: {X_test_asl.shape}")  # Expected: (num_samples, 64, 64, 3)
    print(f"Shape of y_test_asl: {y_test_asl.shape}")  # Expected: (num_samples,)
    print(f"Training Data: {X_train.shape}, Training Labels: {y_train.shape}")
    print(f"Validation Data: {X_val.shape}, Validation Labels: {y_val.shape}")
    print(f"Testing Data: {X_test_asl.shape}, Testing Labels: {y_test_asl.shape}")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(asl_label_map), activation='softmax')  # Output layer for number of classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    test_loss, test_accuracy = model.evaluate(X_test_asl, y_test_asl)
    print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')
    y_pred = np.argmax(model.predict(X_test_asl), axis=-1)
    print(classification_report(y_test_asl, y_pred, target_names=list(asl_label_map.values())))
    conf_matrix = confusion_matrix(y_test_asl, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(asl_label_map.values()), yticklabels=list(asl_label_map.values()))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
