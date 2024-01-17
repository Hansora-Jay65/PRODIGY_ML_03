import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Function to load and preprocess images
def load_and_preprocess_images(folder_path, label, num_images=2000):
    images = []
    labels = []
    for i in range(num_images):
        filename = os.listdir(folder_path)[i]
        img_path = os.path.join(folder_path, filename)  # Uncomment this line
        try:
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img = img.convert('RGB')

            images.append(np.array(img))
            labels.append(label)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    return np.array(images), np.array(labels)

# Load and preprocess Dog images
dog_images, dog_labels = load_and_preprocess_images('PetImages/Dog', 1)

# Load and preprocess Cat images
cat_images, cat_labels = load_and_preprocess_images('PetImages/Cat', 0)

# Concatenate and shuffle the data
X = np.concatenate([dog_images, cat_images], axis=0)
Y = np.concatenate([dog_labels, cat_labels], axis=0)
X, Y = shuffle(X, Y, random_state=2)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Normalize pixel values to be between 0 and 1
X_train_scaled, X_test_scaled = X_train / 255.0, X_test / 255.0

# Load MobileNetV2 model from TensorFlow Hub
mobilenet_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224, 224, 3), trainable=False)

# Build the model
num_of_classes = 2
model = tf.keras.Sequential([pretrained_model, tf.keras.layers.Dense(num_of_classes)])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['acc'])

# Train the model
batch_size = 32
history = model.fit(X_train_scaled, Y_train, epochs=5)

# Evaluate the model on the test set
score, acc = model.evaluate(X_test_scaled, Y_test)
print('Test Loss = ', score)
print('Test Accuracy = ', acc)

# Function to predict the label of a given image
def predict_image(input_img_path):
    input_image = Image.open(input_img_path)
    input_image = input_image.resize((224, 224))
    input_image = input_image.convert('RGB')

    input_image_array = np.array(input_image) / 255.0
    input_image_array = np.reshape(input_image_array, [1, 224, 224, 3])

    input_prediction = model.predict(input_image_array)
    input_pred_label = np.argmax(input_prediction)

    return input_pred_label

# Example usage
input_img_path = input('Path of the image to be predicted:')
predicted_label = predict_image(input_img_path)

if predicted_label == 0:
    print('The Image is represented as a Cat.')
else:
    print('The Image is represented as a Dog.')
