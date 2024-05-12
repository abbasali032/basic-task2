import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load the pre-trained VGG16 model
model = VGG16()
# Function to preprocess an image for VGG16
def preprocess_image(image_path):
    # Load the image with target size 224x224 (required input size for VGG16)
    img = load_img(image_path, target_size=(224, 224))
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Expand the dimensions of the image to match the input shape of VGG16
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image for VGG16 model
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed

# Function to classify an image as dog or cat
def classify_image(image_path):
    # Preprocess the image
    img_preprocessed = preprocess_image(image_path)
    # Predict the probabilities for all classes
    predictions = model.predict(img_preprocessed)
    # Decode the predictions to get the human-readable labels
    label = decode_predictions(predictions)
    # Return the top prediction
    return label[0][0]

# Test the classifier on an example image
image_path = 'C:\\Users\\ABBAS SHAIK\\Downloads\\cat.jpg'  # Replace with the path to your image
prediction = classify_image(image_path)
print('Predicted:', prediction[1], '-', prediction[2])  # Print the label and the probability

# Display the example image
img = load_img(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
