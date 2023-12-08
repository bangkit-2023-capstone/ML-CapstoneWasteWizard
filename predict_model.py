import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = load_model('/Users/wrath/Downloads/ML-CapstoneWasteWizard/model/model_v1.h5')

# # Load and preprocess the image
img_path = '/Users/wrath/Downloads/ML-CapstoneWasteWizard/prediction/kentang1.jpeg'
img = image.load_img(img_path, target_size=(150, 150))  # Resize the image to match the input size expected by the model
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
img_array = preprocess_input(img_array)  # Preprocess the input data according to the model's requirements

predictions = model.predict(img_array)

print(predictions)

predicted_class = np.argmax(predictions)

if predicted_class == 0: print("\033[94m"+"This image -> Recyclable"+"\033[0m")
elif predicted_class == 1: print("\033[94m"+"This image -> Organic"+"\033[0m")
