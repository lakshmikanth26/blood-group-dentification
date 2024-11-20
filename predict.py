from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model('blood_cell_classifier_model.h5')

# Load and preprocess an image for prediction
img = image.load_img('test/BA_47.jpg', target_size=(128, 128))  # Change size to match your model's input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the class
prediction = model.predict(img_array)

# Print the prediction results
print("Prediction result:", prediction)
