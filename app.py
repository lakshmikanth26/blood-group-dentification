from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('blood_cell_classifier_model.h5')

# Define the class names (your categories)
class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

# Upload folder for images
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the blood group map for cell types
blood_group_map = {
    'basophil': 'A+', 
    'eosinophil': 'B+',
    'erythroblast': 'O+',
    'ig': 'AB+',
    'lymphocyte': 'A-', 
    'monocyte': 'B-',
    'neutrophil': 'O-',
    'platelet': 'AB-'
}

# Route to the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Prepare the image for prediction
    img = image.load_img(img_path, target_size=(128, 128))  # Adjust the target size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize image

    # Make the prediction
    predictions = model.predict(img_array)
    print(predictions)  # Print the raw model output
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(f"Predicted Class: {class_names[predicted_class]}")

    # Map the predicted class to the blood group
    predicted_blood_group = blood_group_map[class_names[predicted_class]]
    
    # Return the predicted blood group
    return jsonify({'predicted_blood_group': predicted_blood_group})

if __name__ == '__main__':
    app.run(debug=True)
