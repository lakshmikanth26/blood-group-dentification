from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
model = load_model('blood_cell_classifier_model.h5')

# Define the test data directory
test_data_dir = 'test'

# Prepare the test data generator
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(128, 128),  # Same size as the input size used for training
    batch_size=32,
    class_mode='categorical'  # For multi-class classification
)

# Print the number of samples to check if data is loaded correctly
print("Number of test samples:", test_generator.samples)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
