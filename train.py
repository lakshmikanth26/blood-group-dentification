from data_loader import load_data
from model import create_model

def train_model(train_dir, val_dir, epochs=10):
    # Load data
    train_generator, val_generator = load_data(train_dir, val_dir)
    
    # Create model
    model = create_model()

    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # Save the model
    model.save('blood_cell_classifier_model.h5')
    
    return history

if __name__ == "__main__":
    train_model('PBC_dataset/train', 'PBC_dataset/val', epochs=10)
