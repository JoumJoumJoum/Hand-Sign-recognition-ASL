import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import load_data
from CNN import create_model

def train_model():
    """Train the handsign detection model"""
    # Load and preprocess data
    train_ds, val_ds, class_names = load_data()
    
    # Create model
    img_size = 128
    num_classes = len(class_names)
    model = create_model(img_size, num_classes)
    
    print("Model Summary:")
    model.summary()
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        verbose=1
    )
    
    # Save the trained model
    model.save("hand_sign_model.h5")
    print("Model saved as 'hand_sign_model.h5'")
    
    # Plot accuracy and loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model, history

if __name__ == "__main__":
    train_model()
