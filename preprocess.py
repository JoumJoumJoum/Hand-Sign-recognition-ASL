import tensorflow as tf
import matplotlib.pyplot as plt
import os

def load_data(data_dir="data", img_size=128, batch_size=32):
    """Load and preprocess the handsign dataset"""
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found!")
    
    print(f"Loading data from {data_dir}...")
    
    # Train and validation split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    
    # Get class names
    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")

    # Normalize pixel values to [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Prefetch for performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, class_names

def visualize_data(train_ds, class_names, num_samples=9):
    """Visualize some samples from the dataset"""
    plt.figure(figsize=(10, 10))
    
    for images, labels in train_ds.take(1):
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the data loading
    train_ds, val_ds, class_names = load_data()
    print(f"Training samples: {len(train_ds) * 32}")  # Approximate
    print(f"Validation samples: {len(val_ds) * 32}")  # Approximate
    visualize_data(train_ds, class_names)
