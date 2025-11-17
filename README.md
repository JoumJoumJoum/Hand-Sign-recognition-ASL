# Hand Sign Detection Project

A real-time hand sign recognition system using Convolutional Neural Networks (CNN) with TensorFlow/Keras.

## Features

- **Real-time Detection**: Live hand sign recognition using webcam
- **Multiple Signs**: Supports 26 letters (A-Z) plus space and "nothing" classes
- **High Accuracy**: CNN-based model with data augmentation
- **Easy to Use**: Simple command-line interface

## Dataset

The project uses a dataset with hand sign images organized in folders:
- Each letter (A-Z) has ~3000 training images
- Additional classes: "space", "nothing", "del"
- Images are automatically resized to 128x128 pixels

## Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Make sure you have a webcam** for real-time detection

## Usage

### 1. Train the Model

First, you need to train the model on your dataset:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train a CNN model for 15 epochs
- Save the trained model as `hand_sign_model.h5`
- Display training progress and save accuracy/loss plots

### 2. Real-time Detection

After training, run real-time detection:

```bash
python realtime.py
```

This will:
- Open your webcam
- Show live hand sign recognition
- Display the predicted sign and confidence score
- Press 'q' to quit

### 3. Evaluate the Model

To see detailed performance metrics:

```bash
python evaluate.py
```

This will:
- Load the trained model
- Test on validation data
- Show classification report and confusion matrix
- Display per-class accuracy

### 4. Test Data Loading

To verify your dataset is properly loaded:

```bash
python preprocess.py
```

This will:
- Load and display sample images from each class
- Show dataset statistics

## Project Structure

```
├── data/                    # Dataset folder
│   ├── A/                  # Letter A images
│   ├── B/                  # Letter B images
│   ├── ...                 # Other letters
│   ├── space/              # Space gesture
│   └── nothing/            # No gesture
├── CNN.py                  # Model architecture
├── preprocess.py           # Data loading and preprocessing
├── train.py               # Training script
├── realtime.py            # Real-time detection
├── evaluate.py            # Model evaluation
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Model Architecture

The CNN model includes:
- 3 Convolutional layers (32, 64, 128 filters)
- MaxPooling after each conv layer
- Dense layer with 128 neurons
- Dropout for regularization
- Output layer with softmax activation

## Troubleshooting

### Common Issues:

1. **"Model file not found"**: Run `python train.py` first to train the model
2. **"Could not open webcam"**: Check if your webcam is working and not used by another application
3. **"Data directory not found"**: Make sure the `data/` folder exists with your images
4. **Import errors**: Install all dependencies with `pip install -r requirements.txt`

### Performance Tips:

- Ensure good lighting for better recognition
- Keep your hand centered in the camera frame
- Make clear, distinct hand signs
- The model works best with consistent hand positioning

## Customization

- **Add new signs**: Add new folders to the `data/` directory and retrain
- **Change model**: Modify the architecture in `CNN.py`
- **Adjust training**: Change epochs, batch size, or learning rate in `train.py`
- **Image size**: Modify `img_size` parameter in the scripts (default: 128x128)

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- OpenCV
- Webcam for real-time detection
- ~2GB RAM for training
- GPU recommended for faster training

