# Skin Lesion Classification Project Documentation

## Project Overview

This project implements a deep learning model for classifying skin lesions into nine different categories. It utilizes a modified U-Net architecture adapted for image classification tasks. The model is trained on a dataset of skin lesion images and can predict the type of skin lesion given an input image.

### Key Features:
- Multi-class classification of skin lesions
- Modified U-Net architecture for image classification
- TensorFlow and Keras implementation
- Data augmentation for improved model generalization
- Visualization of model predictions and class probabilities

## Dataset

The dataset used in this project consists of skin lesion images categorized into nine classes:

1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion

The dataset is split into training and testing sets, stored in separate directories.

## Model Architecture

The model is based on the U-Net architecture, originally designed for image segmentation tasks. It has been modified for image classification:

1. **Encoder**: Consists of convolutional and max pooling layers that extract features from the input image.
2. **Decoder**: Uses transposed convolutions to upsample the feature maps.
3. **Classification Head**: 
   - Global Average Pooling layer to reduce spatial dimensions
   - Dense layer with softmax activation for final classification

## Implementation Details

### Dependencies
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

### Data Preprocessing
- Images are resized to 128x128 pixels
- Pixel values are normalized to the range [0, 1]
- Labels are one-hot encoded

### Model Training
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metrics: Accuracy
- Number of epochs: 20 (adjustable)

### Evaluation
- The model is evaluated on a separate test dataset
- Metrics include loss and accuracy

### Visualization
- Sample images from the test set are displayed along with their true and predicted labels
- Class probabilities for each prediction are visualized using bar plots

## Usage

1. Prepare your dataset in the specified directory structure
2. Adjust hyperparameters if needed (e.g., batch size, image dimensions, number of epochs)
3. Run the script to train the model
4. Evaluate the model's performance on the test set
5. Use the `display_samples` function to visualize predictions

## Future Improvements

1. Implement cross-validation for more robust evaluation
2. Experiment with different model architectures or transfer learning
3. Add data augmentation techniques to improve model generalization
4. Implement early stopping to prevent overfitting
5. Explore interpretability techniques (e.g., Grad-CAM) to visualize important regions in the input images

## Conclusion

This project demonstrates the application of deep learning techniques for skin lesion classification. The modified U-Net architecture shows promise in accurately categorizing various types of skin lesions, which could potentially assist in early detection and diagnosis of skin conditions.
