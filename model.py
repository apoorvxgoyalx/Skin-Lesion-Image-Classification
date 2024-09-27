import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your dataset
# Assume images and masks are loaded into numpy arrays
# images = ... (shape: num_images, height, width, channels)
# masks = ... (shape: num_images, height, width, num_classes)
# Set the path to your dataset
train_dir = "C:\\Users\\user\\Desktop\\jagriti\\archive (1)\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train"
test_dir = "C:\\Users\\user\\Desktop\\jagriti\\archive (1)\\Skin cancer ISIC The International Skin Imaging Collaboration\\Test"

# Parameters
batch_size = 32
img_height = 128
img_width = 128
num_classes = 9  # Change according to your classes

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# Display class names
class_names = train_ds.class_names
print(class_names)
# Normalize pixel values to [0, 1]
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, num_classes)))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, num_classes)))

# U-Net Model Definition
def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)
    
    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    x = layers.GlobalAveragePooling2D()(c9)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# Compile the model
model = unet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Ensure this is sparse_categorical_crossentropy

# Train the model
history = model.fit(train_ds, validation_data=test_ds, epochs=20)

# Evaluate the model
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy:.2f}")

# Visualize the results
# Modified display_samples function
def display_samples(dataset, model, num_samples=5):
    plt.figure(figsize=(20, 10))
    for images, labels in dataset.take(1):
        for i in range(num_samples):
            ax = plt.subplot(3, num_samples, i + 1)
            plt.imshow(images[i] * 255)
            plt.title(f"True: {class_names[tf.argmax(labels[i])]}")
            plt.axis("off")

            # Make predictions
            prediction = model.predict(tf.expand_dims(images[i], axis=0))
            predicted_class = tf.argmax(prediction, axis=-1)

            ax = plt.subplot(3, num_samples, i + num_samples + 1)
            plt.imshow(images[i] * 255)  # Display the image again
            plt.title(f"Pred: {class_names[predicted_class[0]]}")
            plt.axis("off")

            ax = plt.subplot(3, num_samples, i + 2*num_samples + 1)
            plt.bar(class_names, prediction[0])
            plt.xticks(rotation=90)
            plt.title("Class Probabilities")
    plt.tight_layout()
    plt.show()

# Display sample images and their predictions
display_samples(test_ds, model)