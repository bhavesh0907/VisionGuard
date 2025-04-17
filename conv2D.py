import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import splitfolders

def build_conv2d_model(input_shape, num_classes):
    """Builds a Conv2D model for eye disease detection."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_dataset(directory, batch_size=32, image_size=(150, 150)): #reduced image size
    """Creates a TensorFlow dataset from a directory of images."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )
    return dataset

def train_model(model, train_dataset, validation_dataset, epochs=10):
    """Trains the Conv2D model."""
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )
    return model

# Example Usage:
if __name__ == "__main__":
    input_folder = 'D:/Yugam/College/3rd Year/6th Sem/Project/Eye_disease_prediction-main/dataset' # Replace with your dataset location
    output_folder = 'splitted_data_conv2d'
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.8, 0.1, 0.1))

    train_dir = os.path.join(output_folder, 'train')
    val_dir = os.path.join(output_folder, 'val')
    test_dir = os.path.join(output_folder, 'test')

    num_classes = len(os.listdir(train_dir))
    input_shape = (150, 150, 3) #reduced image size
    model = build_conv2d_model(input_shape, num_classes)

    train_dataset = create_dataset(train_dir, image_size=(150,150)) #reduced image size
    validation_dataset = create_dataset(val_dir, image_size=(150,150)) #reduced image size
    test_dataset = create_dataset(test_dir, image_size=(150,150)) #reduced image size

    trained_model = train_model(model, train_dataset, validation_dataset, epochs=10)

    test_loss, test_accuracy = trained_model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    trained_model.save('eye_disease_conv2d.h5')
    print("Conv2D model training and evaluation completed and saved.")