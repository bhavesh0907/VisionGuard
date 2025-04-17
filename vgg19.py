import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import os
import splitfolders

def build_vgg19_model(input_shape, num_classes):
    """Builds a VGG19 model for eye disease detection."""
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base VGG19 layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_dataset(directory, batch_size=32, image_size=(224, 224)): # VGG19 input size
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
    """Trains the VGG19 model."""
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )
    return model

# Example Usage:
if __name__ == "__main__":
    input_folder = 'D:/Yugam/College/3rd Year/6th Sem/Project/Eye_disease_prediction-main/dataset' # Replace with your dataset location
    output_folder = 'splitted_data_vgg19'
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.8, 0.1, 0.1))

    train_dir = os.path.join(output_folder, 'train')
    val_dir = os.path.join(output_folder, 'val')
    test_dir = os.path.join(output_folder, 'test')

    num_classes = len(os.listdir(train_dir))
    input_shape = (224, 224, 3) # VGG19 input size
    model = build_vgg19_model(input_shape, num_classes)

    train_dataset = create_dataset(train_dir, image_size=(224, 224)) # VGG19 input size
    validation_dataset = create_dataset(val_dir, image_size=(224, 224)) # VGG19 input size
    test_dataset = create_dataset(test_dir, image_size=(224, 224)) # VGG19 input size

    trained_model = train_model(model, train_dataset, validation_dataset, epochs=10)

    test_loss, test_accuracy = trained_model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    trained_model.save('eye_disease_vgg19.h5')
    print("VGG19 model training and evaluation completed and saved.")