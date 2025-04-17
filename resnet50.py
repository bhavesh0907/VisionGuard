import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import splitfolders

def build_resnet50_model(input_shape, num_classes):
    """Builds a ResNet50 model for eye disease detection."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_dataset(directory, batch_size=32, image_size=(224, 224)):
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
    """Trains the ResNet50 model."""
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )
    return model

# Example Usage:
if __name__ == "__main__":
    input_folder = 'D:/Yugam/College/3rd Year/6th Sem/Project/Eye_disease_prediction-main/dataset' # Replace with your dataset location
    output_folder = 'splitted_data'
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.8, 0.1, 0.1))

    train_dir = os.path.join(output_folder, 'train')
    val_dir = os.path.join(output_folder, 'val')
    test_dir = os.path.join(output_folder, 'test')

    num_classes = len(os.listdir(train_dir))
    input_shape = (224, 224, 3)
    model = build_resnet50_model(input_shape, num_classes)

    train_dataset = create_dataset(train_dir)
    validation_dataset = create_dataset(val_dir)
    test_dataset = create_dataset(test_dir)

    trained_model = train_model(model, train_dataset, validation_dataset, epochs=10)

    test_loss, test_accuracy = trained_model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    trained_model.save('eye_disease_resnet50.h5')
    print("Model training and evaluation completed and saved.")