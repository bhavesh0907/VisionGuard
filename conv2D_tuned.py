import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import splitfolders

def build_tuned_conv2d_model(input_shape, num_classes):
    """Builds a tuned Conv2D model with hyperparameters."""
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_data_generators(train_dir, val_dir, test_dir, image_size=(150, 150), batch_size=32):
    """Creates data generators with image augmentation."""

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator

def lr_schedule(epoch):
    """Learning rate scheduler."""
    lr = 1e-3
    if epoch > 20:
        lr *= 0.1
    return lr

if __name__ == "__main__":
    input_folder = 'D:/Yugam/College/3rd Year/6th Sem/Project/Eye_disease_prediction-main/dataset' # Replace with your dataset location
    output_folder = 'splitted_data_tuned_conv2D'
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.8, 0.1, 0.1))

    train_dir = os.path.join(output_folder, 'train')
    val_dir = os.path.join(output_folder, 'val')
    test_dir = os.path.join(output_folder, 'test')

    num_classes = len(os.listdir(train_dir))
    input_shape = (150, 150, 3)
    model = build_tuned_conv2d_model(input_shape, num_classes)

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_generator, validation_generator, test_generator = create_data_generators(train_dir, val_dir, test_dir)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    trained_model = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[lr_scheduler, early_stopping]
    )

    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    model.save('eye_disease_tuned_conv2d.h5')
    print("Tuned Conv2D model training and evaluation completed and saved.")