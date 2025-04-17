import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import splitfolders

def build_tuned_resnet50_model(input_shape, num_classes, learning_rate=1e-4, dropout_rate=0.5):
    """Builds a tuned ResNet50 model with hyperparameters."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_data_generators(train_dir, val_dir, test_dir, image_size=(224, 224), batch_size=32):
    """Creates data generators with image augmentation for ResNet50."""

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
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
        class_mode='categorical',
        shuffle=False # Important for evaluation
    )

    return train_generator, validation_generator, test_generator

def lr_schedule(epoch):
    """Learning rate scheduler for ResNet50."""
    lr = 1e-4
    if epoch > 15:
        lr *= 0.1
    return lr

if __name__ == "__main__":
    input_folder = 'D:/Yugam/College/3rd Year/6th Sem/Project/Eye_disease_prediction-main/dataset' # Replace with your dataset location
    output_folder = 'splitted_data_tuned_resnet50'
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.2, 0.1)) # Adjusted ratios

    train_dir = os.path.join(output_folder, 'train')
    val_dir = os.path.join(output_folder, 'val')
    test_dir = os.path.join(output_folder, 'test')

    num_classes = len(os.listdir(train_dir))
    input_shape = (224, 224, 3)
    learning_rate = 1e-4
    dropout_rate = 0.5
    epochs = 30
    batch_size = 32

    model = build_tuned_resnet50_model(input_shape, num_classes, learning_rate, dropout_rate)

    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir, val_dir, test_dir, image_size=input_shape[:2], batch_size=batch_size
    )

    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    trained_model = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[lr_scheduler, early_stopping]
    )

    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    model.save('eye_disease_resnet50_tuned.h5')
    print("Tuned ResNet50 model training and evaluation completed and saved.")