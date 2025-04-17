import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_tuned_vgg19_model(model_path, test_dir, image_size=(224, 224), batch_size=32):
    """Evaluates the tuned VGG19 model."""

    # Load the trained model
    trained_model = load_model(model_path)
    print(f"Loaded model from: {model_path}")

    # Create a data generator for the test set (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Important to keep order for true labels
    )

    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = trained_model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Get predictions
    print("Generating predictions...")
    predictions = trained_model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true labels
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification Report
    report_dict = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)

    # Convert to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    # Save as CSV
    report_df.to_csv('classification_report__tuned_vgg19.csv', index=True)

    print("Classification report saved as 'classification_report_tuned_vgg19.csv'")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Tuned VGG19)')
    plt.show()

if __name__ == "__main__":
    model_path = 'eye_disease_vgg19_tuned.h5'  # Replace with the path to your saved tuned VGG19 model
    output_folder = 'splitted_data_tuned_vgg19' # Replace with your output folder
    test_dir = os.path.join(output_folder, 'test')
    image_size = (224, 224)
    batch_size = 32

    evaluate_tuned_vgg19_model(model_path, test_dir, image_size, batch_size)
    print("Evaluation of the tuned VGG19 model completed.")