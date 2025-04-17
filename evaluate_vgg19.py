import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import pandas as pd

def create_dataset(directory, batch_size=32, image_size=(224, 224)):
    """Creates a TensorFlow dataset from a directory of images."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False  # Important for consistent label order
    )
    return dataset

# Load the trained VGG19 model
trained_model = tf.keras.models.load_model('eye_disease_vgg19.h5')  # VGG19 model name.

# Create the test dataset for VGG19
test_dir = 'splitted_data_vgg19/test'  # VGG19 test directory
test_dataset = create_dataset(test_dir, image_size=(224, 224))  # VGG19 image size.

# Evaluate the VGG19 model
test_loss, test_accuracy = trained_model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Get predictions from VGG19
predictions = trained_model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Get true labels from VGG19 test dataset
true_labels = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)
true_labels = np.argmax(true_labels, axis=1)

# Get class names from VGG19 test dataset
class_names = test_dataset.class_names

# Classification Report
report_dict = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Save as CSV
report_df.to_csv('classification_report_vgg19.csv', index=True)

print("Classification report saved as 'classification_report_vgg19.csv'")

# Confusion matrix for VGG19
print("\nConfusion Matrix (VGG19):")
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (VGG19)')
plt.show()

# Display some predictions from VGG19
plt.figure(figsize=(10, 10))
for images, labels in test_dataset.take(1):
    for i in range(min(9, images.shape[0])):  # Handle cases with fewer than 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"True: {class_names[true_labels[i]]}, Predicted: {class_names[predicted_labels[i]]}")
        plt.axis("off")
plt.show()

# # Example for checking individual image with VGG19.
# from tensorflow.keras.preprocessing import image
# img_path = 'splitted_data_vgg19/test/glaucoma/_167_1325871.jpg'  # Replace with your image path.
# img = image.load_img(img_path, target_size=(224, 224))  # VGG19 image size.
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0) / 255.0

# prediction = trained_model.predict(img_array)
# predicted_class = class_names[np.argmax(prediction)]
# print(f"Predicted Class (VGG19): {predicted_class}")