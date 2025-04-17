import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import os

def create_dataset(directory, batch_size=32, image_size=(150, 150)):
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

# Load the trained model
trained_model = tf.keras.models.load_model('eye_disease_conv2d.h5') #or resnet model name.

# Create the test dataset
test_dir = 'splitted_data_conv2d/test' #or resnet path
test_dataset = create_dataset(test_dir, image_size=(150, 150)) #or 224,224 for resnet.

# Evaluate the model
test_loss, test_accuracy = trained_model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Get predictions
predictions = trained_model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Get true labels from test dataset
true_labels = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)
true_labels = np.argmax(true_labels, axis=1)

# Get class names
class_names = test_dataset.class_names

# Classification Report
report_dict = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Save as CSV
report_df.to_csv('classification_report_conv2D.csv', index=True)

print("Classification report saved as 'classification_report_conv2D.csv'")

# Display some predictions
plt.figure(figsize=(10, 10))
for images, labels in test_dataset.take(1):
    for i in range(min(9, images.shape[0])):  # Handle cases with fewer than 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"True: {class_names[true_labels[i]]}, Predicted: {class_names[predicted_labels[i]]}")
        plt.axis("off")
plt.show()

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# # Example for checking individual image.
# from tensorflow.keras.preprocessing import image

# img_path = 'splitted_data_conv2d/test/diabetic_retinopathy/10095_right.jpeg' #replace with your image path.
# img = image.load_img(img_path, target_size=(150, 150)) #or 224,224 if using resnet
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0) / 255.0

# prediction = trained_model.predict(img_array)
# predicted_class = class_names[np.argmax(prediction)]
# print(f"Predicted Class: {predicted_class}")