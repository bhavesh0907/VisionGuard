Automated Ophthalmic Disease Detection

A Comparative Study of CNN Architectures with Emphasis on Hyperparameter Sensitivity and Class Imbalance Challenges

📌 Project Overview

This project presents a deep learning-based solution for the automated detection and classification of ophthalmic diseases from retinal fundus images.
It compares three CNN architectures — Conv2D, ResNet50, and VGG19 — under standard and hyperparameter-tuned settings.
The goal is to identify challenges such as class imbalance, hyperparameter sensitivity, and to determine the most reliable architecture for deployment in healthcare settings.

🧩 Key Features

Automated classification of Diabetic Retinopathy, Glaucoma, Cataract, and other retinal diseases.
Comparison between standard and hyperparameter-tuned versions of Conv2D, ResNet50, and VGG19.
Implementation of robust preprocessing (normalization, augmentation, resizing).
Error analysis through confusion matrices, classification reports, and performance curves.
Focus on ethical AI usage and considerations of clinical deployment challenges.
🗂️ Project Structure

├── dataset/                  # ODIR-5K dataset (after preprocessing)
├── models/
│   ├── conv2d_standard.py
│   ├── conv2d_tuned.py
│   ├── resnet50_standard.py
│   ├── resnet50_tuned.py
│   ├── vgg19_standard.py
│   └── vgg19_tuned.py
├── preprocessing/
│   ├── data_loader.py
│   └── augmentation.py
├── evaluation/
│   ├── classification_report.py
│   ├── confusion_matrix.py
│   └── performance_plots.py
├── utils/
│   └── config.py
├── README.md
├── requirements.txt
└── main.py                   # Master controller to train and evaluate models
⚙️ Technologies Used

Python 3.8+
TensorFlow / Keras
OpenCV
NumPy
Matplotlib & Seaborn
Scikit-learn
Splitfolders
📚 Dataset

ODIR-5K Dataset: Retinal fundus images categorized into 8 diagnostic labels.
Preprocessing included image resizing to (150x150) or (224x224) depending on model, normalization, and augmentation techniques.
🛠️ Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/ophthalmic-disease-detection.git
cd ophthalmic-disease-detection
Install Dependencies:
pip install -r requirements.txt
Prepare Dataset:
Place the ODIR-5K dataset inside the dataset/ folder.
Run preprocessing script to split and augment data:
python preprocessing/data_loader.py
Train and Evaluate Models:
python main.py
Edit the main.py configuration to toggle between different models and settings.

📈 Results Summary


Model	Accuracy (Standard)	Accuracy (Tuned)
Conv2D	~78%	~81%
ResNet50	91.78%	41.92% (overfit)
VGG19	~88%	~90%
Best Model: Standard ResNet50 achieved highest reliability across classes.
Challenge: All models struggled with minority classes like Glaucoma due to class imbalance.
🛡 Ethical Considerations

Data privacy is respected (no patient identifiers used).
The project advocates transparency and fairness in AI-based medical diagnostics.
Clinical deployment would require real-world validation and clinician feedback integration.
🎯 Future Enhancements

Addressing class imbalance with SMOTE or GANs.
Implementing explainability techniques like Grad-CAM.
Expanding to multi-label and multi-task learning setups.
Real-time deployment with cloud and mobile support.
👥 Contributors

Bhavesh Mishra
Yugam Shah
📜 License

This project is intended for academic and research purposes only.
