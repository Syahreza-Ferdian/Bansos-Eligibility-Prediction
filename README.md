## Bansos House Eligibility Prediction
> This project aims to predict the eligibility of houses for social assistance (Bansos) based on images of their exteriors. Using a Convolutional Neural Network (CNN) model, the system classifies houses as either "Eligible" or "Not Eligible" for receiving social assistance.

### Key Features:
- Image Classification: The model uses house images to determine eligibility for social assistance.
- Preprocessing & Data Augmentation: Includes image preprocessing and augmentation to improve model performance despite an imbalanced dataset.
- Flask Web Interface: A user-friendly web interface built with Flask, allowing users to upload images and get predictions.
- Customizable Threshold: Allows adjustment of the sensitivity for eligibility classification.

### Technologies Used:
- Python
- Keras / TensorFlow
- Flask
- NumPy
- HTML/CSS/JavaScript for front-end

### How to Use:
1. Clone the repository and install the required dependencies.
2. Train the model on your dataset (or use the pre-trained model).
3. Run the Flask app to deploy a simple web interface where users can upload house images for eligibility predictions.
