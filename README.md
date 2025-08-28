# Deep Learning-Based Pneumonia Detection using Chest X-Rays

This project is a web application that uses a deep learning model to detect pneumonia from chest X-ray images. Users can upload an image, and the model will predict whether the X-ray shows signs of pneumonia or is normal.

## Features
- **Image-based Prediction:** Upload a chest X-ray image to get an instant diagnosis.
- **Simple Web Interface:** Easy-to-use interface built with Flask.
- **Deep Learning Model:** Utilizes a pre-trained Keras model for accurate predictions.

## Technologies Used
- **Backend:** Python, Flask
- **Deep Learning:** TensorFlow, Keras
- **Frontend:** HTML, CSS, JavaScript
- **Data Handling:** NumPy, Pillow

## Project Structure
```
.
├── Flask Application/
│   ├── app.py                # Main Flask application
│   ├── static/                 # CSS and JS files
│   └── templates/              # HTML templates
├── model_weights/
│   └── chest_xray.h5           # Pre-trained Keras model
├── chestxrays.ipynb            # Jupyter notebook for model exploration/training
└── README.md
```

## Setup and Installation

Follow these steps to get the project running on your local machine.

**1. Clone the repository:**
```bash
git clone https://github.com/AnshulJagotra/Deep--Learning-Based-Pneumonia-Detection-using-Chest-X-Rays.git
cd Deep--Learning-Based-Pneumonia-Detection-using-Chest-X-Rays
```

**2. Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install tensorflow keras flask numpy pillow
```

**3. Run the application:**
```bash
python "Flask Application/app.py"
```
The application will be available at `http://127.0.0.1:5000`.

## How to Use
1. Open your web browser and navigate to `http://127.0.0.1:5000`.
2. Click on the "Choose File" button to select a chest X-ray image from your computer.
3. Click the "Predict" button.
4. The application will display the prediction result: "Pneumonia" or "Normal".
