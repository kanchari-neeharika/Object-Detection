# \# Object Detection Using TensorFlow

# 

# A deep learning-based Object Detection system developed using \*\*TensorFlow\*\* and \*\*Keras\*\* for simultaneous handwritten digit classification and localization. The project uses the \*\*MNIST dataset\*\* and applies \*\*Bounding Box Regression\*\* to detect the position of digits on a larger canvas.

# 

# \---

# 

# \## Project Overview

# 

# Traditional image classification identifies \*\*what\*\* an object is, whereas object detection identifies \*\*what\*\* the object is \*\*and where\*\* it is located.

# 

# In this project, the standard MNIST handwritten digit dataset is converted into an object detection dataset by randomly placing each digit onto a \*\*75 × 75\*\* canvas. A Convolutional Neural Network (CNN) is then trained to perform two tasks simultaneously:

# 

# \- Digit Classification (0–9)

# \- Bounding Box Regression

# 

# \---

# 

# \## Features

# 

# \- Multi-output CNN Architecture

# \- Handwritten Digit Detection

# \- Bounding Box Prediction

# \- TensorFlow \& Keras Implementation

# \- Data Preprocessing Pipeline

# \- IoU (Intersection over Union) Evaluation

# \- Visualization of Predictions

# \- Implemented in Spyder IDE

# 

# \---

# 

# \## Dataset

# 

# \*\*Dataset:\*\* MNIST in CSV Format

# 

# The dataset contains handwritten digits (0–9) represented as pixel values.

# 

# Download the dataset from Kaggle:

# 

# https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

# 

# Required files:

# 

# \- `mnist\_train.csv`

# \- `mnist\_test.csv`

# 

# > \*\*Note:\*\* The training dataset (`mnist\_train.csv`) is not included in this repository because it exceeds GitHub's file size limit. Download it from the Kaggle link above and place it in the project directory before running the code.

# 

# \---

# 

# \## Project Structure

# 

# ```

# Object-Detection/

# │

# ├── object\_detection.py

# ├── requirements.txt

# ├── README.md

# ├── .gitignore

# ├── mnist\_test.csv

# ├── AI\_INT\_KANCHARINEEHARIKA.pdf

# └── screenshots/

# &#x20;   ├── accuracy.png

# &#x20;   ├── prediction.png

# &#x20;   └── model\_output.png

# ```

# 

# \---

# 

# \## Technologies Used

# 

# \- Python

# \- TensorFlow

# \- Keras

# \- NumPy

# \- Pandas

# \- Matplotlib

# \- Pillow

# \- Spyder IDE

# \- Anaconda

# 

# \---

# 

# \## Model Architecture

# 

# The network consists of:

# 

# \- Three Convolutional Layers

# \- Average Pooling Layers

# \- Dense Layer (128 Neurons)

# 

# The extracted features are divided into two outputs:

# 

# \### Classification Head

# 

# \- Softmax Activation

# \- Predicts digit classes (0–9)

# 

# \### Bounding Box Regression Head

# 

# \- Linear Activation

# \- Predicts normalized bounding box coordinates

# 

# \---

# 

# \## Evaluation Metrics

# 

# The model is evaluated using:

# 

# \- Classification Accuracy

# \- Mean Squared Error (MSE)

# \- Intersection over Union (IoU)

# 

# \---

# 

# \## Installation

# 

# Clone the repository:

# 

# ```bash

# git clone https://github.com/kanchari-neeharika/Object-Detection.git

# ```

# 

# Move into the project folder:

# 

# ```bash

# cd Object-Detection

# ```

# 

# Install the required packages:

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \---

# 

# \## Running the Project

# 

# Place the following files in the project folder:

# 

# \- mnist\_train.csv

# \- mnist\_test.csv

# 

# Run:

# 

# ```bash

# python object\_detection.py

# ```

# 

# \---

# 

# \## Output

# 

# The model generates:

# 

# \- Classification Accuracy

# \- Bounding Box MSE

# \- IoU Score

# \- Predicted Bounding Boxes

# \- Visualization of Ground Truth vs Predicted Boxes

# 

# \---

# 

# \## Results

# 

# The model successfully performs:

# 

# \- Handwritten digit classification

# \- Bounding box localization

# \- Simultaneous multi-task learning

# 

# The project demonstrates how object detection concepts can be implemented using a simple CNN before progressing to advanced detectors such as YOLO, SSD, and Faster R-CNN.

# 

# \---

# 

# \## Future Improvements

# 

# \- YOLOv8 Implementation

# \- SSD Detector

# \- Faster R-CNN

# \- Real-time Webcam Detection

# \- COCO Dataset Support

# \- Transfer Learning

# \- Data Augmentation

# 

# \---

# 

# \## Screenshots

# 

# \### Training Accuracy

# 

# \*Add your training accuracy graph here.\*

# 

# \### Bounding Box Prediction

# 

# \*Add prediction output image here.\*

# 

# \### Model Output

# 

# \*Add final output screenshot here.\*

# 

# \---

# 

# \## Author

# 

# \*\*Neeharika Kanchari\*\*

# 

# GitHub: https://github.com/kanchari-neeharika

# 

# \---

# 

# \## License

# 

# This project is intended for educational and academic purposes.

