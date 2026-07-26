# Object Detection Using TensorFlow

## Overview

This project implements an object detection model using TensorFlow and Keras to perform both handwritten digit classification and object localization. The model is trained on the MNIST dataset by placing each digit on a 75×75 canvas and predicting its class along with its bounding box coordinates.

## Features

- Handwritten digit classification
- Bounding box prediction
- Multi-output CNN architecture
- TensorFlow and Keras implementation
- IoU (Intersection over Union) evaluation
- Visualization of predictions

## Dataset

This project uses the MNIST in CSV dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

Required files:

- mnist_train.csv
- mnist_test.csv

Note: The `mnist_train.csv` file is not included in this repository because it exceeds GitHub's file size limit. Download it from Kaggle and place it in the project folder before running the project.

## Project Structure

```text
Object-Detection/
├── object_detection.py
├── requirements.txt
├── README.md
├── .gitignore
├── mnist_test.csv
├── AI_INT_KANCHARINEEHARIKA.pdf
└── screenshots/
    ├── training_accuracy.png
    ├── prediction_output.png
    └── model_output.png
```

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Pillow
- Spyder IDE
- Anaconda

## Model Architecture

The model consists of:

- Three convolutional layers
- Average pooling layers
- Dense layer with 128 neurons

The network has two output heads:

### Classification Head

- Softmax activation
- Predicts digit classes from 0 to 9

### Bounding Box Regression Head

- Linear activation
- Predicts normalized bounding box coordinates

## Evaluation Metrics

- Classification Accuracy
- Mean Squared Error (MSE)
- Intersection over Union (IoU)

## Installation

Clone the repository:

```bash
git clone https://github.com/kanchari-neeharika/Object-Detection.git
```

Move into the project folder:

```bash
cd Object-Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Download the MNIST dataset from Kaggle.
2. Place `mnist_train.csv` and `mnist_test.csv` in the project folder.
3. Run the program:

```bash
python object_detection.py
```

## Output

The model produces:

- Classification Accuracy
- Bounding Box MSE
- IoU Score
- Predicted Bounding Boxes
- Visualization of prediction results

## Future Improvements

- YOLO implementation
- SSD object detector
- Faster R-CNN
- Real-time webcam detection
- Transfer learning
- Data augmentation

## Screenshots

Add screenshots of:

- Training accuracy graph
- Prediction results
- Final model output

## Author

Neeharika Kanchari

GitHub: https://github.com/kanchari-neeharika

## License

This project is intended for educational and academic purposes.
