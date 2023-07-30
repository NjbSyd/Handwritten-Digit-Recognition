# Handwritten Digit Recognition

This repository contains a Python implementation of a digit recognition model using logistic regression. The model is trained on the [sklearn.datasets.load_digits()](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset. The code uses the Scikit-learn library for loading the dataset, train-test splitting, and preprocessing. The trained model is saved to a file for later use.

## Overview
The `DigitRecognitionViaRegression` class provides an implementation of a logistic regression model for digit recognition. The model is trained on a subset of the digits dataset from Scikit-learn. The code uses a sigmoid activation function and cross-entropy loss function for training.

## Installation
1. Make sure you have Python (>= 3.6) installed.
2. Install the required libraries using pip:
   ```
   pip install numpy scikit-learn opencv-python matplotlib
   ```

## Usage
Follow these steps to train and use the digit recognition model:

1. Clone the repository or download the `digit_recognition.py` and `DigitRecognition.py` files.
2. Import the `DigitRecognitionViaRegression` class into your project.
3. Create an instance of the class with optional parameters for customization. Default parameters are provided.
4. The model will be trained on the dataset upon initialization. If a pre-trained model exists in the file system (stored as `trained_model.pkl`), it will be loaded automatically.

### Testing the Model
The provided code uses OpenCV and Matplotlib to test the digit recognition model using custom images. Before running the testing script, ensure you have installed the necessary libraries (`opencv-python` and `matplotlib`). The testing script allows you to select an image of a digit, and the model will predict the digit's value.

1. Import the required libraries at the beginning of your script:
   ```python
   import cv2
   from matplotlib import pyplot as plt
   import tkinter as tk
   from tkinter import filedialog
   from DigitRecognition import DigitRecognitionViaRegression
   ```

2. Create an instance of the `DigitRecognitionViaRegression` class with optional parameters, e.g.:
   ```python
   classifier = DigitRecognitionViaRegression(epochs=5000, learning_rate=0.01)
   ```

3. Use the following code to test the model with custom images:
   ```python
   def open_file_dialog():
       root = tk.Tk()
       root.withdraw()
       file_path = filedialog.askopenfilename()
       return file_path

   while True:
       option = input("Enter 1 to test, any to exit: ")
       if option == "1":
           gray = cv2.imread(open_file_dialog(), 0)
           gray = cv2.resize(255 - gray, (8, 8))
           g1 = gray.reshape(64, 1)
           prediction = classifier.predict(g1)

           plt.imshow(gray, cmap='gray')
           plt.title("Prediction: " + str(prediction))
           plt.axis('off')
           plt.show()
       else:
           exit(0)
   ```

4. Run your script and follow the instructions to test the model using your own digit images.

## Model Details
The logistic regression model is trained using the stochastic gradient descent (SGD) algorithm. The key parameters of the model are as follows:
- `num_classes`: Number of classes (default: 10).
- `num_features`: Number of features (default: 64).
- `learning_rate`: Learning rate for SGD (default: 0.1).
- `epochs`: Number of training epochs (default: 100).

### Training Process
1. The digits dataset is loaded from Scikit-learn.
2. The data is split into training and testing sets, with 75% used for training and 25% for testing.
3. Labels are one-hot encoded to match the model's output format.
4. The model is trained using SGD with cross-entropy loss and sigmoid activation.
5. The trained model's parameters are saved to a file (`trained_model.pkl`) for future use.

### Prediction
To predict the digit in a given input `x`, the following steps are performed:
1. The input is reshaped to match the number of features used during training.
2. The dot product of the model's weights and the input is computed.
3. The sigmoid function is applied to the dot product to get the predicted probabilities for each class.
4. The class with the highest probability is returned as the predicted digit.

## License
This project is licensed under the MIT License.

**Note:** This code is intended for educational purposes and may not be suitable for production use without further optimizations and testing.
