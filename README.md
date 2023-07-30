# Digit Recognition via Regression 🕵️‍♂️🔢

Welcome to the **Digit Recognition via Regression** repository! Here, we've got a cool Python implementation of a digit recognition model using logistic regression. Our little AI agent can predict those mysterious digits from the famous [sklearn.datasets.load_digits()](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset. But that's not all, it's packed with Sci-Fi-like abilities!

## Installation 🚀

To join the futuristic party, make sure you've got Python (version 3.6 and above) installed on your spaceship 🚀. Once you're all set, equip your terminal with the necessary powers:
```
pip install numpy scikit-learn opencv-python matplotlib
```

## Usage 🤖

With the power of the **DigitRecognitionViaRegression** class, you can unleash the full potential of our AI agent! 😎 

1. Clone this repository or download the `digit_recognition.py` and `DigitRecognition.py` files and place them in your coding laboratory 🧪.

2. Import the `DigitRecognitionViaRegression` class into your top-secret project. 😏

3. Construct an instance of the class, and the magic begins! You can customize the AI's training by setting optional parameters like `epochs` and `learning_rate`. Don't worry; it comes with default parameters too! 😌

4. The AI model will initiate the top-secret training protocol upon its awakening 🌟. If it has a pre-trained model (stored as `trained_model.pkl`) from a past adventure, it'll load it to kickstart the mission! 🚀

### Testing the Model 🚀

Here comes the fun part! Our AI is not just about training; it can also predict unknown digits with its all-seeing eye! 👁️

1. Equip your code with the necessary powers by importing these libraries:
   ```python
   import cv2
   from matplotlib import pyplot as plt
   import tkinter as tk
   from tkinter import filedialog
   from DigitRecognition import DigitRecognitionViaRegression
   ```

2. Construct an instance of the `DigitRecognitionViaRegression` class with optional parameters, so it's fully prepared for the adventure! 🛡️

3. Time for the exciting part! You can now use the custom code below to test the AI model with your own digit images! Just run the code and watch the AI in action as it predicts the digit from your mysterious image! 🎉
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

## Model Details 🤓

Our AI is powered by a robust logistic regression model that can handle the complexity of digit recognition! It uses the stochastic gradient descent (SGD) algorithm for training, and it's packed with these cool parameters:
- `num_classes`: Number of classes (default: 10).
- `num_features`: Number of features (default: 64).
- `learning_rate`: Learning rate for SGD (default: 0.1).
- `epochs`: Number of training epochs (default: 100).

### Training Process 🎓

The AI undergoes rigorous training using the digits dataset from Scikit-learn. This data is then split into training and testing sets (75% for training, 25% for testing). The AI decodes the labels using one-hot encoding to align with its output format. Armed with a cross-entropy loss function and sigmoid activation, the AI trains like a champ using the SGD algorithm! 🏋️‍♂️ The learned parameters of the AI are then safely stored in a file (`trained_model.pkl`) for future adventures!

### Prediction 🚀

When you request the AI to predict a digit in your mysterious input `x`, it deploys its superpowers like a true superhero! 🦸‍♂️ It resizes and preprocesses the input to match the features used during training. By using its super intelligence, the AI performs a dot product with its weight, applies the sigmoid function, and BOOM 💥 it predicts the class with the highest probability as the digit you were looking for!

## License 📜

Our AI project is protected under the MIT License! Feel free to join our futuristic endeavors and contribute to the code if you'd like! 🤝

**Disclaimer:** This AI code is all about having fun and exploring the marvels of digit recognition. Be cautious if you're planning to use it for critical missions! 🔒

Let's embark on this incredible journey into the world of digit recognition and unleash the power of AI! 🌌🤖
