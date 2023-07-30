import cv2
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from DigitRecognition import DigitRecognitionViaRegression

classifier = DigitRecognitionViaRegression(epochs=5000, learning_rate=0.01)


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
