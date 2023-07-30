from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import os


class DigitRecognitionViaRegression:
    def __init__(self, num_classes=10, num_features=64, learning_rate=0.1, epochs=100):
        self.num_classes = num_classes
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.B = np.zeros([num_classes, num_features])
        self.model_file = "trained_model.pkl"

        if os.path.exists(self.model_file):
            self.load_model()
        else:
            self.train_model()

    def train_model(self):
        print("Loading digits dataset from sklearn")
        digits_data = load_digits()
        print("Splitting data into training and testing sets")
        digits = digits_data.data
        targets = digits_data.target

        x_train, _, y_train, _ = train_test_split(digits, targets, test_size=0.25)

        encode = OneHotEncoder(sparse_output=False)
        Y_encoded = encode.fit_transform(y_train.reshape(-1, 1))

        mean_loss = []
        for iteration in range(self.epochs):
            print("epochs: ", iteration)
            dB = np.zeros([self.num_classes, self.num_features])
            total_loss = 0

            for j in range(x_train.shape[0]):
                x1 = x_train[j, :].reshape(self.num_features, 1)
                y1 = Y_encoded[j, :].reshape(self.num_classes, 1)

                z1 = np.dot(self.B, x1)
                h = self.sigmoid(z1)

                db = (h - y1) * x1.T
                db = db.reshape(self.num_classes, self.num_features)
                dB += db
                total_loss += self.cost_function(h, y1)

            dB /= float(x_train.shape[0])
            total_loss /= float(x_train.shape[0])
            gradient = self.learning_rate * dB
            self.B -= gradient
            mean_loss.append(np.mean(total_loss))

        print("Final loss:", total_loss)
        self.save_model()

    def predict(self, x):
        x1 = x.reshape(self.num_features, 1)
        z1 = np.dot(self.B, x1)
        h = self.sigmoid(z1)
        return h.argmax()

    def save_model(self):
        with open(self.model_file, "wb") as f:
            pickle.dump(self.B, f)

    def load_model(self):
        with open(self.model_file, "rb") as f:
            self.B = pickle.load(f)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def cost_function(h, y):
        epsilon = 1e-5
        return -y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)

