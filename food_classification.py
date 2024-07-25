# food_classification.py

import os
from sklearn.feature_extraction import image
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images):
    processed_images = []
    for img in images:
        img = img.resize((64, 64))
        img_array = np.array(img).flatten()
        processed_images.append(img_array)
    return np.array(processed_images)

def main():
    train_pizza = load_images_from_folder('C:\\Users\\sachin\\PRODIGY_ML_05\\train\\pizzas')
    train_burger = load_images_from_folder('C:\\Users\\sachin\\PRODIGY_ML_05\\train\\burgers')
    test_pizza = load_images_from_folder('C:\\Users\\sachin\\PRODIGY_ML_05\\test\\pizzas')
    test_burger = load_images_from_folder('C:\\Users\\sachin\\PRODIGY_ML_05\\test\\burgers')

    X_train_pizza = preprocess_images(train_pizza)
    X_train_burger = preprocess_images(train_burger)
    X_test_pizza = preprocess_images(test_pizza)
    X_test_burger = preprocess_images(test_burger)

    X_train = np.vstack([X_train_pizza, X_train_burger])
    X_test = np.vstack([X_test_pizza, X_test_burger])

    y_train = np.array([0]*len(X_train_pizza) + [1]*len(X_train_burger))
    y_test = np.array([0]*len(X_test_pizza) + [1]*len(X_test_burger))

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Plotting a few images from test set
    for i in range(5):
        plt.imshow(X_test[i].reshape(64, 64, 3))
        plt.title('Test Image')
        plt.show()

if __name__ == "__main__":
    main()
