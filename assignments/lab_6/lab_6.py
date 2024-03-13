"""
pip install -U scikit-learn
pip install --upgrade keras
pip install tensorflow[and-cuda]
"""

import keras

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split


def task_1():

    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and test data (60% training, 40% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Create an SVM model
    model = svm.SVC()

    # Train the SVM model
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)

    # Predict the first and second samples and compare with ground truth
    predictions = model.predict(X[:2])
    ground_truths = y[:2]

    print(score, predictions, ground_truths)


# task_1()
