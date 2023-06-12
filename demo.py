import csv
from random import randrange
from naive_bayes_classifier import NaiveBayesClassifier


if __name__ == '__main__':
    # load iris dataset
    X = []
    y = []
    with open('iris.csv') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            y.append(row[-1])
            X.append([float(x) for x in row[:-1]])
    
    # split into trainset (80%) and testset (20%)
    X_test = []
    y_test = []
    for _ in range(int(0.2 * len(y))):
        rand_index = randrange(len(y))
        X_test.append(X[rand_index])
        y_test.append(y[rand_index])
        del X[rand_index]
        del y[rand_index]
    
    # fit the naive bayes model
    nb_model = NaiveBayesClassifier()
    nb_model.fit(X, y)

    # test the model
    y_pred = nb_model.predict(X_test)
    correct = 0
    for pred, true in zip(y_pred, y_test):
        if pred == true:
            correct += 1
    print(f'Test Accuracy: {correct / len(y_test):.2f}')
