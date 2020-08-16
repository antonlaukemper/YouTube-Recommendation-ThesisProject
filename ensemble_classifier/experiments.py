from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


def cross_validate(data, labels, n, metric, **kwargs):
    kf = KFold(n_splits=n)
    train_metric = []
    test_metric = []

    for train_index, test_index in kf.split(data, labels):
        x_train, x_test = data.iloc[train_index, :], data.iloc[test_index, :]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        trained_model = LogisticRegression(random_state=0, solver='liblinear', max_iter=1000, **kwargs).fit(x_train, y_train)
        train_error = calculate_metrics(trained_model, x_train, y_train)['weighted avg'][metric]
        test_error = calculate_metrics(trained_model, x_test, y_test)['weighted avg'][metric]
        train_metric.append(train_error)
        test_metric.append(test_error)
    print(test_metric)
    print("mean %s: %f" % (metric, np.mean(test_metric)))
    return (np.mean(train_metric), np.mean(test_metric))

def calculate_metrics(model, data, ground_truth):
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict([data.iloc[i]]))
    return classification_report(ground_truth, predicted, output_dict=True)


def vary_epochs(data, labels, metric):
    epochs = [50, 100, 150, 200]
    train_metrics = []
    test_metrics = []
    for epoch in tqdm(epochs):
        train_metric, test_metric = cross_validate(data, labels, 5, metric, max_iter=epoch)
        train_metrics.append(train_metric)
        test_metrics.append(test_metric)
    fig, ax = plt.subplots()
    plt.plot(epochs, train_metrics, label="training %s" % metric)
    plt.plot(epochs, test_metrics, label="testing %s" % metric)
    plt.legend()
    ax.set_xlabel('Number of iterations', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title('Varying number of iterations \n with Logistic Regression \n on Affiliation data', fontsize=20)
    plt.show()

def vary_c(data, labels, metric):
    cs = [0.1, 0.5, 1, 5, 10, 15, 20, 30, 40, 100, 200]
    train_metrics = []
    test_metrics = []
    for c in tqdm(cs):
        train_metric, test_metric = cross_validate(data, labels, 5, metric, C=c)
        train_metrics.append(train_metric)
        test_metrics.append(test_metric)
    fig, ax = plt.subplots()
    plt.plot(cs, train_metrics, label="training %s" % metric)
    plt.plot(cs, test_metrics, label="testing %s" % metric)
    plt.legend()
    ax.set_xlabel('Inverse Regularization Strength', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title('Varying regularization parameter \n with Logistic Regression \n on Ensemble Data', fontsize=18)
    plt.show()
    print("Mean test metrics: "+ str(test_metrics))