from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.metrics import calculate_multilabel_metrics_relation
from sklearn.preprocessing import MultiLabelBinarizer


def cross_validate(data, n, metric, **kwargs):
    kf = KFold(n_splits=n)
    train_metric = []
    test_metric = []
    mlb = MultiLabelBinarizer()
    labels = data['label']
    data = data.drop(columns=['label', 'id', 'title'])

    for train_index, test_index in kf.split(data, labels):
        x_train, x_test = data.iloc[train_index, :], data.iloc[test_index, :]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        mlb.fit(y_train.append(y_test))
        y_train = mlb.transform(y_train)
        y_test = mlb.transform(y_test)

        trained_model = OneVsRestClassifier(LogisticRegression(random_state=42,
                                                               solver='lbfgs', max_iter=1000,
                                                               **kwargs)).fit(x_train, y_train)
        train_error = calculate_multilabel_metrics_relation(x_train, y_train, trained_model, metric=metric)
        test_error = calculate_multilabel_metrics_relation(x_test, y_test, trained_model, metric=metric)
        train_metric.append(train_error)
        test_metric.append(test_error)
    print(test_metric)
    print("mean %s: %f" % (metric, np.mean(test_metric)))
    return (np.mean(train_metric), np.mean(test_metric))


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


def vary_c(data, metric):
    cs = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 10, 100, 200]
    train_metrics = []
    test_metrics = []
    for c in tqdm(cs):
        train_metric, test_metric = cross_validate(data, 5, metric, C=c)
        train_metrics.append(train_metric)
        test_metrics.append(test_metric)
    fig, ax = plt.subplots()
    plt.plot(cs, train_metrics, label="training %s" % metric)
    plt.plot(cs, test_metrics, label="testing %s" % metric)
    plt.legend()
    ax.set_xlabel('Inverse Regularization Strength', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title('Varying Regularization Parameter \n with Multi-Label Logistic Regression \n on Affiliation Data',
                 fontsize=17)
    plt.show()
    print(test_metrics)
