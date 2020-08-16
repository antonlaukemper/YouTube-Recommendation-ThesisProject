import fasttext
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.data_IO import write_to_file
from utils.metrics import calculate_multilabel_metrics_text


def vary_epochs(data, metric):
    epochs = [50, 100, 200, 400, 500]
    train_metrics = []
    test_metrics = []
    test_MAPS = []
    for epoch in tqdm(epochs):
        train_metric, test_metric, test_map = cross_validate(data, 5, metric, epoch=epoch)
        train_metrics.append(train_metric)
        test_metrics.append(test_metric)
        test_MAPS.append(test_map)
    fig, ax = plt.subplots()
    plt.plot(epochs, train_metrics, label="training %s" % metric)
    plt.plot(epochs, test_metrics, label="testing %s" % metric)
    plt.plot(epochs, test_MAPS, label="testing MAP")
    plt.legend()
    ax.set_xlabel('Number of epochs', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title('Varying number of epochs \n with FastText classification', fontsize=20)
    plt.show()
    print(test_metrics)


def cross_validate(data, n, metric, **kwargs):
    kf = KFold(n_splits=n)
    train_metric = []
    test_metric = []
    test_MAPs = []

    for train_index, test_index in kf.split(data):
        train, test = data.iloc[train_index, :], data.iloc[test_index, :]
        write_to_file(train, "fasttext_data/captions/training_data.txt")
        write_to_file(test, "fasttext_data/captions/testing_data.txt")
        model = fasttext.train_supervised(input="fasttext_data/captions/training_data.txt", loss='ova', **kwargs)
        train_metric_result = calculate_multilabel_metrics_text("fasttext_data/captions/training_data.txt", model,
                                                                metric=metric)
        test_metric_result = calculate_multilabel_metrics_text("fasttext_data/captions/testing_data.txt", model,
                                                               metric=metric)
        test_map = calculate_multilabel_metrics_text("fasttext_data/captions/testing_data.txt", model,
                                                               metric="MAP")
        train_metric.append(train_metric_result)
        test_metric.append(test_metric_result)
        test_MAPs.append(test_map)
    print(test_metric)
    print("mean MAP: %f" % np.mean(test_MAPs))
    print("mean %s: %f" % (metric, np.mean(test_metric)))
    return np.mean(train_metric), np.mean(test_metric), np.mean(test_MAPs)
