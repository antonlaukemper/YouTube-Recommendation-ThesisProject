from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import fasttext
import numpy as np
from tqdm import tqdm

from text_processing.load_data import cross_validate


def vary_epochs(data, metric, mode):
    epochs = [30, 35, 40, 45, 50, 65, 70]
    train_metrics = []
    test_metrics = []
    for epoch in tqdm(epochs):
        train_metric, test_metric = cross_validate(data, 5, metric, epoch=epoch)
        train_metrics.append(train_metric)
        test_metrics.append(test_metric)
    fig, ax = plt.subplots()
    plt.plot(epochs, train_metrics, label="training %s" % metric)
    plt.plot(epochs, test_metrics, label="testing %s" % metric)
    plt.legend()
    ax.set_xlabel('Number of epochs', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title('Varying number of epochs \n with FastText classification', fontsize=20)
    plt.show()
    print(test_metrics)
    import pickle
    with open(f"cross_validation_results/{mode}/cv_training.pl", "wb") as file:
        pickle.dump(train_metrics, file)
    with open(f"cross_validation_results/{mode}/cv_testing.pl", "wb") as file:
        pickle.dump(test_metrics, file)
    with open(f"cross_validation_results/{mode}/cv_epochs.pl", "wb") as file:
        pickle.dump(epochs, file)

def vary_learning_rate(data, metric):
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]
    train_metrics = []
    test_metrics = []
    for rate in tqdm(learning_rates):
        train_metric, test_metric = cross_validate(data, 5, metric, lr=rate)
        train_metrics.append(train_metric)
        test_metrics.append(test_metric)
    fig, ax = plt.subplots()
    plt.plot(learning_rates, train_metrics, label="Training F1-Score")
    plt.plot(learning_rates, test_metrics, label="Validation F1-Score")
    plt.legend()
    ax.set_xlabel('Learning Rate', fontsize=15)
    ax.set_ylabel("F1-Score", fontsize=15)
    plt.show()
    import pickle
    with open("cross_validation_results/captions/lr_training.pl", "wb") as file:
        pickle.dump(train_metrics, file)
    with open("cross_validation_results/captions/lr_testing.pl", "wb") as file:
        pickle.dump(test_metrics, file)
    with open("cross_validation_results/captions/lr_epochs.pl", "wb") as file:
        pickle.dump(learning_rates, file)


def vary_grams(data, metric):
    grams = [1, 2, 3, 4, 5]
    train_metrics = []
    test_metrics = []
    for gram in tqdm(grams):
        train_metric, test_metric = cross_validate(data, 5, metric, wordNgrams=gram)
        train_metrics.append(train_metric)
        test_metrics.append(test_metric)
    fig, ax = plt.subplots()
    plt.plot(grams, train_metrics, label="Training F1-Score")
    plt.plot(grams, test_metrics, label="Validation F1-Score")
    plt.legend()
    ax.set_xlabel('N-Grams', fontsize=15)
    ax.set_ylabel("F1-Score", fontsize=15)
    plt.show()
    import pickle
    with open("cross_validation_results/captions/gr_training.pl", "wb") as file:
        pickle.dump(train_metrics, file)
    with open("cross_validation_results/captions/gr_testing.pl", "wb") as file:
        pickle.dump(test_metrics, file)
    with open("cross_validation_results/captions/gr_epochs.pl", "wb") as file:
        pickle.dump(grams, file)