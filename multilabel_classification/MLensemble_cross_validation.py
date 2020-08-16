from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd

from classifier.ensemble_classifier import EnsembleClassifier
from classifier.experiments import calculate_metrics
from multilabel_classification.multi_label_ensemble_classifier import MLEnsembleClassifier
from utils.metrics import calculate_multilabel_metrics_relation


def cross_validate(data, n, metric, **kwargs):
    kf = KFold(n_splits=n)
    train_metric = []
    test_metric = []

    for train_index, test_index in kf.split(data):
        ensemble = MLEnsembleClassifier(training_fraction=1)
        ensemble.random_state = 42
        ensemble.load_data()
        caption_validation_data = ensemble.caption_data[0].iloc[test_index, :]
        ensemble.caption_data[0] = ensemble.caption_data[0].iloc[train_index, :]
        comments_validation_data = ensemble.comment_data[0].iloc[test_index, :]
        ensemble.comment_data[0] = ensemble.comment_data[0].iloc[train_index, :]
        snippets_validation_data = ensemble.snippet_data[0].iloc[test_index, :]
        ensemble.snippet_data[0] = ensemble.snippet_data[0].iloc[train_index, :]
        affilitation_validation_data = ensemble.affiliation_data[0].iloc[test_index, :]
        ensemble.affiliation_data[0] = ensemble.affiliation_data[0].iloc[train_index, :]
        subscriptions_validation_data = ensemble.subscription_data[0].iloc[test_index, :]
        ensemble.subscription_data[0] = ensemble.subscription_data[0].iloc[train_index, :]
        cross_comment_validation_data = ensemble.cross_comment_data[0].iloc[test_index, :]
        ensemble.cross_comment_data[0] = ensemble.cross_comment_data[0].iloc[train_index, :]

        ensemble.fit()
        ensemble.add_unlabeled_data(caption_validation_data,
                                    comments_validation_data,
                                    snippets_validation_data,
                                    affilitation_validation_data,
                                    subscriptions_validation_data,
                                    cross_comment_validation_data)
        training_data = ensemble.ensemble_data_training
        validation_data = ensemble.estimate_probabilities(mode='Unlabeled')

        training_x = training_data.drop(columns=['id',
                                                 'title',
                                                 'label',
                                                 'captions',
                                                 'comments',
                                                 'snippets'])
        training_y = ensemble.mlb.transform(training_data['label'])

        validation_x = validation_data.drop(columns=['id',
                                                     'title',
                                                     'label',
                                                     'captions',
                                                     'comments',
                                                     'snippets'])

        validation_y = ensemble.mlb.transform(validation_data['label'])

        trained_model = OneVsRestClassifier(LogisticRegression(random_state=42,
                                                               solver='liblinear', max_iter=10000,
                                                               **kwargs)).fit(training_x, training_y)
        train_error = calculate_multilabel_metrics_relation(training_x, training_y, trained_model, metric='MAP')
        test_error = calculate_multilabel_metrics_relation(validation_x, validation_y, trained_model, metric='MAP')
        train_metric.append(train_error)
        test_metric.append(test_error)
    print(test_metric)
    print("mean %s: %f" % (metric, np.mean(test_metric)))
    return (np.mean(train_metric), np.mean(test_metric))


def vary_c(data, metric):
    cs = [0.01, 0.05, 0.1, 0.3, 0.5]  # 1, 5, 10, 15, 20, 30, 40, 100, 200]
    train_metrics = []
    test_metrics = []
    for c in tqdm(cs):
        train_metric, test_metric = cross_validate(data, 5, metric, C=c)
        train_metrics.append(train_metric)
        test_metrics.append(test_metric)
    fig, ax = plt.subplots()
    plt.plot(cs, train_metrics, label="training %s" % metric)
    plt.plot(cs, test_metrics, label="validation %s" % metric)
    plt.legend()
    ax.set_xlabel('Inverse Regularization Strength', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title('Varying regularization parameter \n with Logistic Regression \n on Ensemble Data', fontsize=18)
    plt.show()
    print("Mean test metrics: " + str(test_metrics))


if __name__ == "__main__":
    ensemble = MLEnsembleClassifier(training_fraction=1)
    data = ensemble.caption_trainer.get_data(random_state=42, binary=False, training_fraction=1)
    vary_c(data=data[0], metric='f1-score')
