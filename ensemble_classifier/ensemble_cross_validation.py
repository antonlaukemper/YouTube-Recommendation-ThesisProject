from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import pickle as pl

from classifier.ensemble_classifier import EnsembleClassifier
from classifier.experiments import calculate_metrics


def cross_validate(data, n, metric, **kwargs):
    kf = KFold(n_splits=n)
    train_metric = []
    test_metric = []

    for train_index, test_index in kf.split(data):
        ensemble = EnsembleClassifier(training_fraction=1)
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

        training_x = pd.DataFrame(ensemble.scaler.fit_transform(ensemble.imputer.fit_transform(
            training_data[['captions_pred',
                           'comments_pred',
                           'snippets_pred',
                           'related_channels_pred',
                           'subscriptions_pred',
                           'cross_comments_pred'
                           ]])))

        training_y = training_data['label']

        validation_x = pd.DataFrame(ensemble.scaler.fit_transform(ensemble.imputer.fit_transform(
            validation_data[['captions_pred',
                                             'comments_pred',
                                             'snippets_pred',
                                             'related_channels_pred',
                                             'subscriptions_pred',
                                             'cross_comments_pred'
                                             ]])))

        validation_y = validation_data['label']

        trained_model = LogisticRegression(random_state=0, solver='liblinear', max_iter=1000, **kwargs).fit(training_x,
                                                                                                            training_y)
        train_error = calculate_metrics(trained_model, training_x, training_y)['weighted avg'][metric]
        test_error = calculate_metrics(trained_model, validation_x, validation_y)['weighted avg'][metric]
        train_metric.append(train_error)
        test_metric.append(test_error)
    print(test_metric)
    print("mean %s: %f" % (metric, np.mean(test_metric)))
    return (np.mean(train_metric), np.mean(test_metric))


def vary_c(data, metric):
    cs = [0.01, 0.1, 0.5, 1, 5, 10, 20, 40, 100, 200]
    train_metrics = []
    test_metrics = []
    for c in tqdm(cs):
        train_metric, test_metric = cross_validate(data, 5, metric, C=c)
        train_metrics.append(train_metric)
        test_metrics.append(test_metric)
    fig, ax = plt.subplots()
    plt.plot(cs, train_metrics, label="Training F1-Score")
    plt.plot(cs, test_metrics, label="Validation F1-Score")
    plt.legend()
    ax.set_xlabel('Inverse Regularization Strength', fontsize=15)
    ax.set_ylabel("F1-Score", fontsize=15)
    plt.show()
    print("Mean test metrics: " + str(test_metrics))
    import pickle
    with open("data_pickles/cross_validation/training.pl", "wb") as file:
        pickle.dump(train_metrics, file)
    with open("cross_validation_results/captions/testing.pl", "wb") as file:
        pickle.dump(test_metrics, file)
    with open("cross_validation_results/captions/cs.pl", "wb") as file:
        pickle.dump(cs, file)

def plot_cv_results():
    with open('cross_validation_results/captions/epochs.pl', 'rb') as file:
        epochs = pl.load(file)
    with open('cross_validation_results/captions/cv_training.pl', 'rb') as file:
        training = pl.load(file)
    with open('cross_validation_results/captions/cv_testing.pl', 'rb') as file:
        validation = pl.load(file)

if __name__ == "__main__":
    ensemble = EnsembleClassifier(training_fraction=1)
    ensemble.load_data()
    data = ensemble.caption_trainer.get_data(random_state=42, training_fraction=1)
    vary_c(data=data[0], metric='f1-score')
