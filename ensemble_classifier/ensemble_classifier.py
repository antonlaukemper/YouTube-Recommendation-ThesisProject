from random import randrange

import fasttext
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm

from ensemble_classifier import experiments, plot_data
from network_processing.relations_trainer import RelationsTrainer
from text_processing.text_trainer import TextTrainer


class EnsembleClassifier:

    def __init__(self, training_fraction=0.9):
        self.scaler = StandardScaler()
        self.caption_trainer = TextTrainer(mode='captions', epochs=45)
        self.comment_trainer = TextTrainer(mode='comments', epochs=45)
        self.snippet_trainer = TextTrainer(mode='snippets', epochs=40)
        self.affiliation_trainer = RelationsTrainer(mode='related_channels', regularization=200)
        self.subscription_trainer = RelationsTrainer(mode='subscriptions', regularization=0.3)
        self.cross_comment_trainer = RelationsTrainer(mode='cross_comments', regularization=1)
        self.random_state = randrange(0, 10000)
        self.imputer = KNNImputer(n_neighbors=3)
        self.training_fraction = training_fraction
        print(f"-- random state: {self.random_state} --")

    def load_data(self):
        # caption_training_data, caption_test_data

        self.caption_data = self.caption_trainer.get_data(random_state=self.random_state,
                                                          training_fraction=self.training_fraction)
        # comment_training_data, comment_test_data
        self.comment_data = self.comment_trainer.get_data(random_state=self.random_state,
                                                          training_fraction=self.training_fraction)
        # snippet_training_data, snippet_test_data
        self.snippet_data = self.snippet_trainer.get_data(random_state=self.random_state,
                                                          training_fraction=self.training_fraction)
        # a_training_x, a_training_y, a_test_x, a_test_y
        self.affiliation_data = self.affiliation_trainer.get_data(random_state=self.random_state,
                                                                  training_fraction=self.training_fraction)
        # s_training_x, s_training_y, s_test_x, s_test_y
        self.subscription_data = self.subscription_trainer.get_data(random_state=self.random_state,
                                                                    training_fraction=self.training_fraction)
        # c_training_x, c_training_y, c_test_x, c_test_y
        self.cross_comment_data = self.cross_comment_trainer.get_data(random_state=self.random_state,
                                                                      training_fraction=self.training_fraction)

    def fit(self, serialized=False):
        self.models = {}
        if serialized:
            self.models['captions'] = fasttext.load_model('models/captions/best_captions_model.bin')
            self.models['comments'] = fasttext.load_model('models/comments/best_comments_model.bin')
            self.models['snippets'] = fasttext.load_model('models/snippets/best_snippets_model.bin')
            with open('models/related_channels/best_model.pl', 'rb') as model_path:
                self.models['related_channels'] = pickle.load(model_path)
            with open('models/subscriptions/best_model.pl', 'rb') as model_path:
                self.models['subscriptions'] = pickle.load(model_path)
            with open('models/cross_comments/best_model.pl', 'rb') as model_path:
                self.models['cross_comments'] = pickle.load(model_path)
        else:
            self.models['captions'] = self.caption_trainer.get_model(self.caption_data[0])
            self.models['comments'] = self.comment_trainer.get_model(self.comment_data[0])
            self.models['snippets'] = self.snippet_trainer.get_model(self.snippet_data[0])
            self.models['related_channels'] = self.affiliation_trainer.get_model(self.affiliation_data[0])
            self.models['subscriptions'] = self.subscription_trainer.get_model(self.subscription_data[0])
            self.models['cross_comments'] = self.cross_comment_trainer.get_model(self.cross_comment_data[0])

        # predict all values with the subclassifiers
        self.ensemble_data_training = self.estimate_probabilities(mode='Training')

        # training the final layer on the data
        # null values are knn-imputed
        training_x = pd.DataFrame(self.scaler.fit_transform(self.imputer.fit_transform(
            self.ensemble_data_training[['captions_pred',
                                         'comments_pred',
                                         'snippets_pred',
                                         'related_channels_pred',
                                         'subscriptions_pred',
                                         'cross_comments_pred'
                                         ]])))

        training_y = self.ensemble_data_training['label']
        self.ensemble_model = LogisticRegression(random_state=0,
                                                 solver='liblinear',
                                                 max_iter=1000,
                                                 C=0.01).fit(training_x, training_y)
        self.weights = self.ensemble_model.coef_

    def estimate_probabilities(self, mode='Training'):
        if mode == 'Training':
            data_index = 0
        elif mode == 'Testing':
            data_index = 1
        elif mode == 'Unlabeled':
            data_index = 2

        caption_estimations = self._classify_text(data=self.caption_data[data_index], content='captions')
        comments_estimations = self._classify_text(data=self.comment_data[data_index], content='comments')
        snippets_estimations = self._classify_text(data=self.snippet_data[data_index], content='snippets')
        affiliation_estimations = self._classify_relations(data=self.affiliation_data[data_index],
                                                           content='related_channels')
        subscription_estimations = self._classify_relations(data=self.subscription_data[data_index],
                                                            content='subscriptions')
        cross_comment_estimations = self._classify_relations(data=self.cross_comment_data[data_index],
                                                             content='cross_comments')

        # takes the data and creates the dataframe for the final logistic regression
        relation_result = affiliation_estimations.join(subscription_estimations).join(
            cross_comment_estimations).reset_index()
        result = caption_estimations.set_index(['id', 'label']).join(
            comments_estimations.set_index(['id', 'label'])).join(
            snippets_estimations.set_index(['id', 'label'])).join(relation_result.set_index(['id', 'label']))
        result.reset_index(inplace=True)

        # if testing mode, we also save this data as a variable, to load it easier in evaluation
        if mode == 'Testing':
            self.ensemble_data_testing = result

        return result

    def predict(self, data):
        data = pd.DataFrame(self.scaler.transform(self.imputer.transform(data[['captions_pred',
                                                                               'comments_pred',
                                                                               'snippets_pred',
                                                                               'related_channels_pred',
                                                                               'subscriptions_pred',
                                                                               'cross_comments_pred'
                                                                               ]])))

        labels = self.ensemble_model.predict(data)
        confidence_scores = self.ensemble_model.decision_function(data)
        probabilities = self.ensemble_model.predict_proba(data)[:, 1]
        return labels, confidence_scores, probabilities

    def evaluate(self, roc=False):
        training_x = pd.DataFrame(
            self.scaler.transform(self.imputer.transform(self.ensemble_data_training[['captions_pred',
                                                                                      'comments_pred',
                                                                                      'snippets_pred',
                                                                                      'related_channels_pred',
                                                                                      'subscriptions_pred',
                                                                                      'cross_comments_pred'
                                                                                      ]])))
        training_y = self.ensemble_data_training['label']

        ensemble_data_testing = self.estimate_probabilities(mode='Testing')
        testing_x = pd.DataFrame(self.scaler.fit_transform(self.imputer.transform(
            ensemble_data_testing[['captions_pred',
                                   'comments_pred',
                                   'snippets_pred',
                                   'related_channels_pred',
                                   'subscriptions_pred',
                                   'cross_comments_pred'
                                   ]])))
        testing_y = ensemble_data_testing['label']

        training_results = experiments.calculate_metrics(self.ensemble_model, training_x, training_y)
        # print(['weighted avg']['f1-score'])
        # print("testing f1")
        test_results = experiments.calculate_metrics(self.ensemble_model, testing_x, testing_y)

        if roc:
            plot_data.plot_roc(self.ensemble_model, testing_x, testing_y)

        return training_results, test_results

    def add_unlabeled_data(self, cap, com, snip, aff, subs, cross):
        # check whether unlabeled data has been added before
        if len(self.caption_data) == 2:
            # if not, it is appended to the respective data tuple
            self.caption_data.append(cap)
            self.comment_data.append(com)
            self.snippet_data.append(snip)
            self.affiliation_data.append(aff)
            self.subscription_data.append(subs)
            self.cross_comment_data.append(cross)
        else:
            # if it is, we override the data
            self.caption_data[2] = cap
            self.comment_data[2] = com
            self.snippet_data[2] = snip
            self.affiliation_data[2] = aff
            self.subscription_data[2] = subs
            self.cross_comment_data[2] = cross

    def _classify_text(self, data, content):
        row_dict_list = []
        for index, row in data.iterrows():
            if row['text'] == '':
                prob_political = None
            else:
                predicted_label, probability = self.models[content].predict(row['text'])
                prob_political = probability[0] if predicted_label[0] == '__label__political' else 1 - probability[0]
            row_dict = {'id': row['id'], 'label': row['label'], content: row['text'], content + '_pred': prob_political}
            row_dict_list.append(row_dict)
        result = pd.DataFrame(row_dict_list)
        return result

    def _classify_relations(self, data, content='related_channels'):
        row_dict_list = []
        for index, row in data.iterrows():
            id = row['id']
            title = row['title']
            label = row['label']
            row_data = row.drop('label').drop('id').drop('title')
            prob_political = self.models[content].decision_function([row_data])[0]
            row_dict = {'id': id,
                        'title': title,
                        'label': label,
                        content + '_pred': prob_political}
            row_dict_list.append(row_dict)
        results = pd.DataFrame(row_dict_list).set_index(['id', 'title', 'label'], inplace=False)
        return results



if __name__ == '__main__':

    ensemble = EnsembleClassifier(training_fraction=1)
    ensemble.load_data()
    ensemble.fit(serialized=False)
    ensemble.estimate_probabilities()
    training_x = pd.DataFrame(
        ensemble.scaler.transform(ensemble.imputer.transform(ensemble.ensemble_data_training[['captions_pred',
                                                                                              'comments_pred',
                                                                                              'snippets_pred',
                                                                                              'related_channels_pred',
                                                                                              'subscriptions_pred',
                                                                                              'cross_comments_pred'
                                                                                              ]])))
    training_y = ensemble.ensemble_data_training['label']
    plot(training_x, training_y, mode='Training', features='Text')
    ensemble.estimate_probabilities(mode='Testing')
    testing_x = pd.DataFrame(
        ensemble.scaler.transform(ensemble.imputer.transform(ensemble.ensemble_data_testing[['captions_pred',
                                                                                              'comments_pred',
                                                                                              'snippets_pred',
                                                                                              'related_channels_pred',
                                                                                              'subscriptions_pred',
                                                                                              'cross_comments_pred'
                                                                                              ]])))
    testing_y = ensemble.ensemble_data_testing['label']
    plot(testing_x, testing_y, mode='Testing', features='Text')
    training_result, test_result = ensemble.evaluate(roc=True)
    print("full training data")
    print(training_result, test_result)

    trials = 10

    avg_f1 = []
    for i in tqdm(range(trials)):
        ensemble = EnsembleClassifier()
        ensemble.load_data()
        ensemble.fit(serialized=False)
        # training_data = ensemble.estimate_probabilities(mode='Training')
        ensemble.estimate_probabilities(mode='Testing')
        training_result, test_result = ensemble.evaluate(roc=False)
        print(training_result, test_result)
        avg_f1.append(test_result['weighted avg']['f1-score'])
    print("##### Final f1 scores #### ")
    print(avg_f1)

    with open("data_pickles/results/ensemble_final_scores_lowerRegularization.pl", 'wb') as f:
        pickle.dump(avg_f1, f)

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    plt.scatter(range(0, trials), avg_f1)
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('f1-score', fontsize=15)
    ax.set_title(f'Training {trials} different Ensemble classifiers on 90 % of the data', fontsize=20)
    plt.show()
