from random import randrange

import fasttext
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt

from tqdm import tqdm

from classifier import plot_data
from network_processing.relations_trainer import RelationsTrainer
from text_processing.text_trainer import TextTrainer
from utils.metrics import calculate_multilabel_metrics_relation


class MLEnsembleClassifier:

    def __init__(self, training_fraction=0.9):
        self.scaler = StandardScaler()
        self.mlb = MultiLabelBinarizer()
        self.caption_trainer = TextTrainer(mode='captions', epochs=500)
        self.comment_trainer = TextTrainer(mode='comments', epochs=500)
        self.snippet_trainer = TextTrainer(mode='snippets', epochs=400)
        self.affiliation_trainer = RelationsTrainer(mode='related_channels', regularization=100)
        self.subscription_trainer = RelationsTrainer(mode='subscriptions', regularization=100)
        self.cross_comment_trainer = RelationsTrainer(mode='cross_comments', regularization=100)
        self.random_state = randrange(0, 10000)
        self.imputer = KNNImputer(n_neighbors=5)
        self.pca = PCA(n_components=18)  # dependent on number of classes
        self.training_fraction = training_fraction
        # self.caption_data = None
        # self.comment_data = None
        # self.snippet_data = None
        # self.related_channel_data = None
        # self.subscription_data = None
        # self.cross_comment_data = None
        # self.caption_model = None
        # self.final_model = None

    def load_data(self, serialized=True):
        self.caption_data = self.caption_trainer.get_data(binary=False, random_state=self.random_state,
                                                          serialized=serialized,
                                                          training_fraction=self.training_fraction)
        # comment_training_data, comment_test_data
        self.comment_data = self.comment_trainer.get_data(binary=False, random_state=self.random_state,
                                                          serialized=serialized,
                                                          training_fraction=self.training_fraction)
        # snippet_training_data, snippet_test_data
        self.snippet_data = self.snippet_trainer.get_data(binary=False, random_state=self.random_state,
                                                          serialized=serialized,
                                                          training_fraction=self.training_fraction)

        self.affiliation_data = self.affiliation_trainer.get_data(binary=False, random_state=self.random_state,
                                                                  serialized=serialized,
                                                                  training_fraction=self.training_fraction)
        self.subscription_data = self.subscription_trainer.get_data(binary=False, random_state=self.random_state,
                                                                    serialized=serialized,
                                                                    training_fraction=self.training_fraction)
        self.cross_comment_data = self.cross_comment_trainer.get_data(binary=False, random_state=self.random_state,
                                                                      serialized=serialized,
                                                                      training_fraction=self.training_fraction)

        # First we need to fit the binarizer to training _and_ test data for the case that all examples of one
        # class end up in the testing data (is this overly cautious?)
        self.mlb.fit(self.affiliation_data[0]['label'].append(self.affiliation_data[1]['label']))

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
            self.models['captions'] = self.caption_trainer.get_model(self.caption_data[0],
                                                                     multilabel=True)
            self.models['comments'] = self.comment_trainer.get_model(self.comment_data[0],
                                                                     multilabel=True)
            self.models['snippets'] = self.snippet_trainer.get_model(self.snippet_data[0],
                                                                     multilabel=True)
            self.models['related_channels'] = self.affiliation_trainer.get_model(self.affiliation_data[0],
                                                                                 multilabel=True,
                                                                                 multilabelbinarizer=self.mlb)
            self.models['subscriptions'] = self.subscription_trainer.get_model(self.subscription_data[0],
                                                                               multilabel=True,
                                                                               multilabelbinarizer=self.mlb)
            self.models['cross_comments'] = self.cross_comment_trainer.get_model(self.cross_comment_data[0],
                                                                                 multilabel=True,
                                                                                 multilabelbinarizer=self.mlb)

        # predict all values with the subclassifiers
        self.ensemble_data_training = self.estimate_probabilities(mode='Training')

        # training the final layer on the data
        training_y = self.mlb.transform(self.ensemble_data_training['label'])
        training_x = self.ensemble_data_training.drop(
            columns=['id', 'title', 'label', 'comments', 'snippets'])

        self.ensemble_model = OneVsRestClassifier(LogisticRegression(random_state=42,
                                                                     class_weight='balanced',
                                                                     solver='liblinear', max_iter=10000,
                                                                     C=0.01)).fit(training_x, training_y)

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

        # labels are for all the same so here I just use the labels from affiliations to update everything
        caption_estimations.label = affiliation_estimations.label.apply(tuple)
        comments_estimations.label = affiliation_estimations.label.apply(tuple)
        snippets_estimations.label = affiliation_estimations.label.apply(tuple)
        affiliation_estimations.label = affiliation_estimations.label.apply(tuple)
        subscription_estimations.label = affiliation_estimations.label.apply(tuple)
        cross_comment_estimations.label = affiliation_estimations.label.apply(tuple)

        # takes the data and creates the dataframe for the final logistic regression
        relation_result = affiliation_estimations.merge(subscription_estimations, on=['id', 'title', 'label']).merge(
            cross_comment_estimations, on=['id', 'title', 'label'])  # .reset_index()
        result = comments_estimations.merge(
            snippets_estimations, on=['id', 'label']).merge(affiliation_estimations, on=['id', 'label'])

        metadata = result[['id', 'title', 'label',  'comments', 'snippets']]
        data = result.drop(columns=['id', 'title', 'label',  'comments', 'snippets'])
        if mode == "Training":
            data = pd.DataFrame(self.scaler.fit_transform(self.imputer.fit_transform(data)))
            principal_df = pd.DataFrame(self.pca.fit_transform(data))
            result = pd.concat([metadata, principal_df], axis=1)

        # if testing mode, we also save this data as a variable, to load it easier in evaluation
        elif mode == 'Testing':
            data = pd.DataFrame(self.scaler.transform(self.imputer.transform(data)))
            principal_df = pd.DataFrame(self.pca.transform(data))
            result = pd.concat([metadata, principal_df], axis=1)
            self.ensemble_data_testing = result
        else:
            data = pd.DataFrame(self.scaler.transform(self.imputer.transform(data)))
            principal_df = pd.DataFrame(self.pca.transform(data))
            result = pd.concat([metadata, principal_df], axis=1)

        return result

    def predict(self, data):
        data = data.drop(columns=['id',
                                  'title',
                                  'label',
                                  'comments',
                                  'snippets'])
        labels = self.ensemble_model.predict(data)
        confidence_scores = self.ensemble_model.decision_function(data)
        probabilities = self.ensemble_model.predict_proba(data)
        return labels, confidence_scores, probabilities

    def evaluate(self, roc=False):
        training_y = self.mlb.transform(self.ensemble_data_training['label'])
        training_x = self.ensemble_data_training.drop(
            columns=['id', 'title', 'label', 'comments', 'snippets'])

        ensemble_data_testing = self.estimate_probabilities(mode='Testing')
        testing_y = self.mlb.transform(ensemble_data_testing['label'])
        testing_x = ensemble_data_testing.drop(columns=['id',
                                                        'title',
                                                        'label',
                                                        'comments',
                                                        'snippets'])

        training_results = calculate_multilabel_metrics_relation(training_x,
                                                                 training_y,
                                                                 self.ensemble_model,
                                                                 metric='MAP')

        test_results = calculate_multilabel_metrics_relation(testing_x,
                                                             testing_y,
                                                             self.ensemble_model,
                                                             metric='MAP')

        pred_report = calculate_multilabel_metrics_relation(testing_x,
                                                            testing_y,
                                                            self.ensemble_model,
                                                            metric='Report',
                                                            mlb=self.mlb)

        if roc:
            plot_data.plot_roc(self.ensemble_model, testing_x, testing_y)

        return training_results, test_results, pred_report

    # def add_unlabeled_data(self, cap, com, snip, aff, subs, cross):
    def add_unlabeled_data(self, cap, com, snip, aff, subs, cross):

        # todo: ugly, consider refactoring this

        # check whether unlabeled data has been added before
        if len(self.comment_data) == 2:
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
                probability_dict = {content + str(i): None for i in range(0, len(self.mlb.classes_))}
                # probabilities = [1 / len(self.mlb.classes_) for item in self.mlb.classes_]
                # probability_dict = {content + str(i): probabilities[i] for i in range(0, len(probabilities))}
            else:
                predicted_label, probabilities = self.models[content].predict(row['text'], k=-1)
                ordered_probabilities = np.zeros(len(self.mlb.classes_))
                for i, label in enumerate(predicted_label):
                    ordered_probabilities[np.where(self.mlb.classes_ == label)] = probabilities[i]
                probability_dict = {content + str(i): ordered_probabilities[i] for i in
                                    range(0, len(ordered_probabilities))}
            row_dict = {'id': row['id'], 'label': row['label'], content: row['text']}
            row_dict.update(probability_dict)
            row_dict_list.append(row_dict)
        result = pd.DataFrame(row_dict_list)
        return result

    def _classify_relations(self, data, content='related_channels'):
        row_dict_list = []
        for index, row in data.iterrows():
            id = row['id']
            title = row['title']
            label = row['label']
            # row_data = row.drop('label').drop('id').drop('title')
            row_data = row.iloc[:-3]
            probabilities = self.models[content].predict_proba([row_data])
            probability_dict = {content + str(i): probabilities[0][i] for i in range(0, len(probabilities[0]))}
            row_dict = {'id': id,
                        'title': title,
                        'label': label}
            row_dict.update(probability_dict)
            row_dict_list.append(row_dict)
        results = pd.DataFrame(row_dict_list)
        return results


if __name__ == '__main__':

    ensemble = MLEnsembleClassifier(training_fraction=1)
    ensemble.load_data(serialized=True)
    ensemble.fit(serialized=False)
    training_result, test_result, report = ensemble.evaluate(roc=False)
    print(report)

    iterations = 10
    reports = []
    avg_MAP = []
    for i in tqdm(range(iterations)):
        ensemble = MLEnsembleClassifier()
        ensemble.load_data(serialized=True)
        ensemble.fit(serialized=False)
        training_result, test_result, report = ensemble.evaluate(roc=False)
        print(training_result, test_result)
        reports.append(report)
        avg_MAP.append(test_result)
    print("##### Final f1 scores #### ")

    with open("data_pickles/results/ensemble_scores_balanced.pl", 'wb') as f:
        pickle.dump(avg_MAP, f)

    with open("data_pickles/results/ensemble_reports_balanced.pl", 'wb') as f:
        pickle.dump(reports, f)
    print(avg_MAP)
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    plt.scatter(range(iterations), avg_MAP)
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('MAP', fontsize=15)
    ax.set_title(f'Training {iterations} different Ensemble classifiers on 90 % of the data', fontsize=20)
    plt.show()
