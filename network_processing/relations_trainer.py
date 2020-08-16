import os

from network_processing.load_data import get_connection_dataframe
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.multiclass import OneVsRestClassifier

from utils.data_IO import load_df


class RelationsTrainer():

    def __init__(self, mode='related_channels', regularization=200):
        self.mode = mode
        self.regularization = regularization
        self.only_known = True if mode == 'subscriptions' else False
        # self.only_known = False # only during 'testing' mode because otherwise there is no sub data
        self.evaluation_path_x = f'relation_data/{mode}/FINAL_EVALUATION.pl'
        self.output_path = f'models/{mode}/best_model.pl'

    def get_data(self, initial=True, binary=True, random_state=42, training_fraction=0.9, serialized=True):
        # training_fraction is the proportion of training data we take
        if serialized:
            data = load_df(content=self.mode, labeled=initial, binary=binary)
        else:
            data = get_connection_dataframe(content=self.mode, only_known=self.only_known, labeled=initial,
                                            binary=binary)

        if initial:
            data = data.sample(frac=1, random_state=42)
            training, testing = train_test_split(data, test_size=0.15, shuffle=False)
            if not os.path.isdir(os.path.dirname(self.evaluation_path_x)):
                os.makedirs(os.path.dirname(self.evaluation_path_x))
            with open(self.evaluation_path_x, 'wb') as f:
                pickle.dump(testing, f)
            # to be able to make a statistical statement about the performance of the classifier,
            # I sample from the training set 90% of the data
            training = training.sample(frac=training_fraction, random_state=random_state)
            return [training, testing]
        else:
            # also only take 90% of the unlabeled data
            return data.sample(frac=training_fraction, random_state=random_state)

    def get_model(self, data, model='logistic_regression', multilabel=False, multilabelbinarizer=None):
        if multilabel:
            data_y = multilabelbinarizer.transform(data['label'])
            data_x = data.drop(columns=['label', 'id', 'title'])
            model = OneVsRestClassifier(LogisticRegression(random_state=42,
                                                           class_weight='balanced',
                                                           solver='liblinear', max_iter=1000,
                                                           C=self.regularization)).fit(data_x, data_y)

        else:
            data_y = data['label']
            data_x = data.drop(columns=['label', 'id', 'title'])

            model = LogisticRegression(random_state=42,
                               solver='liblinear', max_iter=1000,
                               C=self.regularization).fit(data_x, data_y)

        # with open(self.evaluation_path_x, 'rb') as f:
        #     evaluation_data = pickle.load(f)
        #     evaluation_data_y = multilabelbinarizer.transform(evaluation_data['label'])
        #     evaluation_data_x = evaluation_data.drop(columns=['label', 'id', 'title'])
        #
        # results = calculate_multilabel_metrics_relation(evaluation_data_x, evaluation_data_y, model, metric='Hamming')
        # print(results)
        if not os.path.isdir(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))
        with open(self.output_path, 'wb') as f:
            pickle.dump(model, f)
        return model

    def evaluate(self):
        # todo
        pass