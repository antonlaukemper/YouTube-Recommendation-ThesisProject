from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from network_processing import experiments
from network_processing.load_data import get_connection_dataframe
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

from network_processing.relations_trainer import RelationsTrainer

#
# def get_model(model='logistic_regression', content='related_channels'):
#     if model=='logistic_regression':
#         if content=='related_channels':
#             regularization = 200
#         elif content == 'subscriptions':
#             regularization = 0.3
#         else: #content == 'cross_comments'
#             regularization = 1
#
#         return LogisticRegression(random_state=42,
#                                 solver='liblinear', max_iter=1000,
#                                 C=regularization)
#     # add potential other classifiers
#
#
# def train_model(classifier='logistic_regression', content='related_channels'):
#     affiliation_data = get_connection_dataframe(content)
#     affiliation_data = affiliation_data.sample(frac=1, random_state=42)
#     data = affiliation_data.drop(columns='label').drop(columns='id').drop(columns='title')
#     target = affiliation_data['label']
#     final_x_train, final_x_test, final_y_train, final_y_test = train_test_split(data, target, test_size=0.1, shuffle=False)
#     best_model = get_model(classifier, content).fit(final_x_train, final_y_train)
#
#
# affiliation_data = get_connection_dataframe(content='related_channels')
# #shuffle data
# affiliation_data = affiliation_data.sample(frac=1, random_state=42)
# data = affiliation_data.drop(columns='label').drop(columns='id').drop(columns='title')
# target = affiliation_data['label']
# final_x_train, final_x_test, final_y_train, final_y_test = train_test_split(data, target, test_size=0.1, shuffle=False)
# # experiments.vary_c(final_x_train, final_y_train, 'f1-score')
# # best_model = LogisticRegression(random_state=42,
# #                                 solver='liblinear',                                max_iter=1000,
# #                                 C=500).fit(final_x_train, final_y_train)
# best_model = LogisticRegression(random_state=42,
#                                 solver='liblinear', max_iter=1000,
#                                 C=200).fit(final_x_train, final_y_train)
# with open('models/affiliations/best_model.pl', 'wb') as f:
#     pickle.dump(best_model, f)
# print(experiments.calculate_metrics(best_model, final_x_test, final_y_test))
#
# # subscriptions
# subscription_data = get_connection_dataframe(content='subscriptions')
# # shuffle data
# subscription_data = subscription_data.sample(frac=1, random_state=42)
# data = subscription_data.drop(columns='label').drop(columns='title').drop(columns='id')
# target = subscription_data['label']
# final_x_train, final_x_test, final_y_train, final_y_test = train_test_split(data, target, test_size=0.1)
# # train_results, test_results = cross_validate(final_x_train, final_y_train, n=5, metric='f1-score')
# # print(train_results, test_results)
# # experiments.vary_c(final_x_train, final_y_train, 'f1-score')
# best_model = LogisticRegression(random_state=0,
#                                 solver='liblinear',
#                                 max_iter=1000,
#                                 C=0.3).fit(final_x_train, final_y_train)
#
# with open('models/subscriptions/best_model.pl', 'wb') as f:
#     pickle.dump(best_model, f)
# print(experiments.calculate_metrics(best_model, final_x_test, final_y_test))
#
#
# # cross-channel-comments
# cross_comment_data = get_connection_dataframe(content='cross_comments')
# #shuffle data
# cross_comment_data = cross_comment_data.sample(frac=1, random_state=42)
# data = cross_comment_data.drop(columns='label').drop(columns='title').drop(columns='id')
# target = cross_comment_data['label']
# final_x_train, final_x_test, final_y_train, final_y_test = train_test_split(data, target, test_size=0.1)
# # train_results, test_results = cross_validate(final_x_train, final_y_train, n=5, metric='f1-score')
# # print(train_results, test_results)
# # experiments.vary_c(final_x_train, final_y_train, 'f1-score')
# best_model = LogisticRegression(random_state=0,
#                                 solver='liblinear',
#                                 max_iter=1000,
#                                 C=1).fit(final_x_train, final_y_train)
# with open('models/cross-comments/best_model.pl', 'wb') as f:
#     pickle.dump(best_model, f)
# print(experiments.calculate_metrics(best_model, final_x_test, final_y_test))

if __name__ == "__main__":
    relation_trainer = RelationsTrainer()
    relation_data = relation_trainer.get_data(random_state=42)
    relation_model = relation_trainer.get_model(relation_data[0])

    final_y_test = relation_data[1]['label']
    final_x_test = relation_data[1].drop(columns='label').drop(columns='title').drop(columns='id')

    print(experiments.calculate_metrics(relation_model, final_x_test, final_y_test))