import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from multilabel_classification.multi_label_ensemble_classifier import MLEnsembleClassifier


class MLSelfTrainer:

    def __init__(self):
        self.classifier = MLEnsembleClassifier()
        # self.captions_unlabeled = self.classifier.caption_trainer.get_data(initial=False, binary=False,
        #                                                                    random_state=self.classifier.random_state,
        #                                                                    serialized=True)
        self.comments_unlabeled = self.classifier.comment_trainer.get_data(initial=False, binary=False,
                                                                           random_state=self.classifier.random_state,
                                                                           serialized=True)
        self.snippets_unlabeled = self.classifier.snippet_trainer.get_data(initial=False, binary=False,
                                                                           random_state=self.classifier.random_state,
                                                                           serialized=True)
        self.affiliations_unlabeled = self.classifier.affiliation_trainer.get_data(initial=False, binary=False,
                                                                           random_state=self.classifier.random_state,
                                                                           serialized=True)
        # self.subscriptions_unlabeled = self.classifier.subscription_trainer.get_data(initial=False, binary=False,
        #                                                                    random_state=self.classifier.random_state,
        #                                                                    serialized=True)
        # self.cross_comments_unlabeled = self.classifier.cross_comment_trainer.get_data(initial=False, binary=False,
        #                                                                    random_state=self.classifier.random_state,
        #                                                                    serialized=True)

        self.iterations = 6
        self.mean_confidences = []
        self.training_f1_scores = []
        self.testing_MAP_scores = []
        self.reports = []

    def load_data(self, serialized=True):
        self.classifier.load_data()
        self.classifier.add_unlabeled_data(#self.captions_unlabeled,
                                           self.comments_unlabeled,
                                           self.snippets_unlabeled,
                                           self.affiliations_unlabeled)#,
                                           # self.subscriptions_unlabeled,
                                           # self.cross_comments_unlabeled)

        # k is the parameter that determines how many unlabelled examples are added to the training set
        self.k = int(len(self.comments_unlabeled)/self.iterations + 0.5)

    def fit(self, serialized=False):
        self.classifier.fit(serialized=serialized)

    def label_examples(self):
        unlabeled_probabilities = self.classifier.estimate_probabilities(mode='Unlabeled')
        labels, confidence_scores, probabilities = self.classifier.predict(unlabeled_probabilities)
        confidence_scores = [np.mean(abs(i)) for i in confidence_scores]
        lowest_confidence = confidence_scores.copy()
        lowest_confidence.sort()
        lowest_confidence = lowest_confidence[:self.k]
        print("Lowest Confidence of predictions: %s" % str(lowest_confidence))
        self.mean_confidences.append(np.mean(lowest_confidence))
        self.predictions = pd.DataFrame(list(zip(labels, confidence_scores)),
                                        columns=['labels', 'confidence'],
                                        index= self.comments_unlabeled.index)

    def select_surest_predictions(self):
        if self.k < len(self.predictions):
            self.predictions.sort_values(by='confidence', ascending=False, inplace=True)
            print("mean confidence score of predicted examples: %f" % np.mean(self.predictions['confidence'].iloc[:self.k]))
            k_best_predictions = self.predictions.iloc[:self.k]
            indices = k_best_predictions.index
        else:
            indices = self.predictions.index
            print("mean confidence score of predicted examples: %f" % np.mean(self.predictions['confidence']))

        self.move_k_best_predictions(indices)

    def move_k_best_predictions(self, indices):
        labels = [self.classifier.mlb.inverse_transform(np.array([label])) for label in
             self.predictions['labels'].loc[indices]]

        # k_best_captions = self.captions_unlabeled.loc[indices]
        # k_best_captions['label'] = self._transform_labels(labels, 'text')
        k_best_comments = self.comments_unlabeled.loc[indices]
        k_best_comments['label'] = self._transform_labels(labels, 'text')
        k_best_snippets = self.snippets_unlabeled.loc[indices]
        k_best_snippets['label'] = self._transform_labels(labels, 'text')
        k_best_affiliations = self.affiliations_unlabeled.loc[indices]
        k_best_affiliations['label'] = self._transform_labels(labels, 'relation')
        # k_best_subscriptions = self.subscriptions_unlabeled.loc[indices]
        # k_best_subscriptions['label'] = self._transform_labels(labels, 'relation')
        # k_best_cross_comments = self.cross_comments_unlabeled.loc[indices]
        # k_best_cross_comments['label'] = self._transform_labels(labels, 'relation')

        # add the examples to the training data
        # self.classifier.caption_data[0] = self.classifier.caption_data[0].append(k_best_captions)
        self.classifier.comment_data[0] = self.classifier.comment_data[0].append(k_best_comments)
        self.classifier.snippet_data[0] = self.classifier.snippet_data[0].append(k_best_snippets)
        self.classifier.affiliation_data[0] = self.classifier.affiliation_data[0].append(k_best_affiliations)
        # self.classifier.subscription_data[0] = self.classifier.subscription_data[0].append(k_best_subscriptions)
        # self.classifier.cross_comment_data[0] = self.classifier.cross_comment_data[0].append(k_best_cross_comments)

        # self.captions_unlabeled = self.captions_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.comments_unlabeled = self.comments_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.snippets_unlabeled = self.snippets_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.affiliations_unlabeled = self.affiliations_unlabeled.drop(indices).reset_index().drop(columns='index')
        # self.subscriptions_unlabeled = self.subscriptions_unlabeled.drop(indices).reset_index().drop(columns='index')
        # self.cross_comments_unlabeled = self.cross_comments_unlabeled.drop(indices).reset_index().drop(columns='index')

        self.classifier.add_unlabeled_data(#self.captions_unlabeled,
                                           self.comments_unlabeled,
                                           self.snippets_unlabeled,
                                           self.affiliations_unlabeled) #,
                                           # self.subscriptions_unlabeled,
                                           # self.cross_comments_unlabeled)

    def label_all_data(self):
        self_trainer.load_data(serialized=True)
        for i in tqdm(range(0, self.iterations)):
            print("################ New Iteration ######################")
            print("length of unlabeled data: %d" % len(self.comments_unlabeled))
            self_trainer.fit(serialized=False)
            training_scores, testing_scores, report = self.classifier.evaluate()
            self.training_f1_scores.append(training_scores)
            self.testing_MAP_scores.append(testing_scores)
            self.reports.append(report)
            self_trainer.label_examples()
            self_trainer.select_surest_predictions()
        self_trainer.fit(serialized=False)
        training_scores, testing_scores , report = self.classifier.evaluate()
        self.testing_MAP_scores.append(testing_scores)
        self.reports.append(report)

        plt.figure()
        plt.plot(self.mean_confidences)
        plt.title(f"Mean confidence scores \n of {(1/self.iterations)*100}% least confident examples")
        plt.xlabel("Iteration")
        plt.ylabel("Confidence Score")
        plt.show()

        plt.figure()
        plt.plot(self.testing_MAP_scores)
        plt.title("Testing f1 scores")
        plt.xlabel("Iteration")
        plt.show()

        return self.testing_MAP_scores, self.mean_confidences, self.reports

    def _transform_labels(self, labels, mode):
        # todo potentially, empty labels need to be filled
        if mode == 'text':
            return [' '.join(label_list[0]) for label_list in labels]
        else:
            return [list(label_list[0]) for label_list in labels]



class ThesholdMLSelfTrainer:

    def __init__(self):
        self.classifier = MLEnsembleClassifier()
        # self.captions_unlabeled = self.classifier.caption_trainer.get_data(initial=False, binary=False,
        #                                                                    random_state=self.classifier.random_state,
        #                                                                    serialized=True)
        self.comments_unlabeled = self.classifier.comment_trainer.get_data(initial=False, binary=False,
                                                                           random_state=self.classifier.random_state,
                                                                           serialized=True)
        self.snippets_unlabeled = self.classifier.snippet_trainer.get_data(initial=False, binary=False,
                                                                           random_state=self.classifier.random_state,
                                                                           serialized=True)
        self.affiliations_unlabeled = self.classifier.affiliation_trainer.get_data(initial=False, binary=False,
                                                                           random_state=self.classifier.random_state,
                                                                           serialized=True)
        # self.subscriptions_unlabeled = self.classifier.subscription_trainer.get_data(initial=False, binary=False,
        #                                                                    random_state=self.classifier.random_state,
        #                                                                    serialized=True)
        # self.cross_comments_unlabeled = self.classifier.cross_comment_trainer.get_data(initial=False, binary=False,
        #                                                                    random_state=self.classifier.random_state,
        #                                                                    serialized=True)

        self.iterations = 0
        self.mean_confidences = []
        self.training_f1_scores = []
        self.testing_MAP_scores = []
        self.reports = []
        self.threshold = 0.33
        self.additional_data = 0
        self.stop = False

    def load_data(self, serialized=True):
        self.classifier.load_data()
        self.classifier.add_unlabeled_data(#self.captions_unlabeled,
                                           self.comments_unlabeled,
                                           self.snippets_unlabeled,
                                           self.affiliations_unlabeled)#,
                                           # self.subscriptions_unlabeled,
                                           # self.cross_comments_unlabeled)

        # k is the parameter that determines how many unlabelled examples are added to the training set
        self.k = int(len(self.comments_unlabeled)/10 + 0.5)

    def fit(self, serialized=False):
        self.classifier.fit(serialized=serialized)

    def label_examples(self):
        unlabeled_probabilities = self.classifier.estimate_probabilities(mode='Unlabeled')
        labels, confidence_scores, probabilities = self.classifier.predict(unlabeled_probabilities)
        confidence_scores = np.array([abs(0.5 - i) for i in probabilities])
        confidence_scores = [np.mean(abs(i)) for i in confidence_scores]
        lowest_confidence = confidence_scores.copy()
        lowest_confidence.sort()
        lowest_confidence = lowest_confidence[:self.k]
        print("Lowest Confidence of predictions: %s" % str(lowest_confidence))
        self.mean_confidences.append(np.mean(lowest_confidence))
        self.predictions = pd.DataFrame(list(zip(labels, confidence_scores)),
                                        columns=['labels', 'confidence'],
                                        index= self.comments_unlabeled.index)

    def select_surest_predictions(self):
        best_predictions = self.predictions[self.predictions['confidence'] >= self.threshold]
        if len(best_predictions) == 0:
            self.stop = True
            return
        print("length of best examples: %f" % len(best_predictions))
        print("mean confidence score of predicted examples: %f" % np.mean(best_predictions['confidence']))
        self.additional_data += len(best_predictions)

        indices = best_predictions.index


        self.move_k_best_predictions(indices)

    def move_k_best_predictions(self, indices):
        labels = [self.classifier.mlb.inverse_transform(np.array([label])) for label in
             self.predictions['labels'].loc[indices]]

        # k_best_captions = self.captions_unlabeled.loc[indices]
        # k_best_captions['label'] = self._transform_labels(labels, 'text')
        k_best_comments = self.comments_unlabeled.loc[indices]
        k_best_comments['label'] = self._transform_labels(labels, 'text')
        k_best_snippets = self.snippets_unlabeled.loc[indices]
        k_best_snippets['label'] = self._transform_labels(labels, 'text')
        k_best_affiliations = self.affiliations_unlabeled.loc[indices]
        k_best_affiliations['label'] = self._transform_labels(labels, 'relation')
        # k_best_subscriptions = self.subscriptions_unlabeled.loc[indices]
        # k_best_subscriptions['label'] = self._transform_labels(labels, 'relation')
        # k_best_cross_comments = self.cross_comments_unlabeled.loc[indices]
        # k_best_cross_comments['label'] = self._transform_labels(labels, 'relation')

        # add the examples to the training data
        # self.classifier.caption_data[0] = self.classifier.caption_data[0].append(k_best_captions)
        self.classifier.comment_data[0] = self.classifier.comment_data[0].append(k_best_comments)
        self.classifier.snippet_data[0] = self.classifier.snippet_data[0].append(k_best_snippets)
        self.classifier.affiliation_data[0] = self.classifier.affiliation_data[0].append(k_best_affiliations)
        # self.classifier.subscription_data[0] = self.classifier.subscription_data[0].append(k_best_subscriptions)
        # self.classifier.cross_comment_data[0] = self.classifier.cross_comment_data[0].append(k_best_cross_comments)

        # self.captions_unlabeled = self.captions_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.comments_unlabeled = self.comments_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.snippets_unlabeled = self.snippets_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.affiliations_unlabeled = self.affiliations_unlabeled.drop(indices).reset_index().drop(columns='index')
        # self.subscriptions_unlabeled = self.subscriptions_unlabeled.drop(indices).reset_index().drop(columns='index')
        # self.cross_comments_unlabeled = self.cross_comments_unlabeled.drop(indices).reset_index().drop(columns='index')

        self.classifier.add_unlabeled_data(#self.captions_unlabeled,
                                           self.comments_unlabeled,
                                           self.snippets_unlabeled,
                                           self.affiliations_unlabeled) #,
                                           # self.subscriptions_unlabeled,
                                           # self.cross_comments_unlabeled)

    def label_all_data(self):
        self.load_data(serialized=True)
        while not self.stop and len(self.comments_unlabeled) > 0:
            self.iterations += 1
            print("################ New Iteration ######################")
            print("length of unlabeled data: %d" % len(self.comments_unlabeled))
            self.fit(serialized=False)
            training_scores, testing_scores, report = self.classifier.evaluate()
            self.training_f1_scores.append(training_scores)
            self.testing_MAP_scores.append(testing_scores)
            self.reports.append(report)
            self.label_examples()
            self.select_surest_predictions()
        self.fit(serialized=False)
        training_scores, testing_scores , report = self.classifier.evaluate()
        self.testing_MAP_scores.append(testing_scores)
        self.reports.append(report)

        plt.figure()
        plt.plot(self.mean_confidences)
        plt.title(f"Mean confidence scores \n of {(1/self.iterations)*100}% least confident examples")
        plt.xlabel("Iteration")
        plt.ylabel("Confidence Score")
        plt.show()

        plt.figure()
        plt.plot(self.testing_MAP_scores)
        plt.title("Testing f1 scores")
        plt.xlabel("Iteration")
        plt.show()

        return self.testing_MAP_scores, self.mean_confidences, self.reports, self.additional_data

    def _transform_labels(self, labels, mode):
        # todo potentially, empty labels need to be filled
        if mode == 'text':
            return [' '.join(label_list[0]) for label_list in labels]
        else:
            return [list(label_list[0]) for label_list in labels]

if __name__ == '__main__':
    iterations = 10
    maps = []
    confidences = []
    reports = []
    final_maps = []
    additional_data = []
    for i in tqdm(range(iterations)):
        self_trainer = ThesholdMLSelfTrainer()
        testing_scores, confidence, report, added_data = self_trainer.label_all_data()
        maps.append(testing_scores)
        confidences.append(confidence),
        reports.append(report)
        training_result, test_result, final_report = self_trainer.classifier.evaluate(roc=False)
        additional_data.append(added_data)
        print(final_report)
        final_maps.append(test_result)
    with open("data_pickles/results/maps_33.pl", 'wb') as f:
        pickle.dump(maps, f)
    with open("data_pickles/results/confidences_33.pl", 'wb') as f:
        pickle.dump(confidences, f)
    with open("data_pickles/results/reports_33.pl", 'wb') as f:
        pickle.dump(reports, f)
    with open("data_pickles/results/self_trainer_scores_33.pl", 'wb') as f:
        pickle.dump(final_maps, f)
    with open("data_pickles/results/self_trainer_added_data_33.pl", 'wb') as f:
        pickle.dump(additional_data, f)
    print("##### Final map scores #### ")
    print(final_maps)
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.scatter(range(0,iterations), final_maps)
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('MAP', fontsize=15)
    ax.set_title(f'Training {iterations} different Self-Trainer classifiers on 90 % of the data', fontsize=20)
    plt.show()