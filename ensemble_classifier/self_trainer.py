import pickle
import time

from ensemble_classifier import EnsembleClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class SelfTrainer:

    def __init__(self):
        self.classifier = EnsembleClassifier()
        self.captions_unlabeled = self.classifier.caption_trainer.get_data(initial=False,
                                                                           random_state=self.classifier.random_state)
        self.comments_unlabeled = self.classifier.comment_trainer.get_data(initial=False,
                                                                           random_state=self.classifier.random_state)
        self.snippets_unlabeled = self.classifier.snippet_trainer.get_data(initial=False,
                                                                           random_state=self.classifier.random_state)
        self.affiliations_unlabeled = self.classifier.affiliation_trainer.get_data(initial=False,
                                                                                   random_state=self.classifier.random_state)
        self.subscriptions_unlabeled = self.classifier.subscription_trainer.get_data(initial=False,
                                                                                     random_state=self.classifier.random_state)
        self.cross_comments_unlabeled = self.classifier.cross_comment_trainer.get_data(initial=False,
                                                                                       random_state=self.classifier.random_state)

        self.iterations = 6
        self.mean_confidences = []
        self.training_f1_scores = []
        self.testing_f1_scores = []

    def load_data(self, serialized=True):
        self.classifier.load_data()
        self.classifier.add_unlabeled_data(self.captions_unlabeled,
                                           self.comments_unlabeled,
                                           self.snippets_unlabeled,
                                           self.affiliations_unlabeled,
                                           self.subscriptions_unlabeled,
                                           self.cross_comments_unlabeled)

        # k is the parameter that determines how many unlabelled examples are added to the training set
        self.k = int(len(self.captions_unlabeled) / self.iterations + 0.5)

    def fit(self, serialized=False):
        self.classifier.fit(serialized=serialized)

    def label_examples(self):
        unlabeled_probabilities = self.classifier.estimate_probabilities(mode='Unlabeled')
        labels, confidence_scores, probabilities = self.classifier.predict(unlabeled_probabilities)
        confidence_scores = [abs(i) for i in confidence_scores]
        lowest_confidence = confidence_scores.copy()
        lowest_confidence.sort()
        lowest_confidence = lowest_confidence[:self.k]
        print("Lowest Confidence of predictions: %s" % str(lowest_confidence))
        self.mean_confidences.append(np.mean(lowest_confidence))
        self.predictions = pd.DataFrame(list(zip(labels, confidence_scores)),
                                        columns=['labels', 'confidence'],
                                        index=self.captions_unlabeled.index)

    def select_surest_predictions(self):
        if self.k < len(self.predictions):
            self.predictions.sort_values(by='confidence', ascending=False, inplace=True)
            print("mean confidence score of predicted examples: %f" % np.mean(
                self.predictions['confidence'].iloc[:self.k]))
            k_best_predictions = self.predictions.iloc[:self.k]
            indices = k_best_predictions.index
        else:
            indices = self.predictions.index
            print("mean confidence score of predicted examples: %f" % np.mean(self.predictions['confidence']))

        self.move_k_best_predictions(indices)

    def move_k_best_predictions(self, indices):
        k_best_captions = self.captions_unlabeled.loc[indices]
        k_best_captions['label'] = self.predictions['labels'].loc[indices]
        k_best_comments = self.comments_unlabeled.loc[indices]
        k_best_comments['label'] = self.predictions['labels'].loc[indices]
        k_best_snippets = self.snippets_unlabeled.loc[indices]
        k_best_snippets['label'] = self.predictions['labels'].loc[indices]
        k_best_affiliations = self.affiliations_unlabeled.loc[indices]
        k_best_affiliations['label'] = self.predictions['labels'].loc[indices]
        k_best_subscriptions = self.subscriptions_unlabeled.loc[indices]
        k_best_subscriptions['label'] = self.predictions['labels'].loc[indices]
        k_best_cross_comments = self.cross_comments_unlabeled.loc[indices]
        k_best_cross_comments['label'] = self.predictions['labels'].loc[indices]

        # add the examples to the training data
        self.classifier.caption_data[0] = self.classifier.caption_data[0].append(k_best_captions)
        self.classifier.comment_data[0] = self.classifier.comment_data[0].append(k_best_comments)
        self.classifier.snippet_data[0] = self.classifier.snippet_data[0].append(k_best_snippets)
        self.classifier.affiliation_data[0] = self.classifier.affiliation_data[0].append(k_best_affiliations)
        self.classifier.subscription_data[0] = self.classifier.subscription_data[0].append(k_best_subscriptions)
        self.classifier.cross_comment_data[0] = self.classifier.cross_comment_data[0].append(k_best_cross_comments)

        self.captions_unlabeled = self.captions_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.comments_unlabeled = self.comments_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.snippets_unlabeled = self.snippets_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.affiliations_unlabeled = self.affiliations_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.subscriptions_unlabeled = self.subscriptions_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.cross_comments_unlabeled = self.cross_comments_unlabeled.drop(indices).reset_index().drop(columns='index')

        self.classifier.add_unlabeled_data(self.captions_unlabeled,
                                           self.comments_unlabeled,
                                           self.snippets_unlabeled,
                                           self.affiliations_unlabeled,
                                           self.subscriptions_unlabeled,
                                           self.cross_comments_unlabeled)

    def label_all_data(self):
        self_trainer.load_data(serialized=True)
        for i in tqdm(range(0, self.iterations)):
            print("################ New Iteration ######################")
            print("length of unlabeled data: %d" % len(self.captions_unlabeled))
            start_time = time.time()
            self_trainer.fit(serialized=False)
            print(time.time() - start_time)
            training_scores, testing_scores = self.classifier.evaluate()
            self.training_f1_scores.append(training_scores['weighted avg']['f1-score'])
            self.testing_f1_scores.append(testing_scores['weighted avg']['f1-score'])
            start_time = time.time()
            self_trainer.label_examples()
            print(time.time() - start_time)
            self_trainer.select_surest_predictions()
        self_trainer.fit(serialized=False)
        training_scores, testing_scores = self.classifier.evaluate()
        self.testing_f1_scores.append(testing_scores['weighted avg']['f1-score'])

        plt.figure()
        plt.plot(self.mean_confidences)
        plt.title(f"Mean confidence scores \n of {(1/self.iterations)*100}% least confident examples")
        plt.xlabel("Iteration")
        plt.ylabel("Confidence Score")
        plt.show()

        plt.figure()
        plt.plot(self.testing_f1_scores)
        plt.title("Testing f1 scores")
        plt.xlabel("Iteration")
        plt.show()

        return self.mean_confidences, self.testing_f1_scores


class ThresholdSelfTrainer:

    def __init__(self):
        self.classifier = EnsembleClassifier()
        self.captions_unlabeled = self.classifier.caption_trainer.get_data(initial=False,
                                                                           random_state=self.classifier.random_state)
        self.comments_unlabeled = self.classifier.comment_trainer.get_data(initial=False,
                                                                           random_state=self.classifier.random_state)
        self.snippets_unlabeled = self.classifier.snippet_trainer.get_data(initial=False,
                                                                           random_state=self.classifier.random_state)
        self.affiliations_unlabeled = self.classifier.affiliation_trainer.get_data(initial=False,
                                                                                   random_state=self.classifier.random_state)
        self.subscriptions_unlabeled = self.classifier.subscription_trainer.get_data(initial=False,
                                                                                     random_state=self.classifier.random_state)
        self.cross_comments_unlabeled = self.classifier.cross_comment_trainer.get_data(initial=False,
                                                                                       random_state=self.classifier.random_state)

        self.iterations = 10
        self.mean_confidences = []
        self.training_f1_scores = []
        self.testing_f1_scores = []
        self.stop = False
        self.additional_data = 0

    def load_data(self, serialized=True):
        self.classifier.load_data()
        self.classifier.add_unlabeled_data(self.captions_unlabeled,
                                           self.comments_unlabeled,
                                           self.snippets_unlabeled,
                                           self.affiliations_unlabeled,
                                           self.subscriptions_unlabeled,
                                           self.cross_comments_unlabeled)

        # in this setting, k is the parameter that determines how many examples are selected for the confidence comparison
        self.k = int(len(self.captions_unlabeled) / 10 + 0.5)

    def fit(self, serialized=False):
        self.classifier.fit(serialized=serialized)

    def label_examples(self):
        unlabeled_probabilities = self.classifier.estimate_probabilities(mode='Unlabeled')
        labels, confidence_scores, probabilities = self.classifier.predict(unlabeled_probabilities)
        confidence_scores = np.array([abs(0.5 - i) for i in probabilities])
        lowest_confidence = confidence_scores.copy()
        lowest_confidence.sort()
        lowest_confidence = lowest_confidence[
                            :self.k]  # the lowest 10% are selected to see whether the general confidence increases
        print("Lowest Confidence of predictions: %s" % str(np.mean(lowest_confidence)))
        self.mean_confidences.append(np.mean(lowest_confidence))
        self.predictions = pd.DataFrame(list(zip(labels, confidence_scores)),
                                        columns=['labels', 'confidence'],
                                        index=self.captions_unlabeled.index)

    def select_surest_predictions(self):
        best_predictions = self.predictions[self.predictions['confidence'] >= 0.4]
        if len(best_predictions) == 0:
            self.stop = True
            return
        print("length of best examples: %f" % len(best_predictions))
        num_political = len(best_predictions[best_predictions['labels'] == '__label__political'])
        print("%f political examples added and %f non-political" % (
        num_political, (len(best_predictions) - num_political)))
        self.additional_data += len(best_predictions)
        indices = best_predictions.index

        self.move_k_best_predictions(indices)

    def move_k_best_predictions(self, indices):
        k_best_captions = self.captions_unlabeled.loc[indices]
        k_best_captions['label'] = self.predictions['labels'].loc[indices]
        k_best_comments = self.comments_unlabeled.loc[indices]
        k_best_comments['label'] = self.predictions['labels'].loc[indices]
        k_best_snippets = self.snippets_unlabeled.loc[indices]
        k_best_snippets['label'] = self.predictions['labels'].loc[indices]
        k_best_affiliations = self.affiliations_unlabeled.loc[indices]
        k_best_affiliations['label'] = self.predictions['labels'].loc[indices]
        k_best_subscriptions = self.subscriptions_unlabeled.loc[indices]
        k_best_subscriptions['label'] = self.predictions['labels'].loc[indices]
        k_best_cross_comments = self.cross_comments_unlabeled.loc[indices]
        k_best_cross_comments['label'] = self.predictions['labels'].loc[indices]

        # add the examples to the training data
        self.classifier.caption_data[0] = self.classifier.caption_data[0].append(k_best_captions)
        self.classifier.comment_data[0] = self.classifier.comment_data[0].append(k_best_comments)
        self.classifier.snippet_data[0] = self.classifier.snippet_data[0].append(k_best_snippets)
        self.classifier.affiliation_data[0] = self.classifier.affiliation_data[0].append(k_best_affiliations)
        self.classifier.subscription_data[0] = self.classifier.subscription_data[0].append(k_best_subscriptions)
        self.classifier.cross_comment_data[0] = self.classifier.cross_comment_data[0].append(k_best_cross_comments)

        self.captions_unlabeled = self.captions_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.comments_unlabeled = self.comments_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.snippets_unlabeled = self.snippets_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.affiliations_unlabeled = self.affiliations_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.subscriptions_unlabeled = self.subscriptions_unlabeled.drop(indices).reset_index().drop(columns='index')
        self.cross_comments_unlabeled = self.cross_comments_unlabeled.drop(indices).reset_index().drop(columns='index')

        self.classifier.add_unlabeled_data(self.captions_unlabeled,
                                           self.comments_unlabeled,
                                           self.snippets_unlabeled,
                                           self.affiliations_unlabeled,
                                           self.subscriptions_unlabeled,
                                           self.cross_comments_unlabeled)

    def label_all_data(self):
        self_trainer.load_data(serialized=True)
        while not self.stop:
            print("################ New Iteration ######################")
            self.iterations += 1
            print("length of unlabeled data: %d" % len(self.captions_unlabeled))
            start_time = time.time()
            self_trainer.fit(serialized=False)
            print("time it took to train: %f" % ((time.time() - start_time) / 60))
            start_time = time.time()
            training_scores, testing_scores = self.classifier.evaluate()
            self.training_f1_scores.append(training_scores['weighted avg']['f1-score'])
            self.testing_f1_scores.append(testing_scores['weighted avg']['f1-score'])
            self_trainer.label_examples()
            print("time it took to predict: %f" % ((time.time() - start_time) / 60))
            self_trainer.select_surest_predictions()
        self_trainer.fit(serialized=False)
        training_scores, testing_scores = self.classifier.evaluate()
        self.testing_f1_scores.append(testing_scores['weighted avg']['f1-score'])

        plt.figure()
        plt.plot(self.mean_confidences)
        plt.title(f"Mean confidence scores \n of {(1/10)*100}% least confident examples")
        plt.xlabel("Iteration")
        plt.ylabel("Confidence Score")
        plt.show()

        plt.figure()
        plt.plot(self.testing_f1_scores)
        plt.title("Testing f1 scores")
        plt.xlabel("Iteration")
        plt.show()

        return self.mean_confidences, self.testing_f1_scores, self.additional_data, self.iterations


if __name__ == '__main__':
    trials = 5
    print("final evaluation")
    final_test_results = []
    confidences = []
    test_scores = []
    added_examples = []
    iterations = []
    avg_f1 = []
    for i in tqdm(range(trials)):
        self_trainer = SelfTrainer()
        # confidence, test_score, added_example, iteration = self_trainer.label_all_data()
        confidence, test_score = self_trainer.label_all_data()
        confidences.append(confidence)
        test_scores.append(test_score)
        training_result, test_result = self_trainer.classifier.evaluate(roc=False)
        print(test_result)
        final_test_results.append(test_result)
        avg_f1.append(test_result['weighted avg']['f1-score'])
        # added_examples.append(added_example)
        # iterations.append(iteration)
    with open("data_pickles/results/final_scores_wtf.pl", 'wb') as f:
        pickle.dump(final_test_results, f)
    with open("data_pickles/results/confidences_wtf.pl", 'wb') as f:
        pickle.dump(confidences, f)
    with open("data_pickles/results/classifier_scores_wtf.pl", 'wb') as f:
        pickle.dump(test_scores, f)
    # with open("data_pickles/results/added_examples_90.pl", 'wb') as f:
    #     pickle.dump(added_examples, f)
    # with open("data_pickles/results/iterations_90.pl", 'wb') as f:
    #     pickle.dump(iterations, f)
    # print("Number of examples added: ")
    # print(added_examples)
    print("##### Final f1 scores #### ")
    print(avg_f1)
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    plt.scatter(range(0, trials), avg_f1)
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('f1-score', fontsize=15)
    ax.set_title(f'Training {trials} different Self-Trainer classifiers on 90 % of the data', fontsize=20)
    plt.show()
