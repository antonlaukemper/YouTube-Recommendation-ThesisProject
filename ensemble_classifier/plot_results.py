import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import shapiro, ttest_ind


def plot_scores(scores, scores_90):
    mean_scores_90 = []
    for i in range(0, np.max([len(score) for score in scores_90])):
        mean_scores_90.append(np.mean([score[i] if (len(score) > i) else score[-1] for score in scores_90]))

    fig, ax = plt.subplots()
    for index, score in enumerate(scores_90):
        plt.plot(range(0, len(score)), score)
    plt.plot(range(0, len(mean_scores_90)), mean_scores_90, label="mean", linewidth=4.0, c='k')
    plt.legend(fontsize=15)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel('Iteration', fontsize=20)
    ax.set_ylabel('F1-Score', fontsize=20)
    plt.show()
    limit = ax.get_ylim()

    mean_scores = []
    for i in range(0, np.max([len(score) for score in scores])):
        mean_scores.append(np.mean([score[i] if (len(score) > i) else score[-1] for score in scores]))

    fig, ax = plt.subplots()

    for index, score in enumerate(scores):
        plt.plot(range(0, len(score)), score)
    plt.plot(range(0, len(mean_scores)), mean_scores, label="mean", linewidth=4.0, c='k')
    plt.legend(fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlabel('Iteration', fontsize=20)
    ax.set_ylabel('F1-Score', fontsize=20)
    ax.set_ylim(limit)
    plt.show()


def plot_confidences(confidences):
    mean_scores = []
    for i in range(0, 6):
        mean_scores.append(np.mean([score[i] for score in confidences]))

    fig, ax = plt.subplots()
    for index, score in enumerate(confidences):
        plt.plot(range(0, len(score)), score, label="iteration %s" % str(index + 1))
    # plt.plot(range(0, len(mean_scores)), mean_scores, label="mean", linewidth=7.0, c='r')
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('Confidence Score', fontsize=15)
    plt.show()


def scatter_final_scores(scores):
    mean_scores = []
    for i in range(0, 10):
        mean_scores.append(np.mean([score['weighted avg']['f1-score'] for score in scores]))
    scores = [score['weighted avg']['f1-score'] for score in scores]

    fig, ax = plt.subplots()
    plt.scatter(range(0, len(scores)), scores)
    plt.plot(range(0, len(mean_scores)), mean_scores, label="mean", linewidth=3.0, c='r')
    plt.legend()
    plt.annotate(str(mean_scores[0]), xy=(0.01, mean_scores[0] + 0.002))
    ax.set_xlabel('Trial', fontsize=15)
    ax.set_ylabel('F1-Score', fontsize=15)
    plt.show()


def compare_results(supervised, ssl1, ssl2, ssl3):
    ssl1 = [score['weighted avg']['f1-score'] for score in ssl1]
    ssl2 = [score['weighted avg']['f1-score'] for score in ssl2]
    ssl3 = [score['weighted avg']['f1-score'] for score in ssl3]

    # supervised = [score['weighted avg']['f1-score'] for score in supervised]

    supervised_mean_scores = []
    for i in range(0, 10):
        supervised_mean_scores.append(np.mean([score for score in supervised]))
    ssl_mean_scores1 = []
    for i in range(0, 10):
        ssl_mean_scores1.append(np.mean([score for score in ssl1]))
    ssl_mean_scores2 = []
    for i in range(0, 10):
        ssl_mean_scores2.append(np.mean([score for score in ssl2]))
    ssl_mean_scores3 = []
    for i in range(0, 10):
        ssl_mean_scores3.append(np.mean([score for score in ssl3]))

    ## statistical comparison
    # checking normality with +

    stat, p = shapiro(supervised)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    stat, p = shapiro(ssl1)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    print(ttest_ind(supervised, ssl1))
    print(ttest_ind(supervised, ssl2))
    print(ttest_ind(supervised, ssl3))


    fig, ax = plt.subplots()
    plt.scatter(range(0, len(supervised)), supervised, c='c', label="Supervised")
    plt.scatter(range(0, len(ssl1)), ssl1, c='b', label="Self-Training 6 iterations")
    plt.scatter(range(0, len(ssl2)), ssl2, c='g', label="Self-Training 95% Threshold")
    plt.scatter(range(0, len(ssl3)), ssl3, c='r', label="Self-Training 90% Threshold")

    plt.plot(range(0, len(supervised_mean_scores)), supervised_mean_scores, linewidth=1.0, c='c')
    plt.plot(range(0, len(ssl_mean_scores1)), ssl_mean_scores1, linewidth=1.0, c='b')
    plt.plot(range(0, len(ssl_mean_scores2)), ssl_mean_scores2, linewidth=1.0, c='g')
    plt.plot(range(0, len(ssl_mean_scores3)), ssl_mean_scores3, linewidth=1.0, c='r')


    plt.legend()
    plt.annotate("mean: {:.5f}".format(supervised_mean_scores[0]), xy=(2, supervised_mean_scores[0] - 0.003))
    plt.annotate("mean: {:.5f}".format(ssl_mean_scores1[0]), xy=(0.01, ssl_mean_scores1[0] + 0.001))
    plt.annotate("mean: {:.5f}".format(ssl_mean_scores2[0]), xy=(0.01, ssl_mean_scores2[0] + 0.001))
    plt.annotate("mean: {:.5f}".format(ssl_mean_scores3[0]), xy=(0.01, ssl_mean_scores3[0] + 0.001))

    ax.set_xlabel('Trial', fontsize=15)
    ax.set_ylabel('F1-Score', fontsize=15)

    plt.show()


if __name__ == "__main__":
    with open("data_pickles/results/confidences_90.pl", 'rb') as path:
        mean_confidences = pickle.load(path)

    with open("data_pickles/results/classifier_scores_HighConf.pl", 'rb') as path:
        test_scores = pickle.load(path)

    with open("data_pickles/results/classifier_scores_90.pl", 'rb') as path:
        test_scores2 = pickle.load(path)

    with open("data_pickles/results/final_scores_fixed.pl", 'rb') as path:
        self_trainer_results1 = pickle.load(path)

    with open("data_pickles/results/final_scores_HighConf.pl", 'rb') as path:
        self_trainer_results2 = pickle.load(path)

    with open("data_pickles/results/final_scores_90.pl", 'rb') as path:
        self_trainer_results3 = pickle.load(path)

    # the better ensemble
    with open("data_pickles/results/ensemble_final_scores_lowerRegularization.pl", 'rb') as path:
        supervised_results = pickle.load(path)


    # plot_scores(test_scores, test_scores2)
    # plot_confidences(confidences=mean_confidences)
    # scatter_final_scores(self_trainer_results)
    compare_results(supervised_results, self_trainer_results1, self_trainer_results2, self_trainer_results3)

    with open("data_pickles/results/added_examples_HighConf.pl", 'rb') as path:
        added_examples = pickle.load(path)

    print(added_examples)
    print(np.mean(added_examples))

    print(np.mean([len(score) for score in test_scores]))
    print("ok")
