import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ttest_ind


def plot_scores(scores, scores2):
    mean_scores = []
    for i in range(0, np.max([len(score) for score in scores])):
        mean_scores.append(np.mean([score[i] if (len(score) > i) else score[-1] for score in scores]))


    fig, ax = plt.subplots()
    for index, score in enumerate(scores):
        plt.plot(range(0,len(score)), score)
    plt.plot(range(0,len(mean_scores)), mean_scores, label="mean", linewidth=3.0, c='k')
    plt.legend(fontsize=15)
    plt.xticks(range(0,len(mean_scores)+1))
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel('Iteration', fontsize=20)
    ax.set_ylabel('Mean-Average-Precision', fontsize=20)
    limit = ax.get_ylim()
    plt.show()

    mean_scores = []
    for i in range(0, np.max([len(score) for score in scores2])):
        mean_scores.append(np.mean([score[i] if (len(score) > i) else score[-1] for score in scores2]))

    fig, ax = plt.subplots()
    for index, score in enumerate(scores2):
        plt.plot(range(0, len(score)), score)
    plt.plot(range(0, len(mean_scores)), mean_scores, label="mean", linewidth=3.0, c='k')
    plt.legend(fontsize=15)
    plt.xticks(range(0, len(mean_scores) + 1))
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel('Iteration', fontsize=20)
    ax.set_ylabel('Mean-Average-Precision', fontsize=20)
    ax.set_ylim(limit)
    plt.show()

def plot_confidences(confidences):
    mean_scores = []
    for i in range(0, 6):
        mean_scores.append(np.mean([score[i] for score in confidences]))


    fig, ax = plt.subplots()
    for index, score in enumerate(confidences):
        plt.plot(range(0,len(score)), score, label="iteration %s" % str(index+1))
    # plt.plot(range(0,len(mean_scores)), mean_scores, label="mean", linewidth=7.0, c='r')
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('Mean Confidence', fontsize=15)
    ax.set_title('Confidence of least confident examples \n after each Training Iteration', fontsize=20)
    plt.show()


def compare_results(supervised, supervised_o3, ssl, ssl_o3, ssl_30, ssl_33):
    supervised_mean_scores = []
    for i in range(0, 10):
        supervised_mean_scores.append(np.mean([score for score in supervised]))
    supervised_mean_scores_o3 = []
    for i in range(0, 10):
        supervised_mean_scores_o3.append(np.mean([score for score in supervised_o3]))

    ssl_mean_scores = []
    for i in range(0, 10):
        ssl_mean_scores.append(np.mean([score for score in ssl]))
    ssl_mean_scores_o3 = []
    for i in range(0, 10):
        ssl_mean_scores_o3.append(np.mean([score for score in ssl_o3]))
    ssl_mean_scores_30 = []
    for i in range(0, 10):
        ssl_mean_scores_30.append(np.mean([score for score in ssl_30]))

    ssl_mean_scores_33 = []
    for i in range(0, 10):
        ssl_mean_scores_33.append(np.mean([score for score in ssl_33]))

    fig, ax = plt.subplots()
    plt.scatter(range(0, len(supervised)), supervised, c='c', label="Supervised")
    plt.scatter(range(0, len(ssl)), ssl, c='b', label="Self-Training")
    plt.plot(range(0,len(supervised_mean_scores)), supervised_mean_scores, linewidth=1.0, c='c')
    plt.plot(range(0,len(ssl_mean_scores)), ssl_mean_scores,  linewidth=1.0, c='b')

    plt.scatter(range(0, len(supervised_o3)), supervised_o3, c='r', label="Supervised Reduced Ensemble")
    plt.scatter(range(0, len(ssl_o3)), ssl_o3, c='g', label="Self-Training Reduced Ensemble")
    plt.plot(range(0,len(supervised_mean_scores_o3)), supervised_mean_scores_o3, linewidth=1.0, c='r')
    plt.plot(range(0,len(ssl_mean_scores_o3)), ssl_mean_scores_o3,  linewidth=1.0, c='g')

    plt.scatter(range(0, len(ssl_30)), ssl_30, c='y', label="Self-Training 80%")
    plt.plot(range(0,len(ssl_mean_scores_30)), ssl_mean_scores_30, linewidth=1.0, c='y')

    plt.scatter(range(0, len(ssl_33)), ssl_33, c='m', label="Self-Training 83%")
    plt.plot(range(0,len(ssl_mean_scores_33)), ssl_mean_scores_33, linewidth=1.0, c='m')

    plt.legend(loc='lower right', fontsize=8)
    plt.annotate("mean: {:.3f}".format(supervised_mean_scores[0]), xy=(2, supervised_mean_scores[0]+0.002))
    plt.annotate("mean: {:.3f} ".format(ssl_mean_scores[0]), xy=(0.01, ssl_mean_scores[0] + 0.002))
    plt.annotate("mean: {:.3f}".format(supervised_mean_scores_o3[0]), xy=(0.01, supervised_mean_scores_o3[0]-0.006))
    plt.annotate("mean: {:.3f} ".format(ssl_mean_scores_o3[0]), xy=(0.01, ssl_mean_scores_o3[0] + 0.002))
    plt.annotate("mean: {:.3f} ".format(ssl_mean_scores_30[0]), xy=(0.01, ssl_mean_scores_30[0] + 0.002))
    plt.annotate("mean: {:.3f} ".format(ssl_mean_scores_33[0]), xy=(0.01, ssl_mean_scores_33[0] + 0.002))


    ax.set_xlabel('Trial', fontsize=15)
    ax.set_ylabel('Mean-Average-Precision', fontsize=15)
    plt.show()


    print(ttest_ind(supervised, supervised_o3))
    print(ttest_ind(supervised, ssl))
    print(ttest_ind(supervised, ssl_o3))
    print(ttest_ind(supervised, ssl_30))
    print(ttest_ind(supervised, ssl_33))
    print(ttest_ind(supervised_o3, ssl))
    print(ttest_ind(supervised_o3, ssl_o3))
    print(ttest_ind(supervised_o3, ssl_30))
    print(ttest_ind(supervised_o3, ssl_33))


if __name__ == "__main__":

    with open("data_pickles/results/confidences_30.pl", 'rb') as path:
        mean_confidences = pickle.load(path)

    with open("data_pickles/results/maps.pl", 'rb') as path:
        test_scores = pickle.load(path)
    with open("data_pickles/results/maps_30.pl", 'rb') as path:
        test_scores2 = pickle.load(path)

    with open("data_pickles/results/self_trainer_scores.pl", 'rb') as path:
        self_trainer_results = pickle.load(path)

    with open("data_pickles/results/self_trainer_scores_only3.pl", 'rb') as path:
        self_trainer_results_only3 = pickle.load(path)

    with open("data_pickles/results/self_trainer_scores_30.pl", 'rb') as path:
        self_trainer_results_30 = pickle.load(path)

    with open("data_pickles/results/self_trainer_scores_33.pl", 'rb') as path:
        self_trainer_results_33 = pickle.load(path)

    with open("data_pickles/results/ensemble_scores.pl", 'rb') as path:
        supervised_results = pickle.load(path)

    with open("data_pickles/results/ensemble_scores_balanced.pl", 'rb') as path:
        supervised_results_only3 = pickle.load(path)

    plot_scores(test_scores, test_scores2)
    plot_confidences(confidences=mean_confidences)
    compare_results(supervised_results, supervised_results_only3, self_trainer_results, self_trainer_results_only3, self_trainer_results_30, self_trainer_results_33)

