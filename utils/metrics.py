from sklearn.metrics import classification_report, average_precision_score
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss


def calculate_metrics(path, model):
    with open(path, 'r') as test_data:
        ground_truth = []
        predicted = []
        for line in test_data.readlines():
            parts = line.split(' ')
            label = parts[1]
            ground_truth.append(label)
            text = ' '.join(parts[2:])
            predicted_label, probability = model.predict(text[:-1])
            predicted.append(predicted_label)
        # y_true = pd.Series(ground_truth)
        # y_pred = pd.Series(predicted)
        # confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
        return classification_report(ground_truth, predicted, output_dict=True)


def calculate_multilabel_metrics_text(path, model, metric='Hamming', threshold=0.1, verbose=False):
    mlb = MultiLabelBinarizer()
    mlb.fit([model.labels])
    with open(path, 'r') as test_data:
        ground_truth = []
        predicted = []
        probabilities = []
        for line in test_data.readlines():
            parts = line.split(' ')
            label_count = len(line.split('__label__')) - 1
            labels = [parts[i] for i in range(1, label_count * 2 + 1, 2)]
            ground_truth.append(labels)
            text = ' '.join(parts[label_count * 2:])
            predicted_labels, probability = model.predict(text[:-1], k=-1)  # , threshold=threshold)
            ordered_probabilities = np.zeros(len(mlb.classes_))
            for i, label in enumerate(predicted_labels):
                ordered_probabilities[np.where(mlb.classes_==label)] = probability[i]
            predicted_labels = get_best_labels(predicted_labels, probability)
            predicted.append(list(predicted_labels))
            probabilities.append(ordered_probabilities)
            if verbose:
                print(labels, "###", predicted_labels)
        predicted = mlb.transform(predicted)
        ground_truth = mlb.transform(ground_truth)

        if metric == 'Hamming':
            print('Hamming loss: {0}'.format(hamming_loss(ground_truth, predicted)))
            print('Hamming_score: {0}'.format(hamming_score(ground_truth, predicted)))
            return hamming_score(ground_truth, predicted)
        elif metric == "MAP":
            return MAP(ground_truth, probabilities)
        elif metric == "Report":
            return classification_report(ground_truth, predicted, target_names=mlb.classes_)

def calculate_multilabel_metrics_relation(data, ground_truth, model, metric='Hamming', mlb=None):
    prediction = model.predict(data)
    probabilities = model.predict_proba(data)

    if metric == 'Hamming':
        print('Hamming loss: {0}'.format(hamming_loss(ground_truth, prediction)))
        print('Hamming_score: {0}'.format(hamming_score(ground_truth, prediction)))
        return hamming_score(ground_truth, prediction)
    elif metric == "MAP":
        return MAP(ground_truth, probabilities)
    elif metric == "Report":
        return classification_report(ground_truth, prediction, target_names=mlb.classes_, zero_division=0)



def get_best_labels(labels, probabilities):
    prob_deltas = [probabilities[i] - probabilities[i + 1] for i in range(len(probabilities) - 1)]
    # n is the index of the largest difference in the label probabilities between label_i and label_i+1
    n = prob_deltas.index(max(prob_deltas))
    return labels[:n + 1]

def MAP(ground_truth, probabilities):
    avg_precisions = [average_precision_score(true_values, probs) for true_values, probs in zip(ground_truth, probabilities)]
    return np.mean(avg_precisions)

# Code by https://stackoverflow.com/users/1953100/william
# Source: https://stackoverflow.com/a/32239764/395857
# License: cc by-sa 3.0 with attribution required

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        # print('\nset_true: {0}'.format(set_true))
        # print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        # base case if both predict no labels
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / \
                    float(len(set_true.union(set_pred)))
        # print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


if __name__ == "__main__":
    print('Hamming score: {0}'.format(hamming_score(y_true, y_pred)))  # 0.375 (= (0.5+1+0+0)/4)

    # For comparison sake:
    import sklearn.metrics

    # Subset accuracy
    # 0.25 (= 0+1+0+0 / 4) --> 1 if the prediction for one sample fully matches the gold. 0 otherwise.
    print('Subset accuracy: {0}'.format(
        sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))

    # Hamming loss (smaller is better)
    # $$ \text{HammingLoss}(x_i, y_i) = \frac{1}{|D|} \sum_{i=1}^{|D|} \frac{xor(x_i, y_i)}{|L|}, $$
    # where
    #  - \\(|D|\\) is the number of samples
    #  - \\(|L|\\) is the number of labels
    #  - \\(y_i\\) is the ground truth
    #  - \\(x_i\\)  is the prediction.
    # 0.416666666667 (= (1+0+3+1) / (3*4) )
    print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred)))
