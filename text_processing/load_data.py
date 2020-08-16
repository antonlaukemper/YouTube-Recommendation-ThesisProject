from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import fasttext
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer

from utils.data_IO import read_channels, write_to_file, write_df
from utils.metrics import calculate_metrics


def create_caption_data_dict(data, binary=True):
    row_dict_list = []
    for key, value in data.items():
        full_captions = ""
        for caption in value['captions'].values():
            full_captions = (full_captions + caption.lower()) if caption is not None else full_captions
        if binary:
            label = "__label__non-political" if value['SoftTags'][0] == 'Non-Political' \
                else "__label__unlabeled" if value['SoftTags'][0] == 'UNLABELED' \
                else "__label__political"
        else:
            label = ' '.join(["__label__" + tag for tag in value['SoftTags']])
        row_dict = {'id': value['ChannelId'], 'label': label, 'text': full_captions}
        row_dict_list.append(row_dict)

    return row_dict_list


def create_comment_data_dict(data, binary=True):
    row_dict_list = []
    for key, value in data.items():
        full_comments = ""
        for comment_list in value['comments'].values():
            if comment_list is not None:
                for comment in comment_list:
                    full_comments = full_comments + " " + comment['textDisplay'].lower()
        if binary:
            label = "__label__non-political" if value['SoftTags'][0] == 'Non-Political' \
                else "__label__unlabeled" if value['SoftTags'][0] == 'UNLABELED' \
                else "__label__political"
        else:
            label = ' '.join(["__label__" + tag for tag in value['SoftTags']])
        row_dict = {'id': value['ChannelId'], 'label': label, 'text': full_comments}
        row_dict_list.append(row_dict)
    return row_dict_list


def create_snippet_data_dict(data, binary=True):
    row_dict_list = []
    for key, value in data.items():
        full_snippet = value['ChannelTitle'] + " " + value['Description']
        for video in value['top3videos']:
            if 'title' in video.keys():
                title = video['title']
            else:
                title = ''
            if 'description' in video.keys():
                descr = video['description']
            else:
                descr = ''
            if 'tags' in video.keys():
                tags = ' '.join(video['tags'])
            else:
                tags = ''
            full_snippet = full_snippet + " " + title + " " + descr + " " + tags
        if binary:
            label = "__label__non-political" if value['SoftTags'][0] == 'Non-Political' \
                else "__label__unlabeled" if value['SoftTags'][0] == 'UNLABELED' \
                else "__label__political"
        else:
            label = " ".join(["__label__" + tag for tag in value['SoftTags']])
        row_dict = {'id': value['ChannelId'], 'label': label, 'text': full_snippet.lower()}
        row_dict_list.append(row_dict)
    return row_dict_list


def create_text_df(mode='captions', labeled=True, binary=True):
    if labeled:
        if binary:
            data = read_channels("../output/final_data/P_channelData.json")
            data.update(read_channels("../output/final_data/NP_channelData.json"))
        else:
            data = read_channels("../output/final_data/P_channelData_multilabel.json")

    else:
        if binary:
            data = read_channels("../output/final_data/U_channelData.json")
        else:
            data = read_channels("../output/final_data/U_channelData_multilabel.json")

    if mode == 'captions':
        row_dict_list = create_caption_data_dict(data, binary=binary)
    elif mode == 'comments':
        row_dict_list = create_comment_data_dict(data, binary=binary)
    else:  # mode == 'snippets':
        row_dict_list = create_snippet_data_dict(data, binary=binary)
    data = pd.DataFrame(row_dict_list)
    # write_df(data,content=mode, binary=binary, labeled=labeled)
    return data

# happens inplace!
def preprocess(data: pd.DataFrame):
    stop_words = set(stopwords.words('english'))
    stop_words = [re.sub('[^A-Za-z\s]+', '', word) for word in stop_words]
    lemmatizer = WordNetLemmatizer()
    # minimal preprocessing so that we can write to file
    data['text'].replace('\n', ' ', regex=True, inplace=True)
    data['text'].replace('\t', ' ', regex=True, inplace=True)
    # remove urls
    data['text'].replace(
        r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
        '', regex=True, inplace=True)
    # remove html tags
    data['text'].replace('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\\u201c|\\u2019', '', regex=True,
                         inplace=True)
    data['text'].replace('[^A-Za-z\s]+', '', regex=True, inplace=True)  # remove any special symbols
    data['text'].replace(r"(.)\1{2,}", r"\1\1", regex=True, inplace=True)  # spellcheck
    data['text'] = data['text'].apply(lambda x: fasttext.tokenize(x))
    data['text'] = data['text'].apply(lambda x: [token for token in x if not token in stop_words])
    data['text'] = data['text'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])
    print("mean tokens: %f" % data['text'].apply(lambda x: len(x)).mean())
    print("median tokens: %f " % np.median(data['text'].apply(lambda x: len(x))))
    print("min/max: %f" % np.min(data['text'].apply(lambda x: len(x))), np.max(data['text'].apply(lambda x: len(x))))
    print("std: %f" % np.std(data['text'].apply(lambda x: len(x))))
    data['text'] = data['text'].apply(lambda x: " ".join(x))


def remove_missing_data(data):
    return data[data.text != '']


def cross_validate(data, n, metric, **kwargs):
    kf = KFold(n_splits=n, shuffle=True)
    train_metric = []
    test_metric = []

    for train_index, test_index in kf.split(data):
        train, test = data.iloc[train_index, :], data.iloc[test_index, :]
        write_to_file(train, "fasttext_data/captions/training_data.txt")
        write_to_file(test, "fasttext_data/captions/testing_data.txt")
        model = fasttext.train_supervised(input="fasttext_data/captions/training_data.txt", verbose=0, **kwargs)
        train_metric_result = calculate_metrics("fasttext_data/captions/training_data.txt", model)['weighted avg'][
            metric]
        test_metric_result = calculate_metrics("fasttext_data/captions/testing_data.txt", model)['weighted avg'][metric]
        train_metric.append(train_metric_result)
        test_metric.append(test_metric_result)
    print(test_metric)
    print("mean %s: %f" % (metric, np.mean(test_metric)))
    return (np.mean(train_metric), np.mean(test_metric))


def train_and_test(training, testing, output_path, **kwargs):
    model = fasttext.train_supervised(training, **kwargs)
    results = calculate_metrics(testing, model)
    model.save_model(output_path)
    return results


def optimize_hyperparameters(data):
    training, validation = train_test_split(data, test_size=0.2)
    write_to_file(training, "fasttext_data/captions/optimized_training_data.txt")
    write_to_file(validation, "fasttext_data/captions/optimized_validation_data.txt")
    print("starting automatic hyperparameter optimization")
    model = fasttext.train_supervised(input='fasttext_data/captions/optimized_training_data.txt',
                                      autotuneValidationFile='fasttext_data/captions/optimized_validation_data.txt',
                                      autotuneDuration=600,
                                      verbose=3)
    print("finished optimization, saving model")
    model.save_model("models/captions/optimized_model.bin")
    return calculate_metrics("fasttext_data/captions/optimized_validation_data.txt", model)
