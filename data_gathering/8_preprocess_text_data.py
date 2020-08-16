import json
import re

import fasttext
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm

from text_processing.text_trainer import TextTrainer
from utils.data_IO import read_channels

def preprocess(data: str):
    if data is None:
        return ""
    data = data.lower()
    stop_words = set(stopwords.words('english'))
    stop_words = [re.sub('[^A-Za-z\s]+', '', word) for word in stop_words]
    lemmatizer = WordNetLemmatizer()
    # minimal preprocessing so that we can write to file
    data = re.sub('\n', ' ', data)
    data = re.sub('\t', ' ', data)
    # remove urls
    data = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", '', data)
    # remove html tags
    data = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\\u201c|\\u2019', '', data)
    data = re.sub('[^A-Za-z\s]+', '', data) # remove any special symbols
    data = re.sub(r"(.)\1{2,}", r"\1\1", data) # spellcheck
    data = fasttext.tokenize(data)
    data = [token for token in data if not token in stop_words]
    data = [lemmatizer.lemmatize(token) for token in data]
    data = " ".join(data)

    return data

def preprocess_dataset(data: dict):
    for channel in tqdm(data.values()):
        for key, caption in channel['captions'].items():
            channel['captions'][key] = preprocess(caption)
        for key, video in channel['comments'].items():
            if video is not None:
                for index, comment in enumerate(video):
                    video[index]['textDisplay'] = preprocess(comment['textDisplay'])

        channel['ChannelTitle'] = preprocess(channel['ChannelTitle'])
        channel['Description'] = preprocess(channel['Description'])
        for video in channel['top3videos']:
            video['title'] = preprocess(video['title'])
            video['description'] = preprocess(video['description'])
            if 'tags' in video.keys():
                for index, tag in enumerate(video['tags']):
                    video['tags'][index] = preprocess(tag)

if __name__ == '__main__':

    # p_data = read_channels("../output/P_channelData.json")
    # np_data = read_channels("../output/NP_channelData.json")
    # u_data = read_channels("../output/unlabeled_data/U_channelData.json")
    p_ml_data = read_channels("../output/P_channelData_multi_label.json")

    #
    # preprocess_dataset(p_data)
    # with open('../output/final_data/P_channelData.json', "w") as f:
    #     json.dump(p_data, f, indent=4)
    #
    # preprocess_dataset(np_data)
    # with open('../output/final_data/NP_channelData.json', "w") as f:
    #     json.dump(np_data, f, indent=4)
    #
    # preprocess_dataset(u_data)
    # with open('../output/final_data/U_channelData.json', "w") as f:
    #     json.dump(u_data, f, indent=4)

    preprocess_dataset(p_ml_data)
    with open('../output/final_data/P_channelData_multilabel.json', "w") as f:
        json.dump(p_ml_data, f, indent=4)
    print('hello')