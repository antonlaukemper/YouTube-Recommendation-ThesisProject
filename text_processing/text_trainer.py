import fasttext
import os
from sklearn.model_selection import train_test_split
from text_processing.load_data import create_text_df, remove_missing_data
from utils.metrics import calculate_metrics
from utils.data_IO import write_to_file, load_df
import pandas as pd
import jsonlines



class TextTrainer():

    def __init__(self, mode='captions', epochs=300):
        self.test_path = f'fasttext_data/{mode}/FINAL_ENSEMBLE_TESTING_DATA.txt'
        self.evaluation_path = f'fasttext_data/{mode}/FINAL_EVALUATION_DATA.txt'
        self.training_path = f'fasttext_data/{mode}/TRAINING_DATA.txt'
        self.output_path = f'models/{mode}/best_{mode}_model.bin'
        self.mode = mode
        self.epoch = epochs #30 if mode == 'captions' else 55 if mode == 'comments' else 50

    def get_data(self, initial=True, binary=True, random_state=42, training_fraction=0.9, serialized=True):
        # training_fraction is the fraction of training data we take from the available training data
        if serialized:
            data = load_df(content=self.mode, labeled=initial, binary=binary)
        else:
            data = create_text_df(mode=self.mode, labeled=initial, binary=binary)
        if initial:
            data = data.sample(frac=1, random_state=42)
            # these are the dataframes that will be used for the ensemble
            training, testing = train_test_split(data, test_size=0.15, shuffle=False)
            write_to_file(testing, self.test_path)

            # to be able to make a statistical statement about the performance of the classifier,
            # I sample from the training set 90% of the data
            training = training.sample(frac=training_fraction, random_state=random_state)

            # todo: the evaluation of the individual classifier is not checked on the sampled data. does it matter?
            # to test how well the model performs, missing data has to be removed and we split again
            data_for_eval = remove_missing_data(data)
            training_eval, testing_eval = train_test_split(data_for_eval, test_size=0.15, shuffle=False)
            write_to_file(testing_eval, self.evaluation_path)
            return [training, testing]

        else:
            # unlabeled data
            data = data.sample(frac=training_fraction, random_state=random_state)
            return data

    def get_model(self, data, multilabel=False, **kwargs):
        data_for_training = remove_missing_data(data)
        write_to_file(data_for_training, self.training_path)
        if multilabel:
            model = fasttext.train_supervised(self.training_path, epoch=self.epoch, loss='ova', verbose=3, **kwargs)
        else:
            model = fasttext.train_supervised(self.training_path, epoch=self.epoch, verbose=0)
        # results = calculate_metrics(self.evaluation_path, model)
        # print(results)
        if not os.path.isdir(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))
        model.save_model(self.output_path)
        return model


    def get_captions(self):
        row_dict_list = []
        with jsonlines.open("../output/unlabeled_data/class_captions.jsonl") as file:
            for json in file:
                row_dict = {'id': json['CHANNEL_ID'], 'label': 'UNLABELED', 'text': json['CAPTION']}
                row_dict_list.append(row_dict)
        return pd.DataFrame(row_dict_list)

    def get_comments(self):
        row_dict_list = []
        with jsonlines.open("../output/unlabeled_data/class_comments.jsonl") as file:
            for json in file:
                row_dict = {'id': json['CHANNEL_ID'], 'label': 'UNLABELED', 'text': json['CAPTION']}
                row_dict_list.append(row_dict)
        return pd.DataFrame(row_dict_list)

    def get_snippets(self):
        row_dict_list = []
        with jsonlines.open("../output/unlabeled_data/class_videos.jsonl") as file:
            with jsonlines.open("../output/unlabeled_data/class_channels.jsonl") as channel_file:
                for video_json, channel_json in zip(file, channel_file):
                    # this is only for one file. todo: change this for multiple videos
                    if video_json['CHANNEL_TITLE'] is not None:
                        channel_title = video_json['CHANNEL_TITLE']
                    else:
                        channel_title = ''
                    if video_json['VIDEO_TITLE'] is not None:
                        video_title = video_json['VIDEO_TITLE']
                    else:
                        video_title = ''
                    if channel_json['CHANNEL_DECRIPTION'] is not None:
                        channel_descr = channel_json['CHANNEL_DECRIPTION']
                    else:
                        channel_descr = ''
                    if video_json['KEYWORDS'] is not None:
                        tags = video_json['KEYWORDS']
                    else:
                        tags = ''

                    if video_json['DESCRIPTION'] is not None:
                        video_descr = video_json['DESCRIPTION']
                    else:
                        video_descr = ''
                    full_snippet = channel_title \
                                   + channel_descr \
                                   + video_title \
                                   + video_descr \
                                   + tags

                    row_dict = {'id': video_json['CHANNEL_ID'], 'label': 'UNLABELED', 'text': full_snippet}
                    row_dict_list.append(row_dict)
        return pd.DataFrame(row_dict_list)

if __name__ == '__main__':
    os.chdir("C:/Users/Anton Laukemper/Desktop/Uni/MasterThesis/YouTube-Recommendation-Project/multilabel_classification")
    tt = TextTrainer(mode='snippets')
    snippets = tt.get_data(binary=False, serialized=False)
    model = tt.get_model(multilabel=True, data=snippets[0])
    test = model.predict("what the fuck is wrong with this package", k = 3)
    print("t")