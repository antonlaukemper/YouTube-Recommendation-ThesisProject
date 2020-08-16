import pandas as pd
import fasttext
import pickle
from network_processing import load_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_data(training=True):
    if training:
        captions = classify_text("../text_processing/fasttext_data/captions/FINAL_ENSEMBLE_TRAINING_DATA.txt",
                                 '../text_processing/models/captions/best_captions_model.bin',
                                 'captions')
        comments = classify_text("../text_processing/fasttext_data/comments/FINAL_ENSEMBLE_TRAINING_DATA.txt",
                                 '../text_processing/models/comments/best_comments_model.bin',
                                 'comments')
        snippets = classify_text("../text_processing/fasttext_data/snippets/FINAL_ENSEMBLE_TRAINING_DATA.txt",
                                 '../text_processing/models/snippets/best_snippets_model.bin',
                                 'snippets')

    else:
        captions = classify_text("../text_processing/fasttext_data/captions/FINAL_ENSEMBLE_TESTING_DATA.txt",
                                 '../text_processing/models/captions/best_captions_model.bin',
                                 'captions')
        comments = classify_text("../text_processing/fasttext_data/comments/FINAL_ENSEMBLE_TESTING_DATA.txt",
                                 '../text_processing/models/comments/best_comments_model.bin',
                                 'comments')
        snippets = classify_text("../text_processing/fasttext_data/snippets/FINAL_ENSEMBLE_TESTING_DATA.txt",
                                 '../text_processing/models/snippets/best_snippets_model.bin',
                                 'snippets')
    affiliations = classify_relations('../network_processing/models/affiliations/best_model.pl', training=training,
                                      mode='related_channels',
                                      only_known=False)
    subscriptions = classify_relations('../network_processing/models/subscriptions/best_model.pl',
                                       training=training,
                                       mode='subscriptions',
                                       only_known=True)
    cross_channel_comments = classify_relations('../network_processing/models/cross-comments/best_model.pl',
                                                training=training,
                                                mode='cross_comments')

    relation_result = affiliations.join(subscriptions).join(cross_channel_comments).reset_index()
    result = captions.set_index(['id', 'label']).join(comments.set_index(['id', 'label'])).join(
        snippets.set_index(['id', 'label'])).join(relation_result.set_index(['id', 'label']))
    result.reset_index(inplace=True)
    return result

def classify_text(data_path, model_path, content):
    with open(data_path, 'r') as data:
        model = fasttext.load_model(model_path)

        row_dict_list = []
        for line in data.readlines():
            parts = line.split(' ')
            label = parts[1]
            text = ' '.join(parts[2:])
            # join adds a carriage return in the end
            if text == '\n':
                prob_political = 0.5
            else:
                predicted_label, probability = model.predict(text[:-1])
                prob_political = probability[0] if predicted_label[0] == '__label__political' else 1 - probability[0]

            row_dict = {'id': parts[0], 'label': label, content: text, content+'_pred': prob_political}
            row_dict_list.append(row_dict)
        result = pd.DataFrame(row_dict_list)
    return result


def classify_relations(model_path, training=True, mode='related_channels', only_known=True):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        data_tuple = get_training_test_split(mode, only_known)
        if training:
            data = data_tuple[0]
        else:
            data = data_tuple[1]
        row_dict_list = []
        for index, row in tqdm(data.iterrows()):
            id = row['id']
            title = row['title']
            label = row['label']
            row_data = row.drop('label').drop('id').drop('title')
            prob_political = model.decision_function([row_data])[0]
            row_dict = {'id': id,
                        'title': title,
                        'label': '__label__non-political' if label == 'Non-Political' else '__label__political',
                        mode+'_pred': prob_political}
            row_dict_list.append(row_dict)
    results = pd.DataFrame(row_dict_list).set_index(['id', 'title', 'label'])
    return results







# for the relation data
def get_training_test_split(mode='related_channels', only_known=True):
    full_data = load_data.get_connection_dataframe(mode, only_known)
    full_data = full_data.sample(frac=1, random_state=42)
    return train_test_split(full_data, test_size=0.1, shuffle=False)

