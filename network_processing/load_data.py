import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from utils.data_IO import read_channels, write_df
import time


def get_connection_dataframe(content='related_channels', only_known=True, labeled=True, binary=True):
    start_time = time.time()
    if content == 'cross_comments':
        if binary:
            channels = read_channels('../output/final_data/cross_comments.json')
        else:
            channels = read_channels('../output/final_data/cross_comments_multilabel.json')
    else:
        if binary:
            channels = read_channels('../output/final_data/P_channelData.json')
            channels_NP = read_channels('../output/final_data/NP_channelData.json')
            channels.update(channels_NP)
            channels_unlabeled = read_channels('../output/final_data/U_channelData.json')
        else:
            channels = read_channels('../output/final_data/P_channelData_multilabel.json')
            channels_unlabeled = read_channels("../output/final_data/U_channelData_multilabel.json")
        channels.update(channels_unlabeled)
    labeled_index = len([channel for channel in channels.values() if channel['SoftTags'][0] != 'UNLABELED'])
    all_channels = list(channels.keys())

    connections_df = pd.DataFrame({'ChannelId': [],
                                   'ChannelTitle': [],
                                   'label': [],
                                   'connections': []})

    # subs = []
    for key, value in channels.items():
        if only_known:
            tmp = pd.DataFrame({'ChannelId': [key],
                                'ChannelTitle': [value['ChannelTitle']],
                                'label': [determine_label(labels=value['SoftTags'], binary=binary)],
                                'connections': [
                                    [affiliate for affiliate in value[content] if affiliate in all_channels]]})
        else:
            tmp = pd.DataFrame({'ChannelId': [key],
                                'ChannelTitle': [value['ChannelTitle']],
                                'label': [determine_label(labels=value['SoftTags'], binary=binary)],
                                'connections': [value[content]]})

        if len(tmp['connections'][0]) == 0:
            tmp.at[0, 'connections'] = ['None']
        # else:
        #     subs.append(len(tmp['connections'][0]))
        connections_df = connections_df.append(tmp)
    # print('Mean relations: %f  | Median Relations: %f' % (np.mean(subs), np.median(subs)))

    tmp = connections_df['connections'].apply(Counter)
    tmp = tmp.reset_index().drop(columns='index')
    encoded_connections_df = pd.DataFrame.from_records(tmp['connections']).fillna(value=0).drop(columns='None')
    encoded_connections_df['label'] = list(connections_df['label'])
    encoded_connections_df['id'] = list(connections_df['ChannelId'])
    encoded_connections_df['title'] = list(connections_df['ChannelTitle'])
    # encoded_connections_df.reset_index().drop(columns='index')
    end_time = time.time()-start_time
    print(f"finishing:  {content} after {end_time} second" )

    if labeled:
        if binary: # We want political and non political data in the dataset
            # return all labeled channels
            write_df(encoded_connections_df.iloc[:labeled_index], content=content, binary=binary, labeled=labeled)
            return encoded_connections_df.iloc[:labeled_index]
        elif content == 'cross_comments' and not labeled:  # We only want the political data
            # return all labeled political channels
            # unfortunately there is no nice way to index non-scalar values
            for index, row in encoded_connections_df.iterrows():
                if row['label'] == ['UNLABELED']:
                    write_df(encoded_connections_df.iloc[:index], content=content, binary=binary,
                             labeled=labeled)

                    return encoded_connections_df.iloc[:index]

        else:
            # in this case the whole dataset only contains political and unlabeled channels
            write_df(encoded_connections_df.iloc[:labeled_index], content=content, binary=binary, labeled=labeled)

            return encoded_connections_df.iloc[:labeled_index]
    else:
        write_df(encoded_connections_df.iloc[labeled_index:].reset_index().drop(columns='index'), content=content, binary=binary, labeled=labeled)

        return encoded_connections_df.iloc[labeled_index:].reset_index().drop(columns='index')

def determine_label(labels, binary=True):
    if binary:
        return '__label__non-political' if labels[0] == 'Non-Political' else \
            '__label__unlabeled' if labels[0] == 'UNLABELED' else \
                '__label__political'
    else:
        return ["__label__" + label for label in labels]



def create_merged_dataframe(binary: True):
    if binary:
        channels = read_channels('../output/final_data/P_channelData.json')
        channels_NP = read_channels('../output/final_data/NP_channelData.json')
        channels.update(channels_NP)
        channels_unlabeled = read_channels('../output/final_data/U_channelData.json')

    else:

        channels = read_channels('../output/final_data/P_channelData_multilabel.json')
        channels_unlabeled = read_channels('../output/final_data/U_channelData_multilabel.json')
    channels.update(channels_unlabeled)
    all_channels = list(channels.keys())
    cross_comments = {}
    for id, channel in tqdm(channels.items()):
        authors = {}
        for author in all_channels:
            authors[author] = 0
        for video in channel['comments'].values():
            if video is not None:
                for comment in video:
                    if comment['authorChannelId'] is not None:
                        author = comment['authorChannelId']['value']
                        if comment['authorChannelId']['value'] in all_channels:
                            if author != channel['ChannelId']:
                                authors[author] = authors[author] + 1
        cross_comments[id] = {}
        cross_comments[id]['cross_comments'] = [author for author in authors.keys() if authors[author] > 0]
        cross_comments[id]['SoftTags'] = channels[id]['SoftTags']
        cross_comments[id]['ChannelId']  = channels[id]['ChannelId']
        cross_comments[id]['ChannelTitle']  = channels[id]['ChannelTitle']

        # channels[id]['comment_mentions'] = [{channel: mentions[channel]} for channel in mentions.keys() if mentions[channel]>0]
    print('dumping')
    if binary:
        output_path = "../output/final_data/cross_comments.json"
    else:
        output_path = "../output/final_data/cross_comments_multilabel.json"
    with open(output_path, "w") as f:
        json.dump(cross_comments, f, indent=4)
    print('ok')


if __name__ == "__main__":
    test=get_connection_dataframe(content='subscriptions', only_known=True, binary=False)
    create_merged_dataframe(binary=False)
