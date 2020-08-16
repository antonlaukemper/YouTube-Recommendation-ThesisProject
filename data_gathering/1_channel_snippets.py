import json

from googleapiclient import errors
from tqdm import tqdm
import os
import csv
import jsonlines
import googleapiclient.discovery
from dotenv import load_dotenv

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]


def readChannels(path, unlabeled=False):
    if unlabeled:
        channelData = {}
        with jsonlines.open(path) as file:
            for json in file:
                channelData[json['CHANNEL_ID']] = {
                    "ChannelId": json['CHANNEL_ID'],
                    "SoftTags": ["UNLABELED"]
                }
    else:
        channelData = {}
        with open(path, 'r') as f:  # opening file in binary(rb) mode
            reader = csv.reader(f, delimiter=';', quotechar='|')
            next(reader, None)
            for row in reader:
                id = row[0].replace('http://www.youtube.com/channel/', '')
                channelData[id] = {
                    "ChannelId": id,
                    "SoftTags": ['Non-Political'],
                    "YT-category": row[3]
                }
                # print(', '.join(row))
    return channelData


class ChannelAPIScraper():

    def __init__(self, api_keys, api_number):
        self.api_keys = api_keys
        self.api_number = api_number
        self.youtube_client = googleapiclient.discovery.build(
                    'youtube', "v3", developerKey=api_keys[api_number])

    def get_channel_snippet(self, channel_id):
        # Disable OAuthlib's HTTPS verification when running locally.
        # *DO NOT* leave this option enabled in production.

        request = self.youtube_client.channels().list(
            part="snippet,brandingSettings,statistics",
            id=channel_id
        )
        response = request.execute()
        if 'items' not in response.keys():
            print(f"{channel_id} was not found")
            return {'ChannelTitle': None}
        channel_info = response['items'][0]
        # merge the most important info in one dict
        # todo: numbers should be int not string
        relevant_info = dict(ChannelTitle=channel_info['snippet']['title'], MainChannelId=None,
                             Description=channel_info['snippet']['description'], LogoUrl=None, Relevance=1.0,
                             LR=None, Subs=int(channel_info['statistics']['subscriberCount']),
                             ChannelViews=int(channel_info['statistics']['viewCount']),
                             Country=channel_info['snippet']['country'] if 'country' in channel_info[
                                 'snippet'].keys() else None,
                             # This is done later separately so it is redundant here
                             # RelatedChannels=channel_info['brandingSettings']['channel'][
                             #     'featuredChannelsUrls'] if 'featuredChannelsUrls' in
                             #                                channel_info['brandingSettings'][
                             #                                    'channel'].keys() else [],
                             Tags=(channel_info['brandingSettings']['channel']['keywords']) if 'keywords' in channel_info[
                                 'brandingSettings']['channel'] else [])
        return relevant_info


    def gather_info(self, channelId):
        try:
            return self.get_channel_snippet(channelId)
        except errors.HttpError as http_error:
            if http_error.resp.get('content-type', '').startswith('application/json'):
                print('### Error encountered ###')
                print(json.loads(http_error.content).get('error').get('errors')[0].get('reason'))
                if json.loads(http_error.content).get('error').get('errors')[0].get('reason') == 'quotaExceeded':
                    self.youtube_client = googleapiclient.discovery.build(
                        "youtube", "v3", developerKey=self.api_keys[self.api_number + 1])
                    self.api_number += 1
                    # try again
                    return self.gather_info(channelId)
                else:
                    raise http_error

if __name__ == "__main__":
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    load_dotenv()
    api_keys = os.getenv('API_KEYS').split(',')
    api_number = 0


    # channelData = readChannels('data/non_political_channels.csv')
    # channelData = readChannels('../data/unlabeled_channels.jsonl', unlabeled=True)
    channelData = readChannels('../data/class_channel_ids.jsonl', unlabeled=True)

    api_scraper = ChannelAPIScraper(api_keys, api_number)
    for channel in tqdm(channelData.keys()):
        channelData[channel].update(api_scraper.gather_info(channel))



    ## Note! This file still includes deleted channels, that can be identified by the empty channelTitle
    # I run the missingDataTester.py to remove them
    with open("../output/unlabeled_data/U_channelData.json", "w") as f:
        json.dump(channelData, f, indent=4)
