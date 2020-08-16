# -*- coding: utf-8 -*-

import json

from dotenv import load_dotenv
from googleapiclient import errors
from tqdm import tqdm
import os
import googleapiclient.discovery

from utils.data_IO import read_channels


class CommentAPIScraper:
    def __init__(self, api_keys, api_number):
        self.api_keys = api_keys
        self.api_number = api_number
        self.youtube_client = googleapiclient.discovery.build(
            'youtube', "v3", developerKey=api_keys[api_number])

    def gather_info(self, videoId):
        try:
            return self.getCommentsFromYouTube(videoId)
        except errors.HttpError as http_error:
            if http_error.resp.get('content-type', '').startswith('application/json'):
                print('### Error encountered ###')
                print(json.loads(http_error.content).get('error').get('errors')[0].get('reason'))
                if json.loads(http_error.content).get('error').get('errors')[0].get('reason') == 'quotaExceeded':
                    self.youtube_client = googleapiclient.discovery.build(
                        "youtube", "v3", developerKey=self.api_keys[self.api_number + 1])
                    self.api_number += 1
                    # try again
                    return self.gather_info(videoId)
                else:
                    raise http_error

    def getCommentsFromYouTube(self, videoId):
        # Disable OAuthlib's HTTPS verification when running locally.
        # *DO NOT* leave this option enabled in production.

        request = self.youtube_client.commentThreads().list(
            part="snippet",
            maxResults=100,
            order="relevance",
            videoId=videoId)
        commentData = []

        try:
            response = request.execute()
            for commentThread in response['items']:
                topLevelComment = {}
                # some accounts seem to be deleted but their comment still stays
                if len(commentThread['snippet']['topLevelComment']['snippet'][
                           'authorDisplayName']) == 0:
                    topLevelComment['authorDisplayName'] = None
                    topLevelComment['authorChannelId'] = None
                else:
                    topLevelComment['authorDisplayName'] = commentThread['snippet']['topLevelComment']['snippet'][
                        'authorDisplayName']
                    topLevelComment['authorChannelId'] = commentThread['snippet']['topLevelComment']['snippet'][
                        'authorChannelId']
                topLevelComment['textDisplay'] = commentThread['snippet']['topLevelComment']['snippet']['textDisplay']
                commentData.append(topLevelComment)
            return commentData
        except errors.HttpError as other_error:
            if json.loads(other_error.content).get('error').get('errors')[0].get('reason') == 'quotaExceeded':
                raise other_error
            else:
                print("No Comments found for %s" % videoId)
                return None


def main():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    load_dotenv()
    api_keys = os.getenv('API_KEYS').split(',')
    api_number = 0

    api_scraper = CommentAPIScraper(api_keys, api_number)

    channelData = read_channels('../output/unlabeled_data/U_channelDataWithCaptions.json')
    for channel in tqdm(channelData.keys()):
        videoIDs = [video['VideoId'] for video in channelData[channel]['top3videos']]
        commentData = {}
        for videoId in videoIDs:
            commentData[videoId] = api_scraper.gather_info(videoId)
        channelData[channel]['comments'] = commentData

    with open("../output/unlabeled_data/U_channelDataWithComments.json", "w") as f:
        json.dump(channelData, f, indent=4)


if __name__ == "__main__":
    main()
