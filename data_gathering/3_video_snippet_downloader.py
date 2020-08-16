import json

from googleapiclient import errors
from tqdm import tqdm
import os

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from dotenv import load_dotenv

from utils.data_IO import read_channels



class VideoSnippetAPIScraper():
    def __init__(self, api_keys, api_number):
        self.api_keys = api_keys
        self.api_number = api_number
        self.youtube_client = googleapiclient.discovery.build(
            'youtube', "v3", developerKey=api_keys[api_number])

    def getVideoSnippet(self, videoId):
        # Disable OAuthlib's HTTPS verification when running locally.
        # *DO NOT* leave this option enabled in production.

        request = self.youtube_client.videos().list(
            part="snippet,contentDetails,statistics",
            id=videoId
        )
        response = request.execute()
        if len(response['items'])==0:
            print(f"{videoId} was not found - aborting")
            return {'VideoId': None}
        # merge the most important info in one dict
        relevant_info = {'VideoId': videoId}
        relevant_info.update(response['items'][0]['snippet'])
        relevant_info.update({'duration': response['items'][0]['contentDetails']['duration']})
        relevant_info.update({'Statistics': response['items'][0]['statistics']})
        return relevant_info

    def gather_info(self, videoId):
        try: #AIzaSyBmeEN2lltj5ZeSeS6gaQTgEIgJTc59ppU
            return self.getVideoSnippet(videoId)
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

if __name__ == "__main__":
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    load_dotenv()
    api_keys = os.getenv('API_KEYS').split(',')
    api_number = 0

    api_scraper = VideoSnippetAPIScraper(api_keys, api_number)

    channelData = read_channels('../output/unlabeled_data/U_channelDataWithScrapedTopVideos.json')
    for channel in tqdm(channelData.keys()):
        videoIDs = [video for video in channelData[channel]['top3videos'] if not None]
        video_snippets = []
        for videoId in videoIDs:
            video_snippets.append(api_scraper.gather_info(videoId))
        channelData[channel]['top3videos'] = video_snippets

    with open("../output/unlabeled_data/U_channelDataWithTopVideos.json", "w") as f:
        json.dump(channelData, f, indent=4)


