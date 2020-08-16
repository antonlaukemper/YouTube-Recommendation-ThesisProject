import json

from dotenv import load_dotenv
from googleapiclient import errors
from tqdm import tqdm
import os
import googleapiclient.discovery
import math

from utils.data_IO import read_channels

class RelationAPIScraper():
    def __init__(self, api_keys, api_number):
        self.api_keys = api_keys
        self.api_number = api_number
        self.youtube_client = googleapiclient.discovery.build(
            'youtube', "v3", developerKey=api_keys[api_number])

    def get_related_channels(self, channelId):
        request = self.youtube_client.channels().list(
            part="brandingSettings",
            id=channelId
        )

        try:
            response = request.execute()
            if 'items' not in response.keys():
                print("No Related channels found for %s" % channelId)
                return []
            elif 'featuredChannelsUrls' in response['items'][0]['brandingSettings']['channel'].keys():
                return response['items'][0]['brandingSettings']['channel']['featuredChannelsUrls']
            else:
                print("No Related channels found for %s" % channelId)
                return []
        except errors.HttpError as http_error:
            if json.loads(http_error.content).get('error').get('errors')[0].get('reason') == 'quotaExceeded':
                raise http_error
            else:
                print("Error: No Related channels found for %s" % channelId)
                return []
        except(IndexError):
            print("channel was deleted")
            return []

    def get_subscriptions(self, channelId):
        request = self.youtube_client.subscriptions().list(
            part="snippet",
            channelId=channelId,
            maxResults= 50
        )

        try:
            response = request.execute()
            subscriptions = []
            for item in response['items']:
                subscriptions.append(item['snippet']['resourceId']['channelId'])

            if 'nextPageToken' in response.keys():
                next_page = response['nextPageToken']
                # there are 50 results per page
                # we already got the firs 50 results and now we need the rest
                for step in range(0,math.ceil(response['pageInfo']['totalResults']/50)-1):
                    next_request = self.youtube_client.subscriptions().list(
                        part="snippet",
                        channelId=channelId,
                        maxResults=50,
                        pageToken=next_page
                    )
                    next_response = next_request.execute()
                    if 'nextPageToken' in next_response.keys():
                        next_page = next_response['nextPageToken']
                    for next_item in next_response['items']:
                        subscriptions.append(next_item['snippet']['resourceId']['channelId'])
            return subscriptions
        except errors.HttpError as http_error:
            if json.loads(http_error.content).get('error').get('errors')[0].get('reason') == 'quotaExceeded':
                raise http_error
            else:
                print("Error: no access to subscriptions of %s" % channelId)
                return []

    def gather_info(self, channelId, mode):
        try:
            if mode == 'affiliations':
                return self.get_related_channels(channelId)
            else:
                return self.get_subscriptions(channelId)
        except errors.HttpError as http_error:
            if http_error.resp.get('content-type', '').startswith('application/json'):
                print('### Error encountered ###')
                print(json.loads(http_error.content).get('error').get('errors')[0].get('reason'))
                if json.loads(http_error.content).get('error').get('errors')[0].get('reason') == 'quotaExceeded':
                    self.youtube_client = googleapiclient.discovery.build(
                        "youtube", "v3", developerKey=self.api_keys[self.api_number + 1])
                    self.api_number += 1
                    # try again
                    return self.gather_info(channelId, mode)
                else:
                    raise http_error

def main():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    load_dotenv()
    api_keys = os.getenv('API_KEYS').split(',')
    api_number = 0

    api_scraper = RelationAPIScraper(api_keys, api_number)

    channelData = read_channels('../output/unlabeled_data/U_channelDataWithComments.json')
    for channel in tqdm(channelData.keys()):
        channelData[channel]['subscriptions'] = api_scraper.gather_info(channel, 'subscriptions')
        channelData[channel]['related_channels'] = api_scraper.gather_info(channel, 'affiliations')

    with open('../output/unlabeled_data/U_channelData.json', "w") as f:
        json.dump(channelData, f, indent=4)
if __name__ == "__main__":
    main()