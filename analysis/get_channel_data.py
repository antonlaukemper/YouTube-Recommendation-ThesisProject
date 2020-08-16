import json
from tqdm import tqdm
import os

import googleapiclient.discovery
import googleapiclient.errors

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]



def get_channel_snippets(channel_ids):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = 'AIzaSyBrjY7N1vhWtr1G9vATUpFybRTCImRayK4'

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)


    channel_data = {}
    for index, channel_id in tqdm(enumerate(channel_ids)):
        if index == 1000:
            youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey='AIzaSyB6gAvvRl8DIJw5OcLZVTkBkilh7L1GEnE')
        request = youtube.channels().list(
            part="snippet,brandingSettings,statistics",
            id=channel_id
        )
        response = request.execute()
        try:
            channel_info = response['items'][0]
            # merge the most important info in one dict
            # todo: numbers should be int not string
            relevant_info = dict(ChannelTitle=channel_info['snippet']['title'], MainChannelId=None,
                                 Description=channel_info['snippet']['description'], LogoUrl=None, Relevance=1.0,
                                 LR=None, Subs=int(channel_info['statistics']['subscriberCount']),
                                 ChannelViews=int(channel_info['statistics']['viewCount']),
                                 Country=channel_info['snippet']['country'] if 'country' in channel_info[
                                     'snippet'].keys() else None,
                                 related_channels=channel_info['brandingSettings']['channel'][
                                     'featuredChannelsUrls'] if 'featuredChannelsUrls' in
                                                                channel_info['brandingSettings'][
                                                                    'channel'].keys() else [],
                                 Tags=(channel_info['brandingSettings']['channel']['keywords']) if 'keywords' in channel_info[
                                     'brandingSettings']['channel'] else [])
            channel_data[channel_id] = relevant_info
        except Exception as e:
            print("no information on %s" % channel_id)
            print(e)
            print(e.__doc__)

    with open("related_channels.json", "w") as f:
        json.dump(channel_data, f, indent=4)
    return channel_data

