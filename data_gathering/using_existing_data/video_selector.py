import gzip
import json_lines
import json
from tqdm import tqdm

from os import listdir


def load_videos(path):
    videos = []
    try:
        for file in [f for f in listdir(path)]:
            with gzip.open(path + '/' + file, "rb") as f:
                for item in json_lines.reader(f):
                    videos.append(item)
    except(FileNotFoundError):
        print("no videos for channel: "+path)
        return []
    return videos


def getViewcount(video):
    return video['Statistics']['ViewCount']

def containsVideo(list, newVideo):
    for video in list:
        if video['VideoId'] == newVideo['VideoId']:
            return True
    return False

def select_top_n_videos(video_list, n):
    video_list.sort(key=getViewcount, reverse=True)
    if len(video_list) == 0:
        return []
    videos = []
    for video in video_list:
        if len(videos) == 3:
            return videos
        else:
            if not containsVideo(videos, video):
                videos.append(video)
    return videos # for the case that there are less than 3 videos available for that channel


def readChannels(path):
    with gzip.open(path, 'rb') as f:  # opening file in binary(rb) mode
        jsons = {}
        for item in json_lines.reader(f):
            jsons[item['ChannelId']] = item
    return jsons


channelData = readChannels('data/channelData.jsonl.gz')

for channel in tqdm(channelData.keys()):
    channelData[channel]['top3videos'] = select_top_n_videos(load_videos('data/videos/'+channel), 3)

with open("output\channelDataWithTopVideos.json", "w") as f:
    json.dump(channelData, f, indent=4)
