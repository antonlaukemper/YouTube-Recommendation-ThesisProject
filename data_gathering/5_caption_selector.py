import gzip
import json_lines
import json
from tqdm import tqdm
from os import listdir

from utils.data_IO import read_channels


def findCaptions(path, videoIDs):
    captions = {}
    try:
        for file in [f for f in listdir(path)]:
            with gzip.open(path + '/' + file, "rb") as f:
                for captionData in json_lines.reader(f):
                # captionData = json.loads(f.read().decode("ascii"))
                    if captionData['VideoId'] in videoIDs:
                        # The data comes from two different sources,
                        if captionData['Info'] != 'downloaded':
                            # this routine is for captions from the azure database
                            captionLinesData = captionData["Captions"]
                            captionLines = [line["Text"] for line in captionLinesData]
                            full_text = "\n".join(captionLines)
                            captions[captionData['VideoId']] = full_text
                            videoIDs.remove(captionData['VideoId'])
                        else:
                            captions[captionData['VideoId']] = captionData['Captions']
                            videoIDs.remove(captionData['VideoId'])
    except(FileNotFoundError):
        print("no videos for channel: " + path)
        return {}
    return captions

if __name__ == '__main__':
    channelData = read_channels('../output/unlabeled_data/U_channelDataWithTopVideos.json')

    for channel in tqdm(channelData.keys()):
        videoIDs = [video['VideoId'] for video in channelData[channel]['top3videos'] if not None]
        channelData[channel]['captions'] = findCaptions('data/Captions/'+channel, videoIDs)

    with open("../output/unlabeled_data/U_channelDataWithCaptions.json", "w") as f:
        json.dump(channelData, f, indent=4)