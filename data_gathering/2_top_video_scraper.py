import time
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options  # for headless mode
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import os
import json
import gzip
import json_lines
from tqdm import tqdm

DRIVER_PATH = os.path.join('C:\\', 'Program Files (x86)', 'Google', 'Chrome', 'chromedriver.exe')
options = Options()
options.headless = True
options.add_argument("--window-size=1920,1200")
capabilities = DesiredCapabilities.CHROME.copy()
capabilities['acceptSslCerts'] = True
capabilities['acceptInsecureCerts'] = True


def get_top_n_videos(driver, channelid, n):
    # if channel is deleted this simply returns an empty list

    driver.get('https://www.youtube.com/channel/' + channelid + '/videos?view=0&sort=p&flow=grid')


    top_videos = driver.find_elements_by_xpath('//*[@id="video-title"]')
    if len(top_videos )==0:
        print(f"{channel} is not available")
        return None
    if top_videos[0].get_attribute('href') is None:
        print(f"{channel} has no videos")
        return None
    else:
        video_ids = [video.get_attribute('href').replace('https://www.youtube.com/watch?v=', '') for video in top_videos]
    return video_ids[:n]


def read_channels(path):
    with gzip.open(path, 'rb') as f:  # opening file in binary(rb) mode
        jsons = {}
        for item in json_lines.reader(f):
            jsons[item['ChannelId']] = item
    return jsons


def read_non_political_channels(path):
    with open(path, 'rb') as f:  # opening file in binary(rb) mode
        channelData = json.load(f)
    return channelData


if __name__ == '__main__':
    driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH, desired_capabilities=capabilities)
    channelData = read_non_political_channels('../output/unlabeled_data/U_channelData.json')
    # # 483
    # for index, channel in enumerate(channelData.keys()):
    #     print(index,channel)

    for channel in tqdm(channelData.keys()):
        channelData[channel]['top3videos'] = get_top_n_videos(driver, channel, 3)

    with open("../output/unlabeled_data/U_channelDataWithScrapedTopVideos.json", "w") as f:
        json.dump(channelData, f, indent=4)
