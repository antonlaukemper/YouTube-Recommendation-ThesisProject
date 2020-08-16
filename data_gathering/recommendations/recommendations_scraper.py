import datetime
import time
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options  # for headless mode
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementNotInteractableException
import os
import json
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import asyncio
import discord_bot

DRIVER_PATH = os.path.join('C:\\', 'Program Files (x86)', 'Google', 'Chrome', 'chromedriver.exe')
options = Options()
options.headless = False
options.add_argument("--window-size=1920,1200")
capabilities = DesiredCapabilities.CHROME.copy()
capabilities['acceptSslCerts'] = True
capabilities['acceptInsecureCerts'] = True

driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH, desired_capabilities=capabilities)


class Crawler:
    # this class has three modes
    #   1. watch videos
    #   2. get recommendations from videos
    #   3. get recommendations from the feed
    #   4. delete video(s) from history
    def __init__(self):
        self._video_infos = {}
        self.wait = WebDriverWait(driver, 10)
        self.azure_connection = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.blob_service_client = BlobServiceClient.from_connection_string(self.azure_connection)
        self.container_name = "seleniumtest"

    def login(self, user, password, tel_number):
        self.user = user.replace('@gmail.com', '')
        # todo sometimes the phone number confirmation pops up, it has to be dealt with
        driver.get(
            'https://accounts.google.com/signin/v2/identifier?service=youtube&uilel=3&passive=true&continue=https%3A%2F%2Fwww.youtube.com%2Fsignin%3Faction_handle_signin%3Dtrue%26app%3Ddesktop%26hl%3Dde%26next%3D%252F&hl=en&ec=65620&flowName=GlifWebSignIn&flowEntry=ServiceLogin')
        # # headless mode
        # login_email = driver.find_element_by_id('Email').send_keys(user)
        # next_button = driver.find_element_by_id('next').click()
        # password = WebDriverWait(driver, 5).until(
        #     EC.element_to_be_clickable((By.XPATH, "//input[@name='Passwd']"))
        # ).send_keys(password)
        # password_next_button = driver.find_element_by_id('signIn').click()

        # non-headless
        login_email = driver.find_element_by_id('identifierId').send_keys(user)
        next_button = driver.find_element_by_id('identifierNext').click()
        password = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@name='password']"))
        ).send_keys(password)
        password_next_button = driver.find_element_by_id('passwordNext').click()
        time.sleep(1)
        phone_input = driver.find_elements_by_xpath("//input[@type='tel']")
        revalidation = driver.find_elements_by_xpath("//*[@data-sendmethod='SMS']")

        if len(phone_input) != 0:
            # We need to validate for the first time
            phone_input[0].send_keys(tel_number)
            tel_next_button = driver.find_element_by_id('idvanyphonecollectNext').click()
            code_bot = discord_bot.DiscordBot()
            code = code_bot.get_code()
            code_input = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//input[@type='tel']"))
            ).send_keys(code)
            code_next_button = driver.find_element_by_id('idvanyphoneverifyNext').click()
        elif len(revalidation) != 0:
            # revalidation
            revalidation[0].click()  # select sms option
            code_bot = discord_bot.DiscordBot()
            code = code_bot.get_code()
            code_input = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//input[@type='tel']"))
            ).send_keys(code)
            code_next_button = driver.find_element_by_id('idvanyphoneverifyNext').click()


    def get_video_features(self, id, recommendations, personalized_count):
        filename = 'output/recommendations/' + self.user + '_' + id + '_' + str(datetime.datetime.today()).replace(':',
                                                                                                                   '-').replace(
            ' ', '_') + '.json'
        video_info = {
            'id': id,
            'title': self.wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "#container > h1 > yt-formatted-string"))).text,
            'channel': self.wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR,
                 "ytd-channel-name.ytd-video-owner-renderer > div:nth-child(1) > "
                 "div:nth-child(1)"))).text,
            'channel_id': self.wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "#text > a"))).get_attribute('href').strip(
                'https://www.youtube.com/channel/'),
            'recommendations': recommendations,
            'personalization_count': personalized_count
        }
        with open(filename, 'w') as file:
            json.dump(video_info, file, indent=4)
        with open(filename, 'rb') as file:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=filename)
            blob_client.upload_blob(file)

    def get_element_attributes(self, element):
        # this is just a helper method
        return driver.execute_script(
            'var items = {}; for (index = 0; index < arguments[0].attributes.length; ++index) { items[arguments[0].attributes[index].name] = arguments[0].attributes[index].value }; return items;',
            element)

    def get_recommendations_for_video(self, source):
        driver.get("https://www.youtube.com/watch?v=" + source)

        # this is the list of elements from the recommendation sidebar
        # it does not always load all recommendations at the same time, therefore the loop
        all_recs = []
        while len(all_recs) < 19:
            all_recs = WebDriverWait(driver, 10).until(
                EC.visibility_of_all_elements_located((By.XPATH, '//*[@id="dismissable"]/div/div[1]/a'))
            )

        recos = {}
        personalized_counter = 0
        for i in all_recs:
            personalized = 'Recommended for you' in i.text
            if personalized:
                personalized_counter += 1
            # take the link and remove everything except for the id of the video that the link leads to
            recommendation_id = i.get_attribute('href').replace('https://www.youtube.com/watch?v=', '')
            recos[recommendation_id] = {'id': recommendation_id, 'personalized': personalized}
        # store the information about the current video plus the corresponding recommendations
        self.get_video_features(source, recos, personalized_counter)
        # return a number of recommendations up to the branching factor
        return recos

    def delete_last_video_from_history(self):
        driver.get('https://www.youtube.com/feed/history')

        first_video = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH,
                                            '//*[@id="video-title"]'))
        )
        # the link might contain a time stamp so we we need to use split to only get the video id
        id = first_video.get_attribute('href').replace('https://www.youtube.com/watch?v=', '').split('&')[0]
        # todo: maybe replace the full xpath because its ugl, but it works
        last_video_delete_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH,
                                            '/html/body/ytd-app/div/ytd-page-manager/ytd-browse/ytd-two-column-browse'
                                            '-results-renderer/div[1]/ytd-section-list-renderer/div['
                                            '2]/ytd-item-section-renderer[1]/div[3]/ytd-video-renderer/div['
                                            '1]/div/div['
                                            '1]/div/div/ytd-menu-renderer/div/ytd-button-renderer/a/yt-icon-button'
                                            '/button'))
        ).click()

    def delete_history(self):
        driver.get('https://www.youtube.com/feed/history')
        history_delete_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH,
                                            '/html/body/ytd-app/div/ytd-page-manager/ytd-browse/ytd-two-column-browse'
                                            '-results-renderer/div['
                                            '2]/ytd-browse-feed-actions-renderer/div/ytd-button-renderer['
                                            '1]/a/paper-button/yt-formatted-string'))
        ).click()

        confirm_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH,
                                            '/html/body/ytd-app/ytd-popup-container/paper-dialog/yt-confirm-dialog'
                                            '-renderer/div[2]/div/yt-button-renderer['
                                            '2]/a/paper-button/yt-formatted-string'))
        ).click()

    def shutdown(self):
        driver.quit()

    async def watch_video(self, videoId: str, main_window, current_window):
        driver.switch_to.window(current_window)
        driver.get("https://www.youtube.com/watch?v=" + videoId)
        # wait until video is loaded
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH,
                                            '//*[@class="ytp-play-button ytp-button"]'))
        )
        # unfortunately the ad loads slower than the player
        time.sleep(1)
        # First we need to detect whether an ad is playing
        if len(driver.find_elements_by_xpath(
                "//*[@class='ytp-ad-preview-container countdown-next-to-thumbnail']")) != 0:
            print(driver.find_element_by_xpath(
                "//*[@class='ytp-ad-button ytp-ad-visit-advertiser-button ytp-ad-button-link']").text)
            time.sleep(6)
            driver.find_element_by_xpath("//*[@class='ytp-ad-skip-button ytp-button']").click()
        time.sleep(1)
        # second ad is possible
        if len(driver.find_elements_by_xpath(
                "//*[@class='ytp-ad-preview-container countdown-next-to-thumbnail']")) != 0:
            print(driver.find_element_by_xpath(
                "//*[@class='ytp-ad-button ytp-ad-visit-advertiser-button ytp-ad-button-link']").text)
            time.sleep(6)
            driver.find_element_by_xpath("//*[@class='ytp-ad-skip-button ytp-button']").click()

        print("works in non-headless mode")
        duration = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'ytp-time-duration'))).text
        # start the video
        # to make sure that every video is watched long enough
        # todo: this will be a problem if an ad is longer than the respective watch time?
        await asyncio.sleep(10)
        driver.switch_to.window(current_window)
        print('watched %s for %f seconds' % (videoId, 10))
        driver.close()
        driver.switch_to.window(main_window)
        # # todo: replace this with proper logging

        # skip button
        # /html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[1]/div/div[1]/div/div/div/ytd-player/div/div/div[15]/div/div[3]/div/div[2]/span/button/span

    async def watch_videos(self, videos):
        tasks = []
        main_window = driver.window_handles[-1]
        for video in videos:
            driver.execute_script("window.open('');")
            new_tab = driver.window_handles[-1]
            tasks.append(
                self.watch_video(video, main_window, new_tab)
            )
        await asyncio.gather(*tasks)

    def scan_feed(self):
        driver.get("https://www.youtube.com")

        try:
            extra_content = WebDriverWait(driver, 10).until(  # Schließen
                EC.presence_of_all_elements_located((By.XPATH, '//*[@aria-label="Close"]')))
            for button in extra_content:
                try:
                    button.click()
                except ElementNotInteractableException:
                    print(self.get_element_attributes(button))
        except TimeoutException:
            print("no extra covid information")

        themed_content = None
        try:
            themed_content = WebDriverWait(driver, 10).until(  # Schließen
                EC.presence_of_all_elements_located((By.XPATH, '//*[@aria-label="Not interested"]')))
        except TimeoutException:
            print("No themed content")

        if themed_content is not None:
            for button in themed_content:
                button.click()

        time.sleep(1)
        all_videos = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, '//*[@id="video-title-link"]'))
        )
        feed_videos = []
        personalized_counter = 0
        for video in all_videos:
            personalized_counter += 1
            # take the link and remove everything except for the id of the video that the link leads to
            recommendation_id = video.get_attribute('href').replace('https://www.youtube.com/watch?v=', '')
            print(personalized_counter, video.get_attribute('title'))
            feed_videos.append(recommendation_id)
        print(len(feed_videos))


if __name__ == "__main__":
    starting_time = time.time()
    crawler = Crawler()
    crawler.login('colabseleniumexperiment@gmail.com', 'colabseleniumexperiment1!', '+4915734929699')
    # crawler.login('this.is.just.for.research.purposes@gmail.com', 'youtube_radicalization1', '+4915734929699')
    asyncio.run(crawler.watch_videos(['byva0hOj8CU', 'yQ2-yVXFMeE', '7U-RbOKanYs', 'O-3Mlj3MQ_Q']))

    crawler.delete_last_video_from_history()
    # main_window = driver.window_handles[-1]
    # driver.execute_script("window.open('');")
    # new_tab = driver.window_handles[-1]
    # newest_tab = driver.window_handles[-1]
    # asyncio.run(crawler.watch_video('yQ2-yVXFMeE', main_window, newest_tab))
    # asyncio.run(crawler.watch_videos(['byva0hOj8CU', 'yQ2-yVXFMeE', '7U-RbOKanYs', 'O-3Mlj3MQ_Q']))
    # asyncio.run(crawler.watch_videos(['CH50zuS8DD0']))

    print("woop")
    # crawler.get_recommendations_for_video('reCMB81A70k')
    # crawler.shutdown()

    # crawler.login('this.is.just.for.research.purposes@gmail.com', 'youtube_radicalization1')
    # search_results = crawler.get_n_search_results(search_term="Climate Change")
    # for i in range(len(search_results)):
    #     crawler.get_n_recommendations(seed=search_results[i], depth=5)
    #     with open('data/video-infos-' + "Climate_Change" + '-' + "depth_5_branch_5_part_" + str(i) + '.json', 'w') as fp:
    #         json.dump(crawler._video_infos, fp)
    # with open('data/video-infos-' + "Climate_Change" + '-' + "depth_5_branch_5" + '.json', 'w') as fp:
    #     json.dump(crawler._video_infos, fp)
    # crawler.shutdown()
    # end_time = time.time()
    # print("total time is %f s or %f min" % (end_time-starting_time, ((end_time-starting_time)/60)))
