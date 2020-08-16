from utils.data_IO import read_channels, write_df
import numpy as np
from matplotlib import pyplot as plt

def analysis():
    channels = read_channels('../output/final_data/P_channelData.json')
    channels_NP = read_channels('../output/final_data/NP_channelData.json')
    channels.update(channels_NP)
    subscriptions_with_zero = [len(channel['subscriptions']) for channel in channels.values()]
    subscriptions = [len(channel['subscriptions']) for channel in channels.values() if len(channel['subscriptions']) != 0]

    print(np.mean(subscriptions_with_zero))
    print(np.median(subscriptions_with_zero))
    print(max(subscriptions_with_zero))
    print(min(subscriptions_with_zero))

    print(np.mean(subscriptions))
    print(np.median(subscriptions))
    print(max(subscriptions))
    print(min(subscriptions))

    print(len([channel for channel in channels.values() if len(channel['cross'])]))


def plotSubscriptions(data, NP_data):
    subs = [int(channel['Subs']) for channel in data.values()]
    mean_subs = np.mean(subs)
    median = np.median(subs)
    std = np.std(subs)
    print(mean_subs)
    print(median)
    print(std)
    print(np.max(subs))
    print(np.min(subs))
    # x = 10 ** np.random.uniform(size=1000)



    plt.hist(subs, 750)
    # plt.hist(x, bins=10 ** np.linspace(0, 1, 10))
    min_ylim, max_ylim = plt.ylim()
    plt.axvline(mean_subs, color='k', linestyle='dashed', linewidth=1)
    plt.text(mean_subs * 1.1, max_ylim * 0.8, 'Mean: {:.2f}'.format(mean_subs))
    plt.axvline(median, color='g', linestyle='dashed', linewidth=1)
    plt.text(median * 1.1, max_ylim * 0.9, 'Median: {:}'.format(int(median)))
    plt.xscale('log')
    plt.xlabel('Number of Subscribers', size=13)
    plt.ylabel('Number of Channels', size=13)
    plt.show()
    left, right = plt.xlim()

    # NP data
    subs = [int(channel['Subs']) for channel in NP_data.values()]
    subs.append(1)
    mean_subs = np.mean(subs)
    median = np.median(subs)
    std = np.std(subs)
    print(mean_subs)
    print(median)
    print(std)
    print(np.max(subs))
    print(np.min(subs))
    # x = 10 ** np.random.uniform(size=1000)
    plt.hist(subs, 50)
    # plt.hist(x, bins=10 ** np.linspace(0, 1, 10))
    min_ylim, max_ylim = plt.ylim()
    plt.axvline(mean_subs, color='k', linestyle='dashed', linewidth=1)
    plt.text(mean_subs * 1.1, max_ylim * 0.8, 'Mean: {:.2f}'.format(mean_subs))
    plt.axvline(median, color='g', linestyle='dashed', linewidth=1)
    plt.text(median * 0.58, max_ylim * 0.9, 'Median: {:}'.format(int(median)))
    plt.xscale('log')
    # plt.xlim(left, right)
    plt.xlabel('Number of Subscribers', size=13)
    plt.ylabel('Number of Channels', size=13)
    plt.show()

if __name__ == "__main__":
    P_data = read_channels('../output/final_data/P_channelData.json')
    NP_data = read_channels('../output/final_data/NP_channelData.json')

    plotSubscriptions(P_data, NP_data)
