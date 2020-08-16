import csv
import json

from utils.data_IO import read_channels

if __name__ == '__main__':
    p_data = read_channels("../output/P_channelData.json")
    # np_data = read_channels("../output/NP_channelData.json")
    u_data = read_channels("../output/final_data/U_channelData.json")
    duplicate_channels = []
    for key, value in u_data.items():
        if key in p_data.keys():
            print(value['ChannelTitle'])
            duplicate_channels.append(key)
    with open('duplicate_data.csv', 'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(duplicate_channels)
    for key in duplicate_channels:
        u_data.pop(key, None)
    with open("../output/final_data/U_channelData.json", "w") as f:
        json.dump(u_data, f, indent=4)

