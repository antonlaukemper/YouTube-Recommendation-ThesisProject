import csv
import json

from tqdm import tqdm

from utils.data_IO import read_channels


def remove_NP_data(data):
    with open("unlabeled_data_predictions.csv", 'r') as f:  # opening file in binary(rb) mode
        reader = csv.reader(f, delimiter=';', quotechar='|')
        next(reader, None)
        for row in tqdm(reader):
            if row[9]=='__label__non-political':
                data.pop(row[0], None)

    return data

if __name__ == '__main__':
    unlabeled_channels = read_channels("../output/final_data/U_channelData.json")
    remove_NP_data(unlabeled_channels)
    with open("../output/final_data/U_channelData_multilabel.json", "w") as f:
        json.dump(unlabeled_channels, f, indent=4)