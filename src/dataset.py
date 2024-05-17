import torch
import os
from torch.utils.data import Dataset


class EEG_ImageNet_Dataset(Dataset):
    def __init__(self, args):
        loaded = torch.load(os.path.join(args.dataset_dir, "EEG-ImageNet.pth"))
        if args.subject != -1:
            chosen_data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           loaded['dataset'][i]['subject'] == args.subject]
        else:
            chosen_data = loaded['dataset']
        if args.granularity != 'all':
            self.data = [i for i in chosen_data if i['granularity'] == args.granularity]
        else:
            self.data = chosen_data
        self.labels = loaded["labels"]
        self.images = loaded["images"]

    def __getitem__(self, index):
        eeg_data = self.data[index]["eeg_data"].t()
        eeg_data = eeg_data[40:440, :]
        label = self.data[index]["label"]
        return eeg_data, label

    def __len__(self):
        return len(self.data)
