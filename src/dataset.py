import torch
import os
from torch.utils.data import Dataset


class EEGImageNetDataset(Dataset):
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
        self.use_frequency_feat = False
        self.frequency_feat = None

    def __getitem__(self, index):
        label = self.data[index]["label"]
        if self.use_frequency_feat:
            return self.frequency_feat[index], self.labels.index(label)
        else:
            eeg_data = self.data[index]["eeg_data"].float()
            eeg_data = eeg_data[:, 40:440]
            return eeg_data, self.labels.index(label)

    def __len__(self):
        return len(self.data)

    def add_frequency_feat(self, feat):
        if len(feat) == len(self.data):
            self.frequency_feat = torch.from_numpy(feat).float()
        else:
            raise ValueError("Frequency features must have same length")
