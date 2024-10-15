import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from de_feat_cal import de_feat_cal
from dataset import EEGImageNetDataset
from model.mlp_sd import MLPMapper
from utilities import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", required=True, help="choose from coarse, fine0-fine4 and all")
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="pretrained model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset(args)
    with open(os.path.join(args.output_dir, f"s{args.subject}.txt"), "w") as f:
        dataset.use_image_label = True
        for data in dataset:
            f.write(f"{data[1]}\n")
    with open(os.path.join(args.output_dir, f"s{args.subject}_label.txt"), "w") as f:
        dataset.use_image_label = False
        for idx, data in enumerate(dataset):
            if idx % 50 == 0:
                label_wnid = dataset.labels[data[1]]
                f.write(f"{idx + 1}-{idx + 50}: {wnid2category(label_wnid, 'ch')}\n")
