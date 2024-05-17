import argparse
import os.path

import numpy as np
from dataset import EEG_ImageNet_Dataset
from de_feat_cal import de_feat_cal
from model.simple_model import SVM_classifier, RF_classifier, KNN_classifier

if __name__ == '__main__':
    granularity_choice = ["coarse", "fine", "all"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", required=True, choices=granularity_choice,
                        help="choose from coarse, fine and all")
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_path", required=True, help="path of output file")
    args = parser.parse_args()
    print(args)

    dataset = EEG_ImageNet_Dataset(args)
    train_index = np.array([i for i in range(len(dataset)) if i % 50 < 30])
    test_index = np.array([i for i in range(len(dataset)) if i % 50 > 29])
    eeg_data = np.stack([i[0].T.numpy() for i in dataset], axis=0)
    de_feat = de_feat_cal(eeg_data, args)
    train_feat = de_feat[train_index]
    test_feat = de_feat[test_index]
    labels = np.array([dataset.labels.index(i[1]) for i in dataset])
    train_labels = labels[train_index]
    test_labels = labels[test_index]

    if args.model.lower() == 'svm':
        model = SVM_classifier
    elif args.model.lower() == 'rf':
        model = RF_classifier
    elif args.model.lower() == 'knn':
        model = KNN_classifier
    else:
        raise ValueError(f"Couldn't find the model {args.model}")

    acc = model(train_feat, train_labels, test_feat, test_labels)
    with open(args.output_path, "a") as f:
        f.write(str(acc))
        f.write("\n")
