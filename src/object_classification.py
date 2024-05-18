import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset import EEGImageNetDataset
from de_feat_cal import de_feat_cal
from model.simple_model import SimpleModel
from model.eegnet import EEGNet
from model.mlp import MLP
from utilities import *


def model_init(args, if_simple, dataset_size):
    if if_simple:
        _model = SimpleModel(args)
    elif args.model.lower() == 'eegnet':
        _model = EEGNet(args, dataset_size // 50)
    elif args.model.lower() == 'mlp':
        _model = MLP(args, dataset_size // 50)
    else:
        raise ValueError(f"Couldn't find the model {args.model}")
    return _model


def model_main(args, model, train_loader, test_loader, criterion, optimizer, num_epochs):
    running_loss = 0.0
    max_acc = 0.0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 30 == 29:
                print(f"[epoch {epoch}, batch {batch_idx}] loss: {running_loss / 30}")
                running_loss = 0.0
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for (inputs, labels) in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                total += len(labels)
                correct += accuracy_score(labels, predicted, normalize=False)
        acc = correct / total
        print(f"Accuracy on test set: {acc}")
        if acc > max_acc:
            max_acc = acc
    return max_acc


if __name__ == '__main__':
    granularity_choice = ["coarse", "fine", "all"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", required=True, choices=granularity_choice,
                        help="choose from coarse, fine and all")
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="pretrained model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset(args)
    train_index = np.array([i for i in range(len(dataset)) if i % 50 < 30])
    test_index = np.array([i for i in range(len(dataset)) if i % 50 > 29])
    train_subset = Subset(dataset, train_index)
    test_subset = Subset(dataset, test_index)

    simple_model_list = ['svm', 'rf', 'knn', 'dt', 'ridge']
    if_simple = args.model.lower() in simple_model_list
    model = model_init(args, if_simple, len(dataset))
    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_path, str(args.pretrained_model))))
    if if_simple:
        eeg_data = np.stack([i[0].numpy() for i in dataset], axis=0)
        labels = np.array([i[1] for i in dataset])
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        # extract frequency domain features
        de_feat = de_feat_cal(eeg_data, args)
        train_feat = de_feat[train_index]
        test_feat = de_feat[test_index]

        model.fit(train_feat, train_labels)
        y_pred = model.predict(test_feat)
        acc = accuracy_score(test_labels, y_pred)
    elif args.model.lower() == 'eegnet':
        train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
        acc = model_main(args, model, train_dataloader, test_dataloader, criterion, optimizer, 200)
        torch.save(model.state_dict(), os.path.join(args.output_path, 'eegnet.pth'))
    elif args.model.lower() == 'mlp':
        train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
        acc = model_main(args, model, train_dataloader, test_dataloader, criterion, optimizer, 100)
        torch.save(model.state_dict(), os.path.join(args.output_path, 'mlp.pth'))
    with open(os.path.join(args.output_path, "tmp.txt"), "a") as f:
        f.write(str(acc))
        f.write("\n")
