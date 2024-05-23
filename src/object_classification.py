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
from model.rgnn import RGNN, get_edge_weight
from utilities import *


def model_init(args, if_simple, num_classes, device):
    if if_simple:
        _model = SimpleModel(args)
    elif args.model.lower() == 'eegnet':
        _model = EEGNet(args, num_classes)
    elif args.model.lower() == 'mlp':
        _model = MLP(args, num_classes)
    elif args.model.lower() == 'rgnn':
        edge_index, edge_weight = get_edge_weight()
        _model = RGNN(device, 62, edge_weight, edge_index, 5, 200, num_classes, 2)
    else:
        raise ValueError(f"Couldn't find the model {args.model}")
    return _model


def model_main(args, model, train_loader, test_loader, criterion, optimizer, num_epochs, device, labels):
    model = model.to(device)
    unique_labels = torch.from_numpy(labels).unique()
    label_mapping = {original_label.item(): new_label for new_label, original_label in enumerate(unique_labels)}
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    running_loss = 0.0
    max_acc = 0.0
    max_acc_epoch = -1
    report_batch = len(train_loader) / 2
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            labels = torch.tensor([label_mapping[label.item()] for label in labels])
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % report_batch == report_batch - 1:
                print(f"[epoch {epoch}, batch {batch_idx}] loss: {running_loss / report_batch}")
                running_loss = 0.0
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0
            for (inputs, labels) in test_loader:
                labels = torch.tensor([label_mapping[label.item()] for label in labels])
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, dim=1)
                total += len(labels)
                correct += accuracy_score(labels.cpu(), predicted.cpu(), normalize=False)
        acc = correct / total
        print(f"Accuracy on test set: {acc}; Loss on test set: {test_loss / len(test_loader)}")
        if acc > max_acc:
            max_acc = acc
            max_acc_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'eegnet_s0_2x_0.pth'))
    return max_acc, max_acc_epoch


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
    eeg_data = np.stack([i[0].numpy() for i in dataset], axis=0)
    # extract frequency domain features
    de_feat = de_feat_cal(eeg_data, args)
    dataset.add_frequency_feat(de_feat)
    labels = np.array([i[1] for i in dataset])
    train_index = np.array([i for i in range(len(dataset)) if i % 50 < 30])
    test_index = np.array([i for i in range(len(dataset)) if i % 50 > 29])
    train_subset = Subset(dataset, train_index)
    test_subset = Subset(dataset, test_index)

    simple_model_list = ['svm', 'rf', 'knn', 'dt', 'ridge']
    if_simple = args.model.lower() in simple_model_list
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model_init(args, if_simple, len(dataset) // 50, device)
    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, str(args.pretrained_model))))
    if if_simple:
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        train_feat = de_feat[train_index]
        test_feat = de_feat[test_index]
        model.fit(train_feat, train_labels)
        y_pred = model.predict(test_feat)
        acc = accuracy_score(test_labels, y_pred)
        with open(os.path.join(args.output_dir, "tmp.txt"), "a") as f:
            f.write(f"{acc}")
            f.write("\n")
    else:
        if args.model.lower() == 'eegnet':
            train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-2, momentum=0.9)
            acc, epoch = model_main(args, model, train_dataloader, test_dataloader, criterion, optimizer, 1000, device,
                                    labels)
        elif args.model.lower() == 'mlp':
            dataset.use_frequency_feat = True
            train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9)
            acc, epoch = model_main(args, model, train_dataloader, test_dataloader, criterion, optimizer, 1000, device,
                                    labels)
        elif args.model.lower() == 'rgnn':
            dataset.use_frequency_feat = True
            train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            acc, epoch = model_main(args, model, train_dataloader, test_dataloader, criterion, optimizer, 1000, device,
                                    labels)
        with open(os.path.join(args.output_dir, "tmp.txt"), "a") as f:
            f.write(f"{epoch}: {acc}")
            f.write("\n")
