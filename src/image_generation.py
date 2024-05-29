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


def model_init(args, device):
    if args.model.lower() == 'mlp_sd':
        _model = MLPMapper()
    else:
        raise ValueError(f"Couldn't find the model {args.model}")
    return _model


def model_main(args, model, train_loader, test_loader, criterion, optimizer, num_epochs, device, clip_embeddings):
    model = model.to(device)
    running_loss = 0.0
    report_batch = len(train_loader) / 2
    min_loss = 1e10
    min_loss_epoch = -1
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            labels = torch.stack([clip_embeddings[image_name] for image_name in labels]).squeeze()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(inputs)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % report_batch == report_batch - 1:
                print(f"[epoch {epoch}, batch {batch_idx}] loss: {running_loss / report_batch}")
                running_loss = 0.0
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for (inputs, labels) in test_loader:
                labels = torch.stack([clip_embeddings[image_name] for image_name in labels]).squeeze()
                inputs, labels = inputs.to(device), labels.to(device)
                embeddings = model(inputs)
                test_loss += criterion(embeddings, labels)
            print(f"Loss on test set: {test_loss / len(test_loader)}")
            if test_loss < min_loss:
                min_loss = test_loss
                min_loss_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'mlpsd_s{args.subject}_0.pth'))
    return min_loss_epoch, min_loss


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
    train_index = np.array([i for i in range(len(dataset)) if i % 50 < 30])
    test_index = np.array([i for i in range(len(dataset)) if i % 50 > 29])
    train_subset = Subset(dataset, train_index)
    test_subset = Subset(dataset, test_index)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = model_init(args, device)
    clip_embeddings = torch.load(os.path.join(args.output_dir, "clip_embeddings.pth"))
    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, str(args.pretrained_model))))
    if args.model.lower() == 'mlp_sd':
        dataset.use_frequency_feat = True
        dataset.use_image_label = True
        train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        epoch, loss = model_main(args, model, train_dataloader, test_dataloader, criterion, optimizer, 1000, device,
                                 clip_embeddings)
    with open(os.path.join(args.output_dir, f"mlpsd.txt"), "a") as f:
        f.write(f"{epoch}: {loss}")
        f.write("\n")
