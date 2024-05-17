import os

pre_frontal = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4']
frontal = ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8']
central = ['CZ', 'FCZ', 'C1', 'C2', 'C3', 'C4', 'FC1', 'FC2', 'FC3', 'FC4']
l_temporal = ['FT7', 'FC5', 'T7', 'C5', 'TP7', 'CP5', 'P7', 'P5']
r_temporal = ['FT8', 'FC6', 'T8', 'C6', 'TP8', 'CP6', 'P8', 'P6']
parietal = ['CPZ', 'CP1', 'CP3', 'CP2', 'CP4', 'PZ', 'P1', 'P3', 'P2', 'P4']
occipital = ['POZ', 'PO3', 'PO5', 'PO7', 'PO4', 'PO6', 'PO8', 'O1', 'O2', 'OZ', 'CB1', 'CB2']

FREQ_BANDS = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "alpha": [8, 13],
    "beta": [13, 30],
    "gamma": [30, 80]
}

IMG_DIR = '../data/imageNet_images/'


def wnid2category(wnid, language):
    valid_params = ['ch', 'en']
    if language not in valid_params:
        raise ValueError(f"Invalid parameter. Expected one of: {valid_params}")
    with open(os.path.join(IMG_DIR, 'synset_map_' + language + '.txt')) as f:
        lines = f.readlines()
        for line in lines:
            if wnid in line:
                return line.split()[1]
    raise ValueError(f"Could not find wnid: {wnid}")


def category2wnid(category, language):
    valid_params = ['ch', 'en']
    if language not in valid_params:
        raise ValueError(f"Invalid parameter. Expected one of: {valid_params}")
    with open(os.path.join(IMG_DIR, 'synset_map_' + language + '.txt')) as f:
        lines = f.readlines()
        for line in lines:
            if category in line:
                return line.split()[0]
    raise ValueError(f"Could not find category: {category}")
