import numpy as np
import mne
from utilities import *


def de_feat_cal(eeg_data, args):
    if os.path.exists(os.path.join('../data/de_feat/', f"{args.subject}_{args.granularity}_de.npy")):
        return np.load(os.path.join('../data/de_feat/', f"{args.subject}_{args.granularity}_de.npy"))
    else:
        channel_names = [f'EEG{i}' for i in range(1, 63)]
        info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types='eeg')
        _epochs = mne.EpochsArray(data=eeg_data, info=info)

        de_feat_list = []
        for f_min, f_max in FREQ_BANDS.values():
            spectrum = _epochs.compute_psd(fmin=f_min, fmax=f_max)
            psd = spectrum.get_data() + 1e-10
            diff_entropy = np.sum(np.log(psd), axis=-1)
            de_feat_list.append(diff_entropy)
        _de_feat = np.concatenate(de_feat_list, axis=1)
        # print(_de_feat.shape)  # de_feat.shape = (4000, 310), normally
        np.save(os.path.join('../data/de_feat/', f"{args.subject}_{args.granularity}_de.npy"), _de_feat)
        return _de_feat
