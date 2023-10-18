import typing as t
import torch
from torch.utils.data import Dataset

import data
from settings import \
    SETTING_TRAIN_FRACTION, \
    SETTING_VAL_FRACTION, \
    SETTING_TEST_FRACTION


class LifeData(Dataset):

    def __init__(self, game_set: data.GameSet, split: str = 'train',
                 split_frac: t.Tuple[float, float, float] = (SETTING_TRAIN_FRACTION,
                                                             SETTING_VAL_FRACTION,
                                                             SETTING_TEST_FRACTION)
                 ):

        if sum(split_frac) != 1:
            raise Exception('Error: Fractions of train | val | test should add up to 1.0')

        train_frac, val_frac, test_frac = split_frac

        n_samples = game_set.deltas_as_numpy_array().shape[0]

        self.start_grids = game_set.starts_as_numpy_array()
        self.stop_grids = game_set.stops_as_numpy_array()
        self.delta_ts = game_set.deltas_as_numpy_array()

        if split == 'train':
            begin = 0
            last = int(train_frac * n_samples)

        elif split == 'val':
            begin = int(train_frac * n_samples)
            last = int((train_frac + val_frac) * n_samples)

        elif split == 'test':
            begin = int((train_frac + val_frac) * n_samples)
            last = -1

        else:
            raise Exception("Unknown 'split' keyword.")

        self.start_grids = torch.from_numpy(self.start_grids[begin:last]).type(torch.FloatTensor)
        self.stop_grids = torch.from_numpy(self.stop_grids[begin:last]).type(torch.FloatTensor)
        self.delta_ts = torch.from_numpy(self.delta_ts[begin:last]).type(torch.FloatTensor)

    def __len__(self):

        return self.delta_ts.shape[0]

    def __getitem__(self, index):

        out = (self.start_grids[index], self.stop_grids[index], self.delta_ts[index])

        return out
