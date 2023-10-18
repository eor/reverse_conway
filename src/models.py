import torch
from torch import nn

from settings import SETTING_GAME_WIDTH


class MLP1(nn.Module):
    def __init__(self, conf):
        super().__init__()

        def block(features_in: int,
                  features_out: int,
                  normalise: bool = conf.batch_norm,
                  dropout: bool = conf.dropout):

            layers = [nn.Linear(features_in, features_out)]

            if normalise:
                layers.append(nn.BatchNorm1d(features_out))

            if dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        game_squares = SETTING_GAME_WIDTH**2
        self.model = nn.Sequential(
            *block(game_squares + 1, 1024, normalise=False, dropout=False),
            *block(1024, 1024),
            *block(1024, 1024),
            nn.Linear(1024, game_squares)
        )

    def forward(self, stop_grid, delta_t):

        network_input = torch.cat(tensors=(stop_grid, delta_t.view(1)), dim=0)
        return self.model(network_input)
