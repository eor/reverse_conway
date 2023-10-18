import dataclasses
import os
import pickle
import typing as t

import pandas as pd
import numpy as np
from settings import SETTING_GAME_WIDTH, SETTING_DATA_INPUT_DIR

root_path = os.path.abspath(__file__ + "/../..")

Cell = bool


@dataclasses.dataclass
class Grid:
    cells: t.List[Cell]
    width: int

    def as_numpy_array(self) -> np.array:
        return np.array([float(x) for x in self.cells])


@dataclasses.dataclass
class Game:
    id: int
    delta: int
    start_grid: t.Optional[Grid]
    stop_grid: t.Optional[Grid]


@dataclasses.dataclass
class GameSet:
    all: t.List[Game]

    def starts_as_numpy_array(self) -> np.array:
        return np.array([game.start_grid.as_numpy_array() for game in self.all])

    def stops_as_numpy_array(self) -> np.array:
        return np.array([game.stop_grid.as_numpy_array() for game in self.all])

    def deltas_as_numpy_array(self) -> np.array:
        return np.array([game.delta for game in self.all])


def _read_csv(file_name: str, *, grid_width: int) -> GameSet:
    games = []

    cache_file_name = f"{file_name}.cache"
    if os.path.exists(cache_file_name):
        with open(cache_file_name, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(file_name)

    def read_field(ddf_row, field_name):
        cells = []
        i = 0
        while True:
            try:
                cells.append(bool(int(ddf_row[f"{field_name}_{i}"])))
            except KeyError:
                break
            i += 1
        return Grid(cells=cells, width=grid_width) if cells else None

    for i_df_row, df_row in df.iterrows():
        games.append(Game(id=df_row["id"],
                          delta=int(df_row["delta"]),
                          start_grid=read_field(df_row, "start"),
                          stop_grid=read_field(df_row, "stop")))

    game_set = GameSet(games)
    with open(cache_file_name, "wb") as f:
        pickle.dump(game_set, f)

    return game_set


def train_data() -> GameSet:
    return _read_csv(file_name=f"{root_path}/data/raw/train.csv",
                     grid_width=SETTING_GAME_WIDTH)


def test_data() -> GameSet:
    return _read_csv(file_name=f"{root_path}/data/raw/test.csv",
                     grid_width=SETTING_GAME_WIDTH)
