import os

import numpy as np
import matplotlib.pyplot as plt

import data


root_path = os.path.abspath(__file__ + "/../..")
output_dir = F"{root_path}/data/output/"


def plot_single_game(game: data.Game, target_dir=output_dir, file_type='png') -> str:

    start_field_1d = np.array([1 if x else 0 for x in game.start_grid.cells])
    stop_field_1d = np.array([1 if x else 0 for x in game.stop_grid.cells])

    grid_width = game.start_grid.width

    start_field_2d = np.reshape(start_field_1d, (grid_width, grid_width))
    stop_field_2d = np.reshape(stop_field_1d, (grid_width, grid_width))

    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    fig.suptitle(F"Game id = {game.id}      Time step = {game.delta}", size=24)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_title("Start state", size=20, pad=15)
    ax2.set_title("Stop state", size=20, pad=15)

    ax1.tick_params(axis='both', right=True, top=True, labelsize=15)
    ax2.tick_params(axis='both', right=True, top=True, labelsize=15)

    for i in range(0, grid_width):
        ax1.axvline(i + 0.5, color='silver', lw=1, alpha=0.95)
        ax1.axhline(i + 0.5, color='silver', lw=1, alpha=0.95)
        ax2.axvline(i + 0.5, color='silver', lw=1, alpha=0.95)
        ax2.axhline(i + 0.5, color='silver', lw=1, alpha=0.95)

    ax1.imshow(start_field_2d, cmap="binary", interpolation='antialiased')
    ax2.imshow(stop_field_2d, cmap="binary", interpolation='antialiased')

    figure_path = F"{target_dir}/game_id{game.id}_dt{game.delta}.{file_type}"
    fig.savefig(figure_path)

    # should we generate a lot of figures, the following is needed
    plt.close('all')

    return figure_path


if __name__ == "__main__":

    games = data.train_data()
    test_game2 = games.all[42]

    _ = plot_single_game(game=test_game2, file_type='png')
