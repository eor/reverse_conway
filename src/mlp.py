import os
import argparse
import copy

from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import numpy as np

import data
import dataloader
import models
from settings import \
    SETTING_DATA_INPUT_DIR, \
    SETTING_MAIN_OUTPUT_DIR, \
    SETTING_NUM_EPOCHS, \
    SETTING_BATCH_SIZE, \
    SETTING_LEARNING_RATE, \
    SETTING_P_DROPOUT, \
    SETTING_DATA_PRODUCTS_SUBDIR, \
    SETTING_PLOTS_SUBDIR, \
    SETTING_TEST_FREQ, \
    SETTING_GAME_WIDTH

# -----------------------------------------------------------------
#  CUDA?
# -----------------------------------------------------------------
if torch.cuda.is_available():
    cuda = True
    device = torch.device("cuda")
    FloatTensor = torch.cuda.FloatTensor
else:
    cuda = False
    device = torch.device("cpu")
    FloatTensor = torch.FloatTensor


# -----------------------------------------------------------------
#  loss function(s)
# -----------------------------------------------------------------
def loss_function(gen_x, real_x):
    """
     Computes the MLP loss function

    Args:
        gen_x: inferred profile
        real_x: simulated profile

    Returns:
        loss
    """
    # TODO: try cross entropy or hinge loss?
    # F.binary_cross_entropy()
    # F.hinge_embedding_loss()

    return F.mse_loss(input=gen_x,
                      target=real_x.view(-1, SETTING_GAME_WIDTH**2),
                      reduction='mean')


# -----------------------------------------------------------------
#   use MLP with test or validation set
# -----------------------------------------------------------------
def run_evaluation(current_epoch, data_loader, model, path, config,
                       print_results=False, save_results=False, best_model=False):
    """
    function runs the given dataset through the mlp model, returns mse_loss and dtw_loss,
    and saves the results as well as ground truth to file, if save_results is True.

    Args:
        current_epoch: current epoch
        data_loader: data loader used for the inference, most likely the test set
        path: path to output directory
        model: current model state
        config: config object with user supplied parameters
        print_results: print average loss to screen?
        save_results: flag to save generated profiles locally (default: False)
        best_model: flag for testing on best model
    """

    if save_results:
        print("\033[94m\033[1mTesting the MLP now at epoch %d \033[0m" % current_epoch)

    if cuda:
        model.cuda()

    if save_results:
        profiles_gen_all = torch.tensor([], device=device)
        profiles_true_all = torch.tensor([], device=device)
        parameters_true_all = torch.tensor([], device=device)

    # Note: ground truth data could be obtained elsewhere but by getting it from the data loader here
    # we don't have to worry about randomisation of the samples.

    model.eval()

    loss_dtw = 0.0
    loss_mse = 0.0

    with torch.no_grad():
        for i, (profiles, parameters) in enumerate(data_loader):

            # configure input
            profiles_true = Variable(profiles.type(FloatTensor))
            parameters = Variable(parameters.type(FloatTensor))

            # inference
            profiles_gen = model(parameters)


            # compute loss via MSE:
            mse = loss_function('MSE', profiles_true, profiles_gen, config)
            loss_mse += mse

            if save_results:
                # collate data
                profiles_gen_all = torch.cat((profiles_gen_all, profiles_gen), 0)
                profiles_true_all = torch.cat((profiles_true_all, profiles_true), 0)
                parameters_true_all = torch.cat((parameters_true_all, parameters), 0)

    # mean of computed losses
    loss_mse = loss_mse / len(data_loader)
    loss_dtw = loss_dtw / len(data_loader)

    if print_results:
        print("results: MSE: %e DTW %e" % (loss_mse, loss_dtw))

    if save_results:
        # move data to CPU, re-scale parameters, and write everything to file
        profiles_gen_all = profiles_gen_all.cpu().numpy()
        profiles_true_all = profiles_true_all.cpu().numpy()
        parameters_true_all = parameters_true_all.cpu().numpy()

        parameters_true_all = utils_rescale_parameters(limits=config.parameter_limits, parameters=parameters_true_all)

        if best_model:
            prefix = 'best'
        else:
            prefix = 'test'

        utils_save_test_data(
            parameters=parameters_true_all,
            profiles_true=profiles_true_all,
            profiles_gen=profiles_gen_all,
            path=path,
            profile_choice=config.profile_type,
            epoch=current_epoch,
            prefix=prefix
        )

    return loss_mse.item(), loss_dtw.item()


# -----------------------------------------------------------------
#  Training
# -----------------------------------------------------------------
def mlp_train(model, optimizer, train_loader):
    """
    This function trains the network for one epoch.
    Returns: averaged training loss. No need to return the model as the optimizer modifies it inplace.

    Args:
        model: current model state
        optimizer: optimizer object to perform the back-propagation
        train_loader: data loader used for the inference, most likely the test set

    Returns:
          The average loss
    """

    if cuda:
        model.cuda()

    model.train()
    train_loss = 0

    for batch_idx, (start_grids, stop_grids, delta_ts) in enumerate(train_loader):

        # configure input
        start_grids = Variable(start_grids.type(FloatTensor))
        stop_grids = Variable(stop_grids.type(FloatTensor))
        delta_ts = Variable(delta_ts.type(FloatTensor))

        # zero the gradients on each iteration
        optimizer.zero_grad()

        # generate a batch of profiles
        gen_start_grids = model(stop_grids, delta_ts)

        # estimate loss
        loss = loss_function(gen_start_grids, start_grids)

        train_loss += loss.item()    # average loss per batch

        # back propagation
        loss.backward()
        optimizer.step()

    average_loss = train_loss / len(train_loader)   # divide by number of batches (!= batch size)

    return average_loss  # float


# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main(config):


    train_loader = dataloader.LifeData(data.train_data(), split="train")
    exit(1)

    #test_loader = dataloader.LifeData(data.test_data(), split="test")
    #validation_loader = dataloader.LifeData(data.train_data(), split="val")

    model = models.MLP1(conf=config)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    train_loss_array = val_loss_mse_array = np.empty(0)

    # -----------------------------------------------------------------
    # keep the model with min validation loss
    # -----------------------------------------------------------------
    best_model = copy.deepcopy(model)
    best_loss_mse = best_loss_dtw = np.inf
    best_epoch_mse = best_epoch_dtw = 0

    # -----------------------------------------------------------------
    # Early Stopping Criteria
    # -----------------------------------------------------------------
    n_epoch_without_improvement = 0
    stopped_early = False
    epochs_trained = -1

    # -----------------------------------------------------------------
    #  Main training loop
    # -----------------------------------------------------------------
    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, n_epochs + 1):

        train_loss = mlp_train(model, optimizer, train_loader)
        print(f"epoch #{n_epochs} average loss: {train_loss}")
        continue

        train_loss_array = np.append(train_loss_array, train_loss)

        val_loss_mse, val_loss_dtw = mlp_run_evaluation(current_epoch=epoch,
                                                        data_loader=validation_loader,
                                                        model=model,
                                                        path=data_products_path,
                                                        config=config,
                                                        print_results=False,
                                                        save_results=False,
                                                        best_model=False
                                                        )

        val_loss_mse_array = np.append(val_loss_mse_array, val_loss_mse)
        val_loss_dtw_array = np.append(val_loss_dtw_array, val_loss_dtw)

        if val_loss_mse < best_loss_mse:
            best_loss_mse = val_loss_mse
            best_model = copy.deepcopy(model)
            best_epoch_mse = epoch
            n_epoch_without_improvement = 0
        else:
            n_epoch_without_improvement += 1

        if val_loss_dtw < best_loss_dtw:
            best_loss_dtw = val_loss_dtw
            best_epoch_dtw = epoch

        print("[Epoch {}/{}] [Train loss {}: {}] [Validation loss MSE: {}] [Validation loss DTW: {}] "
              "[Best_epoch (mse): {}] [Best_epoch (dtw): {}]"
              .format(epoch, config.n_epochs, config.loss_type, train_loss, val_loss_mse, val_loss_dtw,
                      best_epoch_mse, best_epoch_dtw)
        )

        # check for testing criterion
        if epoch % config.testing_interval == 0 or epoch == config.n_epochs:

            best_test_mse, best_test_dtw = mlp_run_evaluation(current_epoch=best_epoch_mse,
                                                              data_loader=test_loader,
                                                              model=best_model,
                                                              path=data_products_path,
                                                              config=config,
                                                              print_results=True,
                                                              save_results=True,
                                                              best_model=False
                                                              )

        # early stopping check
        if FORCE_STOP or (EARLY_STOPPING and n_epoch_without_improvement >= EARLY_STOPPING_THRESHOLD_MLP):
            print("\033[96m\033[1m\nStopping Early\033[0m\n")
            stopped_early = True
            epochs_trained = epoch
            break

    print("\033[96m\033[1m\nTraining complete\033[0m\n")

    # -----------------------------------------------------------------
    # Save the best model and the final model
    # -----------------------------------------------------------------
    utils_save_model(best_model.state_dict(), data_products_path, config.profile_type, best_epoch_mse, best_model=True)
    utils_save_model(model.state_dict(), data_products_path, config.profile_type, config.n_epochs, best_model=False)

    utils_save_loss(train_loss_array, data_products_path, config.profile_type, config.n_epochs, prefix='train')

    if config.loss_type == 'MSE':
        utils_save_loss(val_loss_mse_array, data_products_path,
                        config.profile_type, config.n_epochs, prefix='val')
    else:
        utils_save_loss(val_loss_dtw_array, data_products_path,
                        config.profile_type, config.n_epochs, prefix='val')

    # -----------------------------------------------------------------
    # Evaluate the best model by using the test set
    # -----------------------------------------------------------------
    best_test_mse, best_test_dtw = mlp_run_evaluation(best_epoch_mse, test_loader, best_model, data_products_path,
                                                      config, print_results=True, save_results=True, best_model=True)

    # -----------------------------------------------------------------
    # Save some results to config object for later use
    # -----------------------------------------------------------------
    config.best_epoch = best_epoch_mse
    config.best_epoch_mse = best_epoch_mse
    config.best_epoch_dtw = best_epoch_dtw

    config.best_val_mse = best_loss_mse
    config.best_val_dtw = best_loss_dtw

    config.best_test_mse = best_test_mse
    config.best_test_dtw = best_test_dtw

    config.stopped_early = stopped_early
    config.epochs_trained = epochs_trained
    config.early_stopping_threshold = EARLY_STOPPING_THRESHOLD_MLP

    # -----------------------------------------------------------------
    # Overwrite config object
    # -----------------------------------------------------------------
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    # finished
    print('\nAll done!')

    # -----------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------
    if config.analysis:
        print("\n\033[96m\033[1m\nRunning analysis\033[0m\n")

        analysis_loss_plot(config)
        analysis_auto_plot_profiles(config, k=30, prefix='best')
        analysis_parameter_space_plot(config, prefix='best')
        analysis_error_density_plot(config, prefix='best')


# -----------------------------------------------------------------
#  The following is executed when the script is executed
# -----------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reverse Conway experiment"
    )

    # arguments for data handling
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to (CSV) data directory",
        default=SETTING_DATA_INPUT_DIR,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=SETTING_MAIN_OUTPUT_DIR,
        help=f"Path to output directory, used for all plots and data products, default: {SETTING_MAIN_OUTPUT_DIR}"
    )

    # network optimisation
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=SETTING_NUM_EPOCHS,
        help="number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=SETTING_BATCH_SIZE,
        help=f"size of the batches, default={SETTING_BATCH_SIZE}"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=SETTING_LEARNING_RATE,
        help=f"adam: learning rate, default={SETTING_LEARNING_RATE} "
    )

    parser.add_argument(
        "--dropout_value",
        type=float,
        default=SETTING_P_DROPOUT,
        help=f"dropout probability, default={SETTING_P_DROPOUT} "
    )

    # BN on / off
    parser.add_argument(
        "--batch_norm",
        dest="batch_norm",
        action="store_true",
        help="use batch normalisation in network"
    )
    parser.add_argument(
        "--no-batch_norm",
        dest="batch_norm",
        action="store_false",
        help="do not use batch normalisation in network (default)"
    )
    parser.set_defaults(batch_norm=False)

    # dropout on / off
    parser.add_argument(
        "--dropout",
        dest="dropout",
        action="store_true",
        help="use dropout regularisation in network"
    )
    parser.add_argument(
        "--no-dropout",
        dest="dropout",
        action="store_false",
        help="do not use dropout regularisation in network (default)"
    )
    parser.set_defaults(dropout=False)

    # run analysis / plotting  routines after training?
    parser.add_argument(
        "--analysis",
        dest="analysis",
        action="store_true",
        help="automatically generate some plots (default)"
    )
    parser.add_argument(
        "--no-analysis",
        dest="analysis",
        action="store_false",
        help="do not run analysis"
    )
    parser.set_defaults(analysis=True)

    my_config = parser.parse_args()

    my_config.out_dir = os.path.abspath(my_config.out_dir)
    my_config.data_dir = os.path.abspath(my_config.data_dir)

    main(my_config)
