import os
import argparse
import random
import csv
import json
import time as time_measure

from typing import Any, Dict, List, Tuple, Union

os.environ['DDE_BACKEND'] = 'tensorflow'
import deepxde as dde
from deepxde.backend import tf

import numpy as np

from data.data_generator import DataLoader
from data.signal import SignalData, SignalPairData
from data.data_processing import ASLFit
from model.domain import SystemDynamics, SystemPairDynamics
from model.model import PINN, PairPINN

import matplotlib.pyplot as plt



def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed command line arguments.
    """

    def parse_list(arg):
        if arg is None:
            return None
        return list(map(int, arg.split(',')))
    
    parser = argparse.ArgumentParser(description="ASL PINNs")
    parser.add_argument("--seed", default=0, type=int, help="Seed value")
    parser.add_argument("--use_synthetic", default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to use synthetic data")
    parser.add_argument("--noise", default=0.0, type=float, help="The Gaussian standard deviation when adding noise")
    parser.add_argument("--use_input_transform", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to use input transform")
    parser.add_argument("--use_output_transform", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to use output transform")
    parser.add_argument("--spatial_average", default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to spatially average")
    parser.add_argument("--spatial_weight", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to weigh each point based on uncertainty")
    parser.add_argument("--pair", default=2, type=int, help="The number of pairs to use")
    parser.add_argument("--train_cbf", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to train cerebral blood flow")
    parser.add_argument("--train_at", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to train arrival time")
    parser.add_argument("--train_t1b", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to train T1 blood")
    parser.add_argument("--cases", default=None, type=parse_list, help="List of cases as comma-separated integers or None")
    
    return parser.parse_args()


def get_output_folder(args) -> Tuple[str, List[str]]:
    """
    Get the output folder and columns based on the given arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    output_folder : str
        The output folder name.
    columns : List[str]
        The list of column names.

    """
    use_synthetic = args.use_synthetic
    noise = args.noise
    
    spatial_average = args.spatial_average
    spatial_weight = args.spatial_weight
    pair = args.pair
    
    train_cbf = args.train_cbf
    train_at = args.train_at
    train_t1b = args.train_t1b

    output_folder = '_outputs'
    columns = ['Epochs']
    if use_synthetic:
        output_folder += '_synth'
    if noise > 0.0:
        output_folder += '_noise_{:.2f}'.format(noise)
    if train_cbf:
        output_folder += '_cbf'
        columns.append('CBF')
        for i in range(pair):
            columns.append('CBF{}'.format(i+2))
    if train_at:
        output_folder += '_at'
        columns.append('AT')
        for i in range(pair):
            columns.append('AT{}'.format(i+2))
    if train_t1b:
        output_folder += '_t1b'
        columns.append('T1b')
    if pair > 0:
        output_folder += '_pair{}'.format(pair)
    if spatial_average:
        output_folder += '_sa'
    if spatial_weight:
        output_folder += '_sw'
        
    return output_folder, columns
    
    
def set_global_seed(seed: int) -> None:
    """
    Set the global seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set.

    Returns
    -------
    None

    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    
def get_callbacks(dynamics: SystemDynamics, train_cbf: bool, train_at: bool,
                  train_t1b: bool, pair: int , checkpoint_path: str,
                  case_path: str) -> Tuple[List[dde.callbacks.Callback],
                                           List[tf.Variable],
                                           List[tf.Variable]]:
    """
    Get the callbacks, callback variables, and trainable variables for the training process.

    Parameters
    ----------
    dynamics : SystemDynamics
        The system dynamics object.
    train_cbf : bool
        Whether to train the cerebral blood flow.
    train_at : bool
        Whether to train the arrival time.
    train_t1b : bool
        Whether to train the T1b_ms.
    pair : int
        The number of pairs.
    checkpoint_path : str
        The path to save the model checkpoints.
    case_path : str
        The path to save the callback variables.

    Returns
    -------
    callbacks : List[dde.callbacks.Callback]
        The list of callbacks for the training process.
    callback_variables : List[tf.Variable]
        The list of callback variables.
    trainable_variables : List[tf.Variable]
        The list of trainable variables.

    """
    trainable_variables = []
    callbacks = []
    callback_variables = []
    if train_cbf:
        trainable_variables.append(dynamics.cerebral_blood_flow)
        for i in range(pair):
            trainable_variables.append(dynamics.cerebral_blood_flow_pair[i])
    if train_at:
        trainable_variables.append(dynamics.arrival_time)
        for i in range(pair):
            trainable_variables.append(dynamics.arrival_time_pair[i])
    if train_t1b:
        trainable_variables.append(dynamics.T1b_ms)
        
    if train_cbf or train_at:
        period = 1
        callback_variables.append(dde.callbacks.VariableValue(trainable_variables,
                                                              period=period,
                                                              filename=os.path.join(case_path, 'variables.dat'),
                                                              precision=8))
    

    callbacks.append(dde.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model'),
                                                   verbose=1, save_better_only=True,
                                                   monitor='train loss'))
    callbacks.append(dde.callbacks.EarlyStopping(min_delta=1e-7, patience=5000, monitor='loss_train'))
    
    return callbacks, callback_variables, trainable_variables
    
    
def macro_optimization(model: dde.model.Model, checkpoint_path: str, learning_rate: float,
                       loss_weights: List[float], iterations: int,
                       callbacks: List[dde.callbacks.Callback]) -> Tuple[dde.model.LossHistory,
                                                                         dde.model.TrainState]:
    """
    Perform macro (1st stage) optimization for the given model.

    Parameters
    ----------
    model : dde.model.Model
        The model to be optimized.
    checkpoint_path : str
        The path to save the model checkpoints.
    learning_rate : float
        The learning rate for the optimization.
    loss_weights : List[float]
        The weights for different losses in the model.
    iterations : int
        The number of iterations for the optimization.
    callbacks : List[dde.callbacks.Callback]
        The list of callbacks to be used during training.

    Returns
    -------
    losshistory : dde.model.LossHistory
        The loss history of the optimization.
    train_state : dde.model.TrainState
        The final state of the training process.

    """
    model.save(os.path.join(checkpoint_path, 'model'))
    model.compile('adam', lr=learning_rate, metrics=['MSE'],
                  loss_weights=loss_weights,
                  decay=('inverse time', 500, 0.1))
    losshistory, train_state = model.train(iterations=iterations,
                                           callbacks=callbacks)
    
    return losshistory, train_state
    
  
def meso_optimization(model: dde.model.Model, checkpoint_path: str,
                      train_state: dde.model.TrainState, learning_rate: float,
                      loss_weights: List[float], trainable_variables: List[tf.Variable],
                      iterations: int, callbacks: dde.callbacks.Callback) -> Tuple[dde.model.LossHistory,
                                                                                   dde.model.TrainState]:
    """
    Perform meso (2nd stage) optimization for the given model.

    Parameters
    ----------
    model : dde.model.Model
        The model used for optimization.
    checkpoint_path : str
        The path to the directory containing the checkpoint files.
    train_state : dde.model.TrainState
        The training state of the model.
    learning_rate : float
        The learning rate for the optimizer.
    loss_weights : List[float]
        The weights for the different loss functions.
    trainable_variables : List[tf.Variable]
        The list of trainable variables.
    iterations : int
        The number of optimization iterations.
    callbacks : dde.callbacks.Callback
        The callbacks to be executed during training.

    Returns
    -------
    losshistory : dde.model.LossHistory
        The loss history of the optimization process.
    train_state : dde.model.TrainState
        The final training state of the model.

    """
    # Load previous best model for this optimization step
    model.restore(os.path.join(checkpoint_path, 'model-' + str(train_state.best_step) + '.ckpt'), verbose=1)
    model.compile('adam', lr=learning_rate, metrics=['MSE'],
                  loss_weights=loss_weights,
                  decay=('inverse time', 2000, 0.15),
                  external_trainable_variables=trainable_variables)
    losshistory, train_state = model.train(iterations=iterations, display_every=1000,
                                           callbacks=callbacks)
  
    return losshistory, train_state

    
def micro_optimization(model: dde.model.Model, checkpoint_path: str, train_state: dde.model.TrainState,
                       loss_weights: List[float], trainable_variables: List[tf.Variable],
                       callbacks: dde.callbacks.Callback) -> Tuple[dde.model.LossHistory,
                                                                  dde.model.TrainState]:
    """
    Perform micro (3rd stage) optimization for the given model.

    Parameters
    ----------
    model : dde.model.Model
        The deep learning model to be optimized.
    checkpoint_path : str
        The path to the directory where the model checkpoints will be saved.
    train_state : dde.model.TrainState
        The training state of the model.
    loss_weights : List[float]
        The weights for the different loss terms in the model.
    trainable_variables : List[tf.Variable]
        The list of trainable variables in the model.
    callbacks : dde.callbacks.Callback
        The callbacks to be used during training.

    Returns
    -------
    losshistory : dde.model.LossHistory
        The history of losses during training.
    train_state : dde.model.TrainState
        The updated training state of the model.

    """
    max_iter = 100
    dde.optimizers.config.set_LBFGS_options(
        maxcor=100,
        ftol=1e-8,
        gtol=1e-8,
        maxiter=max_iter,
        maxfun=None,
        maxls=100)
    
    model.compile('L-BFGS', metrics=['MSE'], loss_weights=loss_weights,
                  external_trainable_variables=trainable_variables)
    model.restore(os.path.join(checkpoint_path, 'model-' + str(train_state.best_step) + '.ckpt'), verbose=1)
    losshistory, train_state = model.train(callbacks=callbacks,
                                            display_every=1,
                                            model_save_path=os.path.join(checkpoint_path, 'model'))
    
    # A workaround to save the last step variables obtained from L-BFGS
    model.compile('adam', lr=1e-12, metrics=['MSE'], external_trainable_variables=trainable_variables)
    losshistory, train_state = model.train(iterations=1, callbacks=callbacks, display_every=1,
                                           model_save_path=os.path.join(checkpoint_path, 'model'))
    
    return losshistory, train_state


def multi_optimization(model: dde.model.Model, pair: int, callbacks: List[dde.callbacks.Callback],
                       trainable_variables: List[tf.Variable], callback_variables: List[tf.Variable],
                       checkpoint_path: str) -> Tuple[dde.model.LossHistory, dde.model.TrainState]:
    """
    Perform multi-stage optimization for a given model.

    Parameters
    ----------
    model : dde.model.Model
        The model to be optimized.
    pair : int
        The number of pairs of ODEs and boundary conditions.
    callbacks : List[dde.callbacks.Callback]
        List of callbacks to be used during optimization.
    trainable_variables : List[tf.Variable]
        List of trainable variables in the model.
    callback_variables : List[tf.Variable]
        List of callback variables to be added to the callbacks.
    checkpoint_path : str
        The path to save the checkpoints during optimization.

    Returns
    -------
    losshistory : dde.model.LossHistory
        The loss history during optimization.
    train_state : dde.model.TrainState
        The final training state after optimization.

    """
    learning_rate_1 = 0.001
    learning_rate_2 = 0.0001
    iterations_1 = 5000
    iterations_2 = 20000
    # Weight order: ODE, ODE, bcs, bcs
    ode_loss = [1.0] * (pair+1)
    bcs_loss = [0.005] * (pair+1)
    loss_weights = ode_loss + bcs_loss

    
    losshistory, train_state = macro_optimization(model, checkpoint_path,
                                                  learning_rate_1, loss_weights,
                                                  iterations_1, callbacks)
    
    
    # Add trainable variables to the callbacks
    callbacks.extend(callback_variables)
    losshistory, train_state = meso_optimization(model, checkpoint_path, train_state,
                                                 learning_rate_2, loss_weights,
                                                 trainable_variables, iterations_2,
                                                 callbacks)
    
    losshistory, train_state = micro_optimization(model, checkpoint_path, train_state,
                                                  loss_weights, trainable_variables,
                                                  callbacks)
    
    # Make sure best model is used for predictions
    model.compile('adam', lr=0, metrics=['MSE'])
    model.restore(os.path.join(checkpoint_path, 'model-' + str(train_state.best_step) + '.ckpt'), verbose=1)
    
    return losshistory, train_state


def prepare_data(loader: DataLoader, case_index: int, point_index: int, pair: int,
                 use_synthetic: bool, noise: float, rng: np.random.RandomState,
                 train_cbf: bool, train_at: bool, train_t1b: bool) -> Tuple[Any]:
    """
    Prepare the data for training.

    Parameters
    ----------
    loader : DataLoader
        The data loader object.
    case_index : int
        The index of the case.
    point_index : int
        The index of the data point.
    pair : int
        The number of pairs.
    use_synthetic : bool
        Whether to use synthetic data.
    noise : float
        The amount of noise to add to the data.
    rng : np.random.RandomState
        The random number generator.
    train_cbf : bool
        Whether to train the cerebral blood flow.
    train_at : bool
        Whether to train the arrival time.
    train_t1b : bool
        Whether to train the T1b value.

    Returns
    -------
    Tuple[Any]
        A tuple containing the prepared data.

    """
    data_points = np.asarray(list(loader.train_generator_index(index=case_index, point_index=point_index,
                                                               verbose=0)))
    
    # Select the 'time' from the points
    observe_x = np.expand_dims(np.insert(data_points[:, 0], 0, 0), axis=-1)
    # Select the perfusion 'signal' from the points
    observe_y = np.expand_dims(np.insert(data_points[:, 1], 0, 0), axis=-1)
    
    base_index = 10
    increment = 3
    perfusion_index = 3
    cbf_index = 1
    at_index = 2
    if pair > 0:
        observe_y_list = [observe_y]
        for i in range(pair):
            index = base_index + perfusion_index + increment * i
            observe_y_temp = np.expand_dims(np.insert(data_points[:, index], 0, 0), axis=-1)
            observe_y_list.append(observe_y_temp)
        observe_y = np.hstack(observe_y_list)
    # Select the config data
    observe_config = data_points[:, :]
    select = list(range(2, 11))
    base_index -= 2
    increment -= 1
    cbf_list = []
    at_list = []
    for i in range(pair):
        index = base_index + cbf_index + increment * i
        select.append(index)    
        cbf_list.append(index)
        index = base_index + at_index + increment * i
        select.append(index) 
        at_list.append(index)
    observe_config = observe_config[:, select]
    
    config = observe_config[0].copy()
    time = np.expand_dims(np.insert(data_points[:, 0], 0, 0), axis=-1)
    t_min = np.min(time)
    t_max = np.max(time)
    alpha = 1.0

    cbf_true = config[0]
    at_true = config[1]
    t1b_true = config[3]

    if pair > 0:
        signal = SignalPairData(FA=config[2], T1b=config[3], T1t=config[4], blood_lambda=config[5],
                                bolus_duration=config[6], dt=config[7], M0b=config[8], a=alpha,
                                cerebral_blood_flow=config[0], cerebral_blood_flow_pair=config[[cbf_list]],
                                arrival_time=config[1], arrival_time_pair=config[at_list], min_t=t_min, max_t=t_max)
    else:
        signal = SignalData(FA=config[2], T1b=config[3], T1t=config[4], blood_lambda=config[5],
                            bolus_duration=config[6], dt=config[7], M0b=config[8], a=alpha,
                            cerebral_blood_flow=config[0], arrival_time=config[1], min_t=t_min, max_t=t_max)
    
    if use_synthetic:
        observe_y_buxton = signal.buxton_signal_continuous(time)
        if pair == 0:
            np.expand_dims(observe_y_buxton, axis=-1)
        observe_y = observe_y_buxton
    if noise > 0.0:
        noise_seed = rng.randint(low=0, high=10000)
        # Loop over signal to make sure noise is consistent across number of pairs
        for i in range(observe_y.shape[-1]):
            observe_y[:, i] = signal.add_noise(signal=observe_y[:, i],
                                               time=observe_x[:, 0],
                                               mean=0.0,
                                               std=noise,
                                               seed=noise_seed)
    
    data_y = np.copy(observe_y)
    
    if train_cbf:
        observe_config[:, 0] = 0.5
        for i in range(pair):
            index = base_index + cbf_index + increment * i
            observe_config[:, index] = 0.5
    
    if train_at:
        observe_config[:, 1] = 0.5
        for i in range(pair):
            index = base_index + at_index + increment * i
            observe_config[:, index] = 0.5
            
    if train_t1b:
        observe_config[:, 3] = 2.0
        
    return (observe_x, observe_y, observe_config, time, t_min, t_max, alpha, signal,
            cbf_true, at_true, t1b_true, config, data_y)
        
        
def run_lsf(train_cbf: bool, train_at: bool, train_t1b: bool, time: np.ndarray,
            observe_y: np.ndarray, observe_config: np.ndarray) -> Tuple[float]:
    """
    Run the LSF (Least Squares Fit) algorithm to estimate the values of CBF, AT, and T1b.

    Parameters
    ----------
    train_cbf : bool
        Flag indicating whether to train the CBF parameter.
    train_at : bool
        Flag indicating whether to train the AT parameter.
    train_t1b : bool
        Flag indicating whether to train the T1b parameter.
    time : np.ndarray
        Array of time values.
    observe_y : np.ndarray
        Array of observed y values.
    observe_config : np.ndarray
        Array of observed configuration values.

    Returns
    -------
    cbf_lsf : float
        Estimated value of CBF.
    at_lsf : float
        Estimated value of AT.
    t1b_lsf : float
        Estimated value of T1b.
    """
    t_start = time_measure.time()
    asl_fit = ASLFit(train_cbf=train_cbf, train_at=train_at, train_t1b=train_t1b)
    lsf_data_points = np.hstack((time[1:], observe_y[1:, 0:1], observe_config))
    lsf_data_points = asl_fit.fit_and_update(lsf_data_points)
    t_end = time_measure.time()
    
    t_total = t_end - t_start
    print('LSF execution (s): ', t_total)
    
    cbf_lsf = lsf_data_points[0, 2]
    at_lsf = lsf_data_points[0, 3]
    t1b_lsf = lsf_data_points[0, 5]
    
    return cbf_lsf, at_lsf, t1b_lsf
    

def run_pinn(train_cbf: bool, train_at: bool, train_t1b: bool, observe_x: np.ndarray,
             observe_y: np.ndarray, observe_config: np.ndarray, t_min: float,
             t_max: float, alpha: float, signal: np.ndarray, pair: int,
             use_input_transform: bool, use_output_transform: bool,
             case_path: str) -> Tuple[dde.model.Model, Union[SystemDynamics, SystemPairDynamics],
                                      dde.model.TrainState, str]:
    """
    Run the PINN training process.

    Parameters
    ----------
    train_cbf : bool
        Whether to train the CBF component of the system dynamics.
    train_at : bool
        Whether to train the AT component of the system dynamics.
    train_t1b : bool
        Whether to train the T1b component of the system dynamics.
    observe_x : np.ndarray
        The observed x values.
    observe_y : np.ndarray
        The observed y values.
    observe_config : np.ndarray
        The observed configuration values.
    t_min : float
        The minimum time value.
    t_max : float
        The maximum time value.
    alpha : float
        The alpha value.
    signal : np.ndarray
        The signal values.
    pair : int
        The pair value.
    use_input_transform : bool
        Whether to use input transformation.
    use_output_transform : bool
        Whether to use output transformation.
    case_path : str
        The path to the case.

    Returns
    -------
    model : dde.model.Model
        The trained PINN model.
    dynamics : Union[SystemDynamics, SystemPairDynamics]
        The system dynamics.
    train_state : dde.model.TrainState
        The training state.
    checkpoint_path : str
        The path to the model checkpoint.

    """
    if pair > 0:
        dynamics = SystemPairDynamics()
    else:
        dynamics = SystemDynamics()
    
    dynamics.set_configuration(observe_config, t_min, t_max, alpha,
                               is_cbf_trainable=train_cbf, is_at_trainable=train_at,
                               is_t1b_trainable=train_t1b)
    

    checkpoint_path = os.path.join(case_path, 'model_checkpoint')
    (callbacks, callback_variables,
     trainable_variables) = get_callbacks(dynamics, train_cbf, train_at,
                                          train_t1b, pair, checkpoint_path,
                                          case_path)
    
    
    ode = dynamics.ode_system_continuous
    solution = signal.buxton_signal_continuous
    
    # Prepare the initial conditions
    bcs = []
    bcs.append(dde.icbc.PointSetBC(observe_x, observe_y[:, 0:1], component=0, shuffle=False))
    for i in range(1, pair+1):
        bcs.append(dde.icbc.PointSetBC(observe_x, observe_y[:, i:i+1], component=i, shuffle=False))
    
    data = dde.data.PDE(geometry=dynamics.time_domain(), pde=ode,
                        bcs=bcs, num_domain=observe_x.shape[0], num_boundary=10,
                        anchors=observe_x, solution=solution, num_test=10)

    if pair > 0:
        net = PairPINN().get_network(pair=pair, input_transform=use_input_transform,
                                     output_transform=use_output_transform)
    else:
        net = PINN().get_network(input_transform=use_input_transform,
                                 output_transform=use_output_transform)
    model = dde.Model(data, net)
    
    
    t_start = time_measure.time()
    
    
    losshistory, train_state = multi_optimization(model, pair, callbacks,
                                                  trainable_variables,
                                                  callback_variables,
                                                  checkpoint_path)
    
    t_end = time_measure.time()
    t_total = t_end - t_start
    print('PINN execution (s): ', t_total)
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=case_path)
    dde.utils.external.plot_loss_history(losshistory, fname=os.path.join(case_path, 'loss_history.png'))
    plt.close()
    
    return model, dynamics, train_state, checkpoint_path
    
    
def plot_best_state(train_state: dde.model.TrainState, y_train_true: np.ndarray,
                    gen_lsf: SignalData = None, gen_pinn: Union[SignalData, SignalPairData] = None,
                    title: str = '', case_path: str = ''):
    """
    Plot the best result of the smallest training loss.

    Parameters
    ----------
    train_state : dde.model.TrainState
        The training state object containing information about the training process.
    y_train_true : np.ndarray
        The true values of the target variable used for training.
    gen_lsf : SignalData, optional
        The generated LSF (Least Squares Fit) signal data. Default is None.
    gen_pinn : Union[SignalData, SignalPairData], optional
        The generated PINN (Physics-Informed Neural Network) signal data. Default is None.
    title : str, optional
        The title of the plot. Default is an empty string.
    case_path : str, optional
        The path where the plot file will be saved. Default is an empty string.

    Returns
    -------
    None

    """
    if isinstance(train_state.X_train, (list, tuple)):
        print(
            "Error: The network has multiple inputs, and plotting such result han't been implemented."
        )
        return

    y_train, y_test, best_y, best_ystd = dde.utils.external._pack_data(train_state)
    y_dim = best_y.shape[1]

    # Regression plot
    # 1D
    idx = np.argsort(train_state.X_test[:, 0])
    X = train_state.X_test[idx, 0]
    y_lsf_test = None
    if gen_lsf is not None:
        time = np.expand_dims(X, axis=-1)
        y_lsf_test = gen_lsf.buxton_signal_continuous(time)
    if gen_pinn is not None:
        time = np.expand_dims(X, axis=-1)
        y_pinn_test = gen_pinn.buxton_signal_continuous(time)
    
    
    for i in range(y_dim):
        plt.figure(figsize=(12, 10))
        
        plt.scatter(y_train_true[0][:, 0], y_train_true[1][:, i], s=46, c='b', marker='x', label='Measured PWI Signal')
        if y_test is not None:
            plt.plot(X, y_test[idx, i], '-k', alpha=0.75, label='True')
        if y_lsf_test is not None:
            plt.plot(X, y_lsf_test[:, 0], '--m', alpha=0.75, label='LSF')
        if y_pinn_test is not None:
            plt.plot(X, y_pinn_test[:, 0], '-.g', alpha=0.75, label='PINN (Inv)')
        plt.plot(X, best_y[idx, i], '-r', label='PINN (Fwd)')
        if best_ystd is not None:
            plt.plot(X, best_y[idx, i] + 2 * best_ystd[idx, i], '-b', label='95% CI')
            plt.plot(X, best_y[idx, i] - 2 * best_ystd[idx, i], '-b')
    
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized PWI Signal')
        plt.legend(loc='upper left')
        
        plt.title(title)
        
        if i == 0:
            file = 'prediction.png'
        else:
            file = 'prediction_{}'.format(i+1)
        plt.savefig(os.path.join(case_path, file))
        plt.close()

        
def dat_to_csv(dat_file_path: str, csv_file_path: str, columns: List[str]) -> None:
    """
    Converts a dat file to CSV format and saves it.

    Parameters
    ----------
    dat_file_path : str
        Path of the dat file.
    csv_file_path : str
        Desired path of the CSV file.
    columns : List[str]
        Column names to be added in the CSV file.

    Returns
    -------
    None

    """
    with open(dat_file_path, "r", encoding="utf-8") as dat_file, open(
        csv_file_path, "w", encoding="utf-8", newline=""
    ) as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(columns)
        for line in dat_file:
            if "#" in line:
                continue
            row = []
            for field in line.split(" "):
                field = field.strip()
                field = field.replace('[', '')
                field = field.replace(']', '')
                field = field.replace(',', '')
                row.append(field)
            csv_writer.writerow(row)
        

def merge_dicts(*dicts) -> Dict[Any, Any]:
    """
    Merge multiple dictionaries into a single dictionary.

    Parameters
    ----------
    *dicts : dict
        The dictionaries to be merged.

    Returns
    -------
    dict
        The merged dictionary.

    """
    merged_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = [value]
            else:
                merged_dict[key].append(value)
    return merged_dict


def csv_to_dict(case_path: str) -> Dict[Any, Any]:
    """
    Convert a CSV file to a dictionary.

    Parameters
    ----------
    case_path : str
        The path to the directory containing the CSV file.

    Returns
    -------
    Dict[Any, Any]
        A dictionary containing the data from the CSV file.

    """
    rows = []
    with open(os.path.join(case_path, 'variables.csv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in row.keys():
                if key == 'Epochs':
                    row[key] = int(row[key])
                else:
                    row[key] = float(row[key])
            rows.append(row)
    return merge_dicts(*rows)


def get_variables(case_path: str, columns: List[str], dynamics: Union[SystemDynamics, SystemPairDynamics],
                  pair: int) -> Dict[Any, Any]:
    """
    Retrieve variables from a given case path and return them as a dictionary.

    Parameters
    ----------
    case_path : str
        The path to the case directory.
    columns : List[str]
        The list of column names for the variables.
    dynamics : Union[SystemDynamics, SystemPairDynamics]
        The dynamics of the system.
    pair : int
        The pair number.

    Returns
    -------
    Dict[Any, Any]
        A dictionary containing the retrieved variables.

    """
    dat_to_csv(dat_file_path=os.path.join(case_path, 'variables.dat'),
               csv_file_path=os.path.join(case_path, 'variables.csv'),
               columns=columns)
    data = csv_to_dict(case_path)
    
    return data
    

def save_cbf_plot(variables: List[str], cbf_true: float, cbf_pred: float,
                  cbf_lsf: float, best_iteration: int, title: str, title_lsf: str,
                  case_path: str, output_folder: str) -> None:
    """
    Save a plot of the CBF values.

    Parameters
    ----------
    variables : List[str]
        A list of variables containing the epochs and CBF values.
    cbf_true : float
        The true CBF value.
    cbf_pred : float
        The predicted CBF value.
    cbf_lsf : float
        The least squares fitted CBF value.
    best_iteration : int
        The best iteration number.
    title : str
        The title for the plot.
    title_lsf : str
        The title for the least squares fitted plot.
    case_path : str
        The path to the case.
    output_folder : str
        The folder where the plot will be saved.

    Returns
    -------
    None

    """
    plt.plot(variables['Epochs'], variables['CBF'], '-b', label='Trained')
    plt.plot(variables['Epochs'], cbf_true, '--k', label='True')
    plt.plot(variables['Epochs'], np.abs(np.asarray(cbf_true)-np.asarray(variables['CBF'])),
             '-.r', alpha=0.5, label='Error')
    if best_iteration is not None:
        plt.axvline(x=best_iteration)
    plt.title('True: {:.4f}\n{}: {:.4f} {}: {:.4f}'.format(cbf_true[0], title, cbf_pred,
                                                           title_lsf, cbf_lsf))
    plt.xlabel('Epoch')
    plt.ylabel('CBF')
    plt.legend()
    plt.savefig(os.path.join(case_path, output_folder))
    plt.close()


def save_at_plot(variables: List[str], at_true: float, at_pred: float,
                 at_lsf: float, best_iteration: int, title: str, title_lsf: str,
                 case_path: str, output_folder: str) -> None:
    """
    Save a plot of the AT values.

    Parameters
    ----------
    variables : List[str]
        A list of variables.
    at_true : float
        The true AT value.
    at_pred : float
        The predicted AT value.
    at_lsf : float
        The AT value from the least squares fit.
    best_iteration : int
        The best iteration number.
    title : str
        The title for the plot.
    title_lsf : str
        The title for the least squares fit.
    case_path : str
        The path to the case.
    output_folder : str
        The output folder to save the plot.

    Returns
    -------
    None

    """
    plt.plot(variables['Epochs'], variables['AT'], '-b', label='Trained')
    plt.plot(variables['Epochs'], at_true, '--k', label='True')
    plt.plot(variables['Epochs'], np.abs(np.asarray(at_true)-np.asarray(variables['AT'])),
             '-.r', alpha=0.5, label='Error')
    if best_iteration is not None:
        plt.axvline(x=best_iteration)
    plt.title('True: {:.4f}\n{}: {:.4f} {}: {:.4f}'.format(at_true[0], title, at_pred,
                                                           title_lsf, at_lsf))
    plt.xlabel('Epoch')
    plt.ylabel('AT')
    plt.legend()
    plt.savefig(os.path.join(case_path, output_folder))
    plt.close()
    

def save_t1b_plot(variables: List[str], t1b_true: float, t1b_pred: float,
                  best_iteration: int, title: str, case_path: str,
                  output_folder: str) -> None:
    """
    Save a plot of T1b values.

    Parameters
    ----------
    variables : List[str]
        A list of variables.
    t1b_true : float
        The true T1b value.
    t1b_pred : float
        The predicted T1b value.
    best_iteration : int
        The best iteration.
    title : str
        The title of the plot.
    case_path : str
        The path to the case.
    output_folder : str
        The output folder.

    Returns
    -------
    None

    """
    plt.plot(variables['Epochs'], variables['T1b'], '-b', label='Trained')
    plt.plot(variables['Epochs'], t1b_true, '--k', label='True')
    plt.plot(variables['Epochs'], np.abs(np.asarray(t1b_true)-np.asarray(variables['T1b'])),
             '-.r', alpha=0.5, label='Error')
    if best_iteration is not None:
        plt.axvline(x=best_iteration)
    plt.title('True: {:.4f}\n{}: {:.4f}'.format(t1b_true[0], title, t1b_pred))
    plt.xlabel('Epoch')
    plt.ylabel('T1b')
    plt.legend()
    plt.savefig(os.path.join(case_path, output_folder))
    plt.close()
    
    
def calculate_metrics(model: dde.model.Model, true_cbf: float, true_at: float,
                      true_t1b: float, predicted_cbf: float, predicted_at: float, 
                      predicted_t1b: float, lsf_cbf: float, lsf_at: float,
                      lsf_t1b: float, config: np.ndarray, alpha: float,
                      min_t: float, max_t: float, output_path: str) -> None:
    """
    Calculate metrics for evaluating the performance of a model.

    Parameters
    ----------
    model : dde.model.Model
        The trained model used for prediction.
    true_cbf : float
        The true cerebral blood flow value.
    true_at : float
        The true arrival time value.
    true_t1b : float
        The true T1b value.
    predicted_cbf : float
        The predicted cerebral blood flow value.
    predicted_at : float
        The predicted arrival time value.
    predicted_t1b : float
        The predicted T1b value.
    lsf_cbf : float
        The LSF cerebral blood flow value.
    lsf_at : float
        The LSF arrival time value.
    lsf_t1b : float
        The LSF T1b value.
    config : np.ndarray
        The configuration array containing various parameters.
    alpha : float
        The alpha value.
    min_t : float
        The minimum time value.
    max_t : float
        The maximum time value.
    output_path : str
        The path to save the results.

    Returns
    -------
    None

    """
    true_function = SignalData(FA=config[2], T1b=config[3], T1t=config[4],
                               blood_lambda=config[5], bolus_duration=config[6],
                               dt=config[7], M0b=config[8], a=alpha,
                               cerebral_blood_flow=true_cbf,
                               arrival_time=true_at,
                               min_t=min_t, max_t=max_t)
    
    lsf_function = SignalData(FA=config[2], T1b=config[3], T1t=config[4],
                              blood_lambda=config[5], bolus_duration=config[6],
                              dt=config[7], M0b=config[8], a=alpha,
                              cerebral_blood_flow=lsf_cbf,
                              arrival_time=lsf_at,
                              min_t=min_t, max_t=max_t)
    
    
    t = np.linspace(start=min_t, stop=max_t, num=1000, endpoint=True)
    
    true_signal = np.asarray(true_function.buxton_signal_continuous(t, qp_to_1=True, smoothness=0.01), dtype=np.float32)
    predicted_signal = model.predict(np.expand_dims(t, axis=-1)) #predicted_function.buxton_signal_continuous(t, qp_to_1=True, smoothness=0.01)
    lsf_signal = np.asarray(lsf_function.buxton_signal_continuous(t, qp_to_1=True, smoothness=0.01), dtype=np.float32)
    
    if len(predicted_signal.shape) == 2:
        predicted_signal = predicted_signal[:, 0]
    predicted_signal = predicted_signal.flatten()
    
    true_predicted_mse = (np.square(true_signal - predicted_signal))
    true_predicted_mse_std = true_predicted_mse.std()
    true_predicted_mse = true_predicted_mse.mean()
    
    true_lsf_mse = (np.square(true_signal - lsf_signal))
    true_lsf_mse_std = true_lsf_mse.std()
    true_lsf_mse = true_lsf_mse.mean()
    
    relative_predicted_error_cbf = (predicted_cbf - true_cbf) / true_cbf * 100
    relative_predicted_error_at = (predicted_at - true_at) / true_at * 100
    relative_predicted_error_t1b = (predicted_t1b - true_t1b) / true_t1b * 100
    
    relative_lsf_error_cbf = (lsf_cbf - true_cbf) / true_cbf * 100
    relative_lsf_error_at = (lsf_at - true_at) / true_at * 100
    relative_lsf_error_t1b = (lsf_t1b - true_t1b) / true_t1b * 100
    
    results = {'mse_pred': true_predicted_mse, 'mse_pred_std': true_predicted_mse_std,
               're_pred_cbf': relative_predicted_error_cbf, 're_pred_at': relative_predicted_error_at, 're_pred_t1b': relative_predicted_error_t1b,
               'mse_lsf': true_lsf_mse, 'mse_lsf_std': true_lsf_mse_std,
               're_lsf_cbf': relative_lsf_error_cbf, 're_lsf_at': relative_lsf_error_at, 're_lsf_t1b': relative_lsf_error_t1b,
               'true_cbf': true_cbf, 'true_at': true_at, 'true_t1b': true_t1b,
               'pred_cbf': predicted_cbf, 'pred_at': predicted_at, 'pred_t1b': predicted_t1b,
               'lsf_cbf': lsf_cbf, 'lsf_at': lsf_at, 'lsf_t1b': lsf_t1b}
    
    for key, value in results.items():
        if isinstance(value, np.float32) or isinstance(value, np.float64):
            results[key] = float(value)
    
    # Save dictionary to a file
    output_file = os.path.join(output_path, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    
def get_best_variables(variables: List[tf.Variable], best_step: int, init_cbf: float,
                       init_at: float, init_t1b: float) -> Tuple[float]:
    """
    Get the best values for the variables based on the given best step.

    Parameters
    ----------
    variables : List[tf.Variable]
        A list of TensorFlow variables.
    best_step : int
        The best step to find the variables for.
    init_cbf : float
        The initial value for CBF (Cerebral Blood Flow).
    init_at : float
        The initial value for AT (Arterial Transit Time).
    init_t1b : float
        The initial value for T1b (Blood T1 Relaxation Time).

    Returns
    -------
    Tuple[float]
        A tuple containing the best values for CBF, AT, and T1b.

    """
    def set_bounds(val):
        val = max(1e-2, val)
        val = min(2.5, val)
        return val
    
    # Initial values
    cbf = init_cbf
    at = init_at
    t1b = init_t1b
    
    # Find if best epoch exists
    for i in range(len(variables['Epochs'])):
        if variables['Epochs'][i] == best_step:
            if 'CBF' in variables:
                cbf = variables['CBF'][i]
            if 'AT' in variables:
                at = variables['AT'][i]
            if 'T1b' in variables:
                t1b = variables['T1b'][i]
            break
    
    return set_bounds(cbf), set_bounds(at), set_bounds(t1b)


def save_results(train_cbf: bool, train_at: bool, train_t1b: bool, case_path: str,
                 columns: List[str], dynamics: Union[SystemDynamics, SystemPairDynamics],
                 pair: int, model: dde.model.Model, train_state: dde.model.TrainState,
                 config: np.ndarray, observe_config: np.ndarray, observe_x: np.ndarray,
                 data_y: np.ndarray, cbf_true: float, at_true: float, t1b_true: float,
                 cbf_lsf: float, at_lsf: float, t1b_lsf: float, t_min: float,
                 t_max: float, alpha: float) -> None:
    """
    Save the results of the training process.

    Parameters
    ----------
    train_cbf : bool
        Flag indicating whether to train the cerebral blood flow (CBF) model.
    train_at : bool
        Flag indicating whether to train the arrival time (AT) model.
    train_t1b : bool
        Flag indicating whether to train the T1b model.
    case_path : str
        Path to the case directory.
    columns : List[str]
        List of column names.
    dynamics : Union[SystemDynamics, SystemPairDynamics]
        System dynamics or system pair dynamics.
    pair : int
        Pair index.
    model : dde.model.Model
        Model object.
    train_state : dde.model.TrainState
        Train state object.
    config : np.ndarray
        Configuration array.
    observe_config : np.ndarray
        Observed configuration array.
    observe_x : np.ndarray
        Observed x array.
    data_y : np.ndarray
        Data y array.
    cbf_true : float
        True cerebral blood flow (CBF) value.
    at_true : float
        True arrival time (AT) value.
    t1b_true : float
        True T1b value.
    cbf_lsf : float
        Least squares fit (LSF) cerebral blood flow (CBF) value.
    at_lsf : float
        Least squares fit (LSF) arrival time (AT) value.
    t1b_lsf : float
        Least squares fit (LSF) T1b value.
    t_min : float
        Minimum time value.
    t_max : float
        Maximum time value.
    alpha : float
        Alpha value.

    Returns
    -------
    None

    """
    if train_cbf or train_at or train_t1b:
        variables = get_variables(case_path, columns, dynamics, pair)
        cbf, at, t1b = get_best_variables(variables, train_state.best_step,
                                          init_cbf=observe_config[0, 0],
                                          init_at=observe_config[0, 1],
                                          init_t1b=observe_config[0, 3])
        
        if train_cbf:
            save_cbf_plot(variables, [cbf_true] * len(variables['CBF']),
                          cbf, cbf_lsf, train_state.best_step, 'PINN Pred', 
                          'LSF Pred', case_path, 'cbf.png')
        if train_at:
            save_at_plot(variables, [at_true] * len(variables['AT']),
                         at, at_lsf, train_state.best_step, 'PINN Pred', 
                         'LSF Pred', case_path, 'at.png')
            
        if train_t1b:
            save_t1b_plot(variables, [t1b_true] * len(variables['T1b']),
                          t1b, train_state.best_step, 'PINN Pred', 
                          case_path, 't1b.png')
            

        updated_lsf_data = SignalData(FA=config[2], T1b=t1b_lsf, T1t=config[4],
                                      blood_lambda=config[5], bolus_duration=config[6],
                                      dt=config[7], M0b=config[8], a=alpha,
                                      cerebral_blood_flow=cbf_lsf,
                                      arrival_time=at_lsf,
                                      min_t=t_min, max_t=t_max)
        updated_pinn_data = SignalData(FA=config[2], T1b=t1b, T1t=config[4],
                                      blood_lambda=config[5], bolus_duration=config[6],
                                      dt=config[7], M0b=config[8], a=alpha,
                                      cerebral_blood_flow=cbf,
                                      arrival_time=at,
                                      min_t=t_min, max_t=t_max)
        
        
        title = ('CBF - True: {:.4f}, PINN: {:.4f}, LSF: {:.4f}\n'
                 'AT  - True: {:.4f}, PINN: {:.4f}, LSF: {:.4f}\n'
                 'T1b - True: {:.4f}, PINN: {:.4f}, LSF: {:.4f}').format(
                 cbf_true, cbf, cbf_lsf,
                 at_true, at, at_lsf,
                 t1b_true, t1b, t1b_lsf)
        plot_best_state(train_state, [observe_x, data_y], gen_lsf=updated_lsf_data,
                        gen_pinn=updated_pinn_data, title=title, case_path=case_path)
                              
        calculate_metrics(model, true_cbf=cbf_true, true_at=at_true,
                          true_t1b=t1b_true, predicted_cbf=cbf,
                          predicted_at=at, predicted_t1b=t1b, lsf_cbf=cbf_lsf,
                          lsf_at=at_lsf, lsf_t1b=t1b_lsf, config=config, alpha=alpha,
                          min_t=t_min, max_t=t_max, output_path=case_path)
    else:
        plot_best_state(train_state, [observe_x, data_y], case_path=case_path)
        

def cleanup_models(path: str, best_iteration: int) -> None:
    """
    Remove unnecessary model files from the specified path, except for the best model.

    Parameters
    ----------
    path : str
        The path to the directory containing the model files.
    best_iteration : int
        The iteration number of the best model to be kept.

    Returns
    -------
    None

    """
    files = sorted(os.listdir(path))
    for file in files:
        if file == 'checkpoint':
            continue
        if file.startswith('model-{}'.format(best_iteration)):
            continue
        os.remove(os.path.join(path, file))

    
def run_model() -> None:
    """
    Run the PINNs/LSF models and save the results.

    Returns
    -------
    None

    """
    args = parse_args()
    
    seed = args.seed
    set_global_seed(seed)
    
    use_synthetic = args.use_synthetic
    noise = args.noise
    
    use_input_transform = args.use_input_transform
    use_output_transform = args.use_output_transform
    
    spatial_average = args.spatial_average
    spatial_weight = args.spatial_weight
    pair = args.pair
    
    train_cbf = args.train_cbf
    train_at = args.train_at
    train_t1b = args.train_t1b
    
    cases = args.cases
    
    rng = np.random.RandomState(156)

    data_path = os.path.join('..', 'data_mat')
    output_folder, columns = get_output_folder(args)
    
    if cases is None:
        max_cases = 7
        cases = list(range(max_cases))
        
    
    output_folder += '_' + str(seed)
    base_output_path = os.path.join('..', output_folder)
    
    print(base_output_path)
    
    if use_synthetic:
        cases = [2]
    
    for case_index in cases:
        
        loader = DataLoader(data_path=data_path, memory_cache=True, units='s',
                            normalize=True, spatial_average=spatial_average, 
                            spatial_weight=spatial_weight, pair=pair)
        max_points = loader.prepare_generator_index(index=case_index, verbose=0)
        if use_synthetic:
            max_points = min(40, max_points)
            
        for point_index in range(max_points): 
            case_path = os.path.join(base_output_path, '{:02d}'.format(case_index), '{:02d}'.format(point_index))
            os.makedirs(case_path, exist_ok=True)
            
            (observe_x, observe_y, observe_config,
             time, t_min, t_max, alpha, signal,
             cbf_true, at_true, t1b_true,
             config, data_y) = prepare_data(loader, case_index, point_index,
                                            pair, use_synthetic, noise,
                                            rng, train_cbf, train_at, train_t1b)
            
                                            
            cbf_lsf, at_lsf, t1b_lsf = run_lsf(train_cbf, train_at, train_t1b,
                                               time, observe_y, observe_config)
            
            (model, dynamics,
             train_state, checkpoint_path) = run_pinn(train_cbf, train_at, train_t1b,
                                                      observe_x, observe_y, observe_config,
                                                      t_min, t_max, alpha, signal, pair,
                                                      use_input_transform, use_output_transform,
                                                      case_path)
            
                                                      
            save_results(train_cbf, train_at, train_t1b, case_path, columns, dynamics, pair,
                             model, train_state, config, observe_config, observe_x, data_y,
                             cbf_true, at_true, t1b_true, cbf_lsf, at_lsf, t1b_lsf,
                             t_min, t_max, alpha)                                          
            
            cleanup_models(checkpoint_path, train_state.best_step)
    

if __name__ == '__main__':
    run_model()
    
