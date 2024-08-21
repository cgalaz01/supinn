import os
os.environ['DDE_BACKEND'] = 'tensorflow'

from typing import List

import numpy as np

import deepxde as dde
from deepxde.backend import tf


class SystemDynamics():
    
    def __init__(self):
        """
        Initialize the class.

        Returns
        -------
        None

        """
        # Initialise with default parameters
        self.FA_degrees = 35
        self.T1b_ms = 1700        
        self.T1t_ms = 1330      
        self.blood_lambda = 0.98
        self.bolus_duration_ms = 700    
        self.dt_ms = 600
        self.M0b_au = 1000
        self.a = 0.91
        
        self.cerebral_blood_flow = 0
        self.arrival_time = 0
        
        self.min_t_ms = 150
        self.max_t_ms = 3450
        
    
    def set_configuration(self, config: np.ndarray, t_min: float, t_max: float, alpha: float,
                          is_cbf_trainable: bool = False, is_at_trainable: bool = False,
                          is_t1b_trainable: bool = False, smoothness: float = 0.1) -> None:
        """
        Set the configuration parameters for the model.

        Parameters
        ----------
        config : np.ndarray
            The configuration array containing the model parameters.
        t_min : float
            The minimum time value.
        t_max : float
            The maximum time value.
        alpha : float
            The alpha value.
        is_cbf_trainable : bool, optional
            Flag indicating whether the cerebral blood flow parameter is trainable. Default is False.
        is_at_trainable : bool, optional
            Flag indicating whether the arrival time parameter is trainable. Default is False.
        is_t1b_trainable : bool, optional
            Flag indicating whether the T1b parameter is trainable. Default is False.
        smoothness : float, optional
            The smoothness parameter. Default is 0.1.

        Returns
        -------
        None

        """
        config = tf.cast(config, tf.float32)
        
        # Get only 1 point as they should be the same (broadcast)
        if is_cbf_trainable:
            self.cerebral_blood_flow = dde.Variable(config[0, 0])
        else:
            self.cerebral_blood_flow = config[0, 0]
            
        if is_at_trainable:
            self.arrival_time = dde.Variable(config[0, 1])
        else:
            self.arrival_time = config[0, 1]
        self.FA_rads = np.deg2rad(config[0, 2])
        if is_t1b_trainable:   
            self.T1b_ms = dde.Variable(config[0, 3])
        else:
            self.T1b_ms = config[0, 3]
        self.T1t_ms = config[0, 4]
        self.blood_lambda = config[0, 5]
        self.bolus_duration_ms = config[0, 6]
        self.dt_ms = config[0, 7]
        self.M0b_au = config[0, 8]
        
        self.a = alpha
        
        self.min_t_ms = t_min
        self.max_t_ms = t_max
        
        self.smoothness = smoothness
        
        self.is_cbf_trainable = is_cbf_trainable
        self.is_at_trainable = is_at_trainable
        self.is_t1b_trainable = is_t1b_trainable
        
        
    def time_domain(self) -> dde.geometry.TimeDomain:
        """
        Create a time domain geometry object.

        Returns
        -------
        geometry : dde.geometry.TimeDomain
            The time domain geometry object.

        """
        geometry = dde.geometry.TimeDomain(self.min_t_ms, self.max_t_ms)
        return geometry
    
    
    @staticmethod
    def get_variable(var: tf.Tensor) -> tf.Tensor:
        """
        Apply the ReLU activation function to the given tensor.

        Parameters
        ----------
        var : tf.Tensor
            The input tensor.

        Returns
        -------
        tf.Tensor
            The tensor after applying the ReLU activation function.

        """
        return tf.keras.activations.relu(var, threshold=1e-2, max_value=2.5)
    
    
    @staticmethod
    def ode(time: tf.Tensor, cerebral_blood_flow: tf.Variable, arrival_time: tf.Variable,
            bolus_duration_ms: tf.Variable, T1b_ms: tf.Variable) -> tf.Tensor:
        """
        Compute the output of the Ordinary Differential Equation (ODE) for modeling cerebral blood flow.

        Parameters
        ----------
        time : tf.Tensor
            The time values at which to evaluate the ODE.
        cerebral_blood_flow : tf.Variable
            The cerebral blood flow value.
        arrival_time : tf.Variable
            The arrival time of the bolus.
        bolus_duration_ms : tf.Variable
            The duration of the bolus in milliseconds.
        T1b_ms : tf.Variable
            The relaxation time of blood in milliseconds.

        Returns
        -------
        results : tf.Tensor
            The computed results of the ODE.

        """
        cond_1 = tf.math.less(time, arrival_time)
        cond_3 = tf.math.less_equal(arrival_time + bolus_duration_ms, time)
        
        case_lower = tf.zeros_like(time, dtype=tf.float32)
        case_range = cerebral_blood_flow * tf.math.exp(-time / T1b_ms) * (1.0 - (time - arrival_time) / T1b_ms)
        case_upper = -cerebral_blood_flow * bolus_duration_ms / T1b_ms * tf.math.exp(-time / T1b_ms)
        
        results = tf.where(cond_1, tf.cast(case_lower, tf.float32), tf.cast(case_range, tf.float32))
        results = tf.where(cond_3, tf.cast(case_upper, tf.float32), tf.cast(results, tf.float32))
        
        return results
    
    
    def ode_system(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Compute the error between the time derivative of the signal and the ODE model.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor representing time.
        y : tf.Tensor
            The input tensor representing the signal.

        Returns
        -------
        error : tf.Tensor
            The tensor representing the error between the time derivative of the signal and the ODE model.

        """
        time = x
        signal = y
        
        ds_dt = dde.grad.jacobian(signal, time, i=0)
        
        if self.is_cbf_trainable:
            cbf = SystemDynamics.get_variable(self.cerebral_blood_flow)
        else:
            cbf = self.cerebral_blood_flow
        if self.is_at_trainable:
            at = SystemDynamics.get_variable(self.arrival_time)
        else:
            at = self.arrival_time
        if self.is_t1b_trainable:
            t1b = SystemDynamics.get_variable(self.T1b_ms)
        else:
            t1b = self.T1b_ms
        
        error = ds_dt - self.ode(time, cbf, at,
                                 self.bolus_duration_ms, t1b)
        
        return error

    
    @staticmethod
    def shift_tanh(x: tf.Tensor, shift: float = 0, smooth: float = 1) -> tf.Tensor:
        """
        Applies a shifted and smoothed hyperbolic tangent function to the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        shift : float, optional
            The shift value to be applied to the input tensor. The default is 0.
        smooth : float, optional
            The smoothing factor to be applied to the input tensor. The default is 1.

        Returns
        -------
        shifted : tf.Tensor
            The tensor with the shifted and smoothed hyperbolic tangent function applied.

        """
        return 0.5 + 0.5 * tf.math.tanh((x - shift) / smooth)
    
    
    @staticmethod
    def inverse_shift_tanh(x: tf.Tensor, shift: float = 0, smooth: float = 1) -> tf.Tensor:
        """
        Applies the inverse shift tanh function to the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        shift : float, optional
            The shift value. The default is 0.
        smooth : float, optional
            The smoothness parameter. The default is 1.

        Returns
        -------
        shifted : tf.Tensor
            The transformed tensor.

        """
        return 0.5 - 0.5 * tf.math.tanh((x - shift) / smooth)
    
    
    @staticmethod
    def ode_continuous(time: tf.Tensor, cerebral_blood_flow: tf.Variable,
                        arrival_time: tf.Variable, bolus_duration_ms: tf.Variable,
                        T1b_ms: tf.Variable, smoothness: float) -> tf.Tensor:
        """
        Calculate the continuous solution of the ordinary differential equation (ODE) for cerebral blood flow.

        Parameters
        ----------
        time : tf.Tensor
            The time points at which to evaluate the ODE.
        cerebral_blood_flow : tf.Variable
            The cerebral blood flow variable.
        arrival_time : tf.Variable
            The arrival time of the bolus.
        bolus_duration_ms : tf.Variable
            The duration of the bolus in milliseconds.
        T1b_ms : tf.Variable
            The relaxation time of blood in milliseconds.
        smoothness : float
            The smoothness parameter for the shift-tanh function.

        Returns
        -------
        results : tf.Tensor
            The continuous solution of the ODE.

        """
        t_range = SystemDynamics.shift_tanh(time, arrival_time, smoothness) * \
            SystemDynamics.inverse_shift_tanh(time, bolus_duration_ms + arrival_time, smoothness)
        case_range = cerebral_blood_flow * tf.math.exp(-time / T1b_ms) * (1.0 - (time - arrival_time) / T1b_ms)
        
        t_upper = SystemDynamics.shift_tanh(time, bolus_duration_ms + arrival_time, smoothness)
        case_upper = -cerebral_blood_flow * bolus_duration_ms / T1b_ms * tf.math.exp(-time / T1b_ms)
        
        results = t_range * case_range + t_upper * case_upper
        
        return results

    
    def ode_system_continuous(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Compute the continuous ordinary differential equation (ODE) system.

        Parameters
        ----------
        x : tf.Tensor
            The independent variable of the ODE system.
        y : tf.Tensor
            The dependent variable of the ODE system.

        Returns
        -------
        error : tf.Tensor
            The error between the computed ODE system and the actual ODE system.

        """
        time = x
        signal = y
        
        ds_dt = dde.grad.jacobian(signal, time, i=0)
        
        if self.is_cbf_trainable:
            cbf = SystemDynamics.get_variable(self.cerebral_blood_flow)
        else:
            cbf = self.cerebral_blood_flow
        if self.is_at_trainable:
            at = SystemDynamics.get_variable(self.arrival_time)
        else:
            at = self.arrival_time
        if self.is_t1b_trainable:
            t1b = SystemDynamics.get_variable(self.T1b_ms)
        else:
            t1b = self.T1b_ms
        
        error = ds_dt - self.ode_continuous(time, cbf, at,
                                            self.bolus_duration_ms, t1b,
                                            self.smoothness)
        
        return error
    


class SystemPairDynamics(SystemDynamics):
    
    def __init__(self):
        """
        Initializes the class.

        Returns
        -------
        None

        """
        # Initialise with default parameters
        self.FA_degrees = 35
        self.T1b_ms = 1700        
        self.T1t_ms = 1330      
        self.blood_lambda = 0.98
        self.bolus_duration_ms = 700    
        self.dt_ms = 600
        self.M0b_au = 1000
        self.a = 0.91
        
        self.cerebral_blood_flow = 0
        self.arrival_time = 0
        self.cerebral_blood_flow_pair = {}
        self.arrival_time_pair = {}
        
        self.min_t_ms = 150
        self.max_t_ms = 3450
        
    
    @staticmethod
    def get_cbf_pair_indexes(config: np.ndarray) -> List[int]:
        """
        Get the indexes of the CBF pairs in the given configuration.

        Parameters
        ----------
        config : np.ndarray
            The configuration array.

        Returns
        -------
        indexes : List[int]
            A list of indexes representing the CBF pairs in the configuration.

        """
        base = 8
        cbf = 1
        increment = 2
        indexes = []
        i = base + cbf
        while i < config.shape[1]:
            indexes.append(i)
            i += increment
        return indexes
    
    
    @staticmethod
    def get_at_pair_indexes(config: np.ndarray) -> List[int]:
        """
        Get the indexes of AT pairs in the given configuration.

        Parameters
        ----------
        config : np.ndarray
            The configuration array.

        Returns
        -------
        indexes : List[int]
            A list of indexes representing the AT pairs in the configuration.

        """
        base = 8
        at = 2
        increment = 2
        indexes = []
        i = base + at
        while i < config.shape[1]:
            indexes.append(i)
            i += increment
        return indexes
    
    
    def set_configuration(self, config: np.ndarray, t_min: float, t_max: float, alpha: float,
                          is_cbf_trainable: bool = False, is_at_trainable: bool = False,
                          is_t1b_trainable: bool = False, smoothness: float = 0.1) -> None:
        """
        Set the configuration parameters for the model.

        Parameters
        ----------
        config : np.ndarray
            The configuration array containing the model parameters.
        t_min : float
            The minimum time value.
        t_max : float
            The maximum time value.
        alpha : float
            The alpha value.
        is_cbf_trainable : bool, optional
            Flag indicating whether the cerebral blood flow parameters are trainable. 
            The default is False.
        is_at_trainable : bool, optional
            Flag indicating whether the arrival time parameters are trainable.
            The default is False.
        is_t1b_trainable : bool, optional
            Flag indicating whether the T1b parameters are trainable. The default is False.
        smoothness : float, optional
            The smoothness parameter. The default is 0.1.

        Returns
        -------
        None

        """
        config = tf.cast(config, tf.float32)
        
        cbf_indexes = self.get_cbf_pair_indexes(config)
        at_indexes = self.get_at_pair_indexes(config)
        
        # Get only 1 point as they should be the same (broadcast)
        if is_cbf_trainable:
            self.cerebral_blood_flow = dde.Variable(config[0, 0])
            for i in range(len(cbf_indexes)):
                self.cerebral_blood_flow_pair[i] = dde.Variable(config[0, cbf_indexes[i]])
        else:
            self.cerebral_blood_flow = config[0, 0]
            for i in range(len(cbf_indexes)):
                self.cerebral_blood_flow_pair[i] = config[0, cbf_indexes[i]]
                
        if is_at_trainable:
            self.arrival_time = dde.Variable(config[0, 1])
            for i in range(len(at_indexes)):
                self.arrival_time_pair[i] = dde.Variable(config[0, at_indexes[i]])
        else:
            self.arrival_time = config[0, 1]
            for i in range(len(at_indexes)):
                self.arrival_time_pair[i] = config[0, at_indexes[i]]
            
        self.FA_rads = np.deg2rad(config[0, 2])
        if is_t1b_trainable:   
            self.T1b_ms = dde.Variable(config[0, 3])
        else:
            self.T1b_ms = config[0, 3]
        self.T1t_ms = config[0, 4]
        self.blood_lambda = config[0, 5]
        self.bolus_duration_ms = config[0, 6]
        self.dt_ms = config[0, 7]
        self.M0b_au = config[0, 8]
        
        self.a = alpha
        
        self.min_t_ms = t_min
        self.max_t_ms = t_max
        
        self.smoothness = smoothness
        
        self.is_cbf_trainable = is_cbf_trainable
        self.is_at_trainable = is_at_trainable
        self.is_t1b_trainable = is_t1b_trainable
        
        
    def ode_system_continuous(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Calculate the continuous system of ordinary differential equations (ODEs) for the given input.

        Parameters
        ----------
        x : tf.Tensor
            The independent variable representing time.
        y : tf.Tensor
            The dependent variable representing the state of the system.

        Returns
        -------
        errors : tf.Tensor
            The errors between the calculated derivatives and the ODEs.

        """
        time = x
        
        if self.is_cbf_trainable:
            cbf = SystemDynamics.get_variable(self.cerebral_blood_flow,)
        else:
            cbf = self.cerebral_blood_flow
            
        if self.is_at_trainable:
            at = SystemDynamics.get_variable(self.arrival_time)
        else:
            at = self.arrival_time
        
        if self.is_t1b_trainable:
            t1b = self.get_variable(self.T1b_ms)
        else:
            t1b = self.T1b_ms
        
        errors = []
        ds_dt = dde.grad.jacobian(y, time, i=0)
        error = ds_dt - self.ode_continuous(time, cbf, at,
                                            self.bolus_duration_ms, t1b,
                                            self.smoothness)
        errors.append(error)
        
        for i in range(len(self.cerebral_blood_flow_pair)):
            if self.is_cbf_trainable:
                cbf = SystemDynamics.get_variable(self.cerebral_blood_flow_pair[i])
            else:
                cbf = self.cerebral_blood_flow_pair[i]
                
            if self.is_at_trainable:
                at = SystemDynamics.get_variable(self.arrival_time_pair[i])
            else:
                at = self.arrival_time_pair[i]
                
            if self.is_t1b_trainable:
                t1b = self.get_variable(self.T1b_ms)
            else:
                t1b = self.T1b_ms
                
            ds_dt = dde.grad.jacobian(y, time, i=i+1)
            error = ds_dt - self.ode_continuous(time, cbf, at,
                                                self.bolus_duration_ms, t1b,
                                                self.smoothness)
            errors.append(error)
            
        
        return errors