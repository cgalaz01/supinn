import math

from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from scipy import stats

from data.signal import SignalData


class ASLFit():

    def __init__(self, train_cbf: bool, train_at: bool, train_t1b: bool):
        """
        Initialize the least squares fit ASL class.

        Parameters
        ----------
        train_cbf : bool
            Flag indicating whether the CBF (Cerebral Blood Flow) parameter is a learnable.
        train_at : bool
            Flag indicating whether the AT (Arterial Transit) parameter is a learnable.
        train_t1b : bool
            Flag indicating whether the T1b (Blood T1) parameter is a learnable.

        Returns
        -------
        None

        """
        self.train_cbf = train_cbf
        self.train_at = train_at
        self.train_t1b = train_t1b
        # Placeholder for unused parameters
        self.fa = None
        self.t1t = None
        self.blood_lambda = None
        self.dt = None
        self.min_t = None
        self.max_t = None
        

    def set_config(self, M0b: float, a: float, T1b: float, bolus_duration: float,
                   cbf: float, at: float) -> None:
        """
        Set the non-trainable parameters.

        Parameters
        ----------
        M0b : float
            The equilibrium magnetization of blood.
        a : float
            The labeling efficiency.
        T1b : float
            The longitudinal relaxation time of blood.
        bolus_duration : float
            The duration of the labeling bolus.
        cbf : float
            The cerebral blood flow.
        at : float
            The arterial transit time.

        Returns
        -------
        None

        """
        self.M0b_au = M0b
        self.a = a
        self.t1b = T1b
        self.bolus_duration_ms = bolus_duration
        self.cbf = cbf
        self.at = at

    
    def buxton_signal(self, time: np.ndarray, cerebral_blood_flow: float, arrival_time: float,
                      t1b: float) -> np.ndarray:
        """
        Calculate the Buxton signal.

        Parameters
        ----------
        time : np.ndarray
            Array of time points.
        cerebral_blood_flow : float
            Cerebral blood flow value.
        arrival_time : float
            Arrival time of the bolus.
        t1b : float
            T1 relaxation time of blood.

        Returns
        -------
        delta_s : np.ndarray
            The calculated Buxton signal.

        """
        signal = SignalData(FA=self.fa, T1b=t1b, T1t=self.t1t, blood_lambda=self.blood_lambda,
                            bolus_duration=self.bolus_duration_ms, dt=self.dt, M0b=self.M0b_au,
                            a=self.a, cerebral_blood_flow=cerebral_blood_flow, arrival_time=arrival_time,
                            min_t=self.min_t, max_t=self.max_t)
        
        delta_s = signal.buxton_signal_continuous(time, qp_to_1=True, smoothness=0.01)
        
        return delta_s
    
    
    def estimate_parameters(self, time: np.ndarray, signal: np.ndarray, cbf: float,
                            at: float, t1b: float) -> Tuple[float]:
        """
        Estimate the parameters for the ASL model based on the given time and signal data.

        Parameters
        ----------
        time : np.ndarray
            The time values of the ASL signal.
        signal : np.ndarray
            The ASL signal values.
        cbf : float
            The initial value for cerebral blood flow (CBF) parameter.
        at : float
            The initial value for arterial transit time (AT) parameter.
        t1b : float
            The initial value for T1 of blood (T1b) parameter.

        Returns
        -------
        updated_cbf : float
            The updated value for the cerebral blood flow (CBF) parameter.
        updated_at : float
            The updated value for the arterial transit time (AT) parameter.
        updated_t1b : float
            The updated value for the T1 of blood (T1b) parameter.

        """
        parameters = {'maxfev': 50000,
                      'bounds': (1e-2, 2.5),
                      'method': 'dogbox',
                      'loss': 'huber',
                      'gtol': 1e-6,
                      'xtol': 1e-5,
                      'ftol': 1e-5}
                      
        p0 = []
        bounds_func = []
        if self.train_cbf:
            p0.append(cbf)
            bounds_func.append(lambda x: max(1e-2, x))
        if self.train_at:
            p0.append(at)
            bounds_func.append(lambda x: max(1e-2, x))
        if self.train_t1b:
            p0.append(t1b)
            bounds_func.append(lambda x: x)

        if p0:
            try:
                popt, _ = curve_fit(self.buxton_signal, time, signal, p0=p0, **parameters)
                popt = [func(val) for func, val in zip(bounds_func, popt)]
            except:
                popt = p0
        else:
            popt = []

        # Map the optimized parameters back to their correct variables
        idx = 0
        updated_cbf = popt[idx] if self.train_cbf else cbf
        idx += 1 if self.train_cbf else 0
        updated_at = popt[idx] if self.train_at else at
        idx += 1 if self.train_at else 0
        updated_t1b = popt[idx] if self.train_t1b else t1b
        
        return updated_cbf, updated_at, updated_t1b
    
    
    def fit_and_update(self, point: np.ndarray) -> np.ndarray:
        """
        Fit the model parameters using the given data point and updates the point with the
        estimated parameters.

        Parameters
        ----------
        point : np.ndarray
            The data point to be processed.

        Returns
        -------
        point : np.ndarray
            The updated data point with the estimated parameters.

        """
        self.set_config(M0b=point[0, 10], a=1.0, T1b=point[0, 5],
                        bolus_duration=point[0, 8], cbf=point[0, 2], at=point[0, 3])

        cbf, at, t1b = self.estimate_parameters(point[:, 0], point[:, 1], point[0, 2],
                                                point[0, 3], point[0, 5])
        point[:, 2] = cbf
        point[:, 3] = at
        point[:, 5] = t1b

        return point
        
    
    
class ASLFilter():
    
    
    @staticmethod
    def point_filter(mask_points: np.ndarray, cbf_points: np.ndarray, at_points: np.ndarray,
                     units: str) -> Tuple[np.ndarray]:
        """
        Filter the points based on a set threshold.

        Parameters
        ----------
        mask_points : np.ndarray
            Array of mask points.
        cbf_points : np.ndarray
            Array of CBF (Cerebral Blood Flow) points.
        at_points : np.ndarray
            Array of AT (Arterial Transit) points.
        units : str
            Units of the points.

        Returns
        -------
        mask_points : np.ndarray
            Filtered array of mask points.
        cbf_points : np.ndarray
            Filtered array of CBF points.
        at_points : np.ndarray
            Filtered array of AT points.
        """
        if units == 's':
            cbf_threshold = 2.0
            at_threshold = 2.5
            
        to_delete = []
        for i in range(len(mask_points[0])):
            if cbf_points[i] > cbf_threshold or at_points[i] > at_threshold:
                to_delete.append(i)
            elif cbf_points[i] < 1e-2 or at_points[i] < 1e-2:
                to_delete.append(i)
                
        cbf_points = np.delete(cbf_points, to_delete)
        at_points = np.delete(at_points, to_delete)
        mask_points = (np.delete(mask_points[0], to_delete),
                        np.delete(mask_points[1], to_delete))
        
        return mask_points, cbf_points, at_points


class ASLUncertainty():
    
    
    def __init__(self, radius: float, indices: np.ndarray):
        """
        Initialize the ASL weight uncertainty class.

        Parameters
        ----------
        radius : float
            The radius value for data processing.
        indices : np.ndarray
            The array of indices for data processing.

        Returns
        -------
        None

        """
        self.radius = radius
        self.indices = indices
        self.construct_tree(indices)
    
    
    def construct_tree(self, indices: np.ndarray) -> None:
        """
        Constructs a KDTree from the given indices.

        Parameters
        ----------
        indices : np.ndarray
            An array of indices.

        Returns
        -------
        None

        """
        self.index_point_mapping = np.transpose(indices)  # index to point
        self.tree = KDTree(self.index_point_mapping)

    
    def query_tree(self, index: int):
        """
        Query the tree for neighbor indices within a given radius.

        Parameters
        ----------
        index : int
            The index of the point to query.

        Returns
        -------
        neighbor_indices : list
            A list of neighbor indices within the specified radius.

        """
        point = self.index_point_mapping[index]
        # Query the tree for points within radius
        neighbor_indices = self.tree.query_ball_point(point, self.radius, return_sorted=True)
        return neighbor_indices
    
    
    def get_point_weights(self, points: np.ndarray, index: int, min_val: float = 0.0,
                          max_val: float = 1.0):
        """
        Calculate the weights for each point based on the error of the signal.

        Parameters
        ----------
        points : np.ndarray
            The array of points.
        index : int
            The index of the point for which to calculate the weights.
        min_val : float, optional
            The minimum value for the weights. The default is 0.0.
        max_val : float, optional
            The maximum value for the weights. The default is 1.0.

        Returns
        -------
        weights : np.ndarray
            The array of weights for each point.

        """
        def normalize(values, min_val, max_val):
            # Find the smallest value in the list
            min_value = min(values)
            # Find the largest value in the list
            max_value = max(values)
            
            # Normalize each value to the range [min_val, max_val]
            normalized_values = [(max_val - min_val) * (x - min_value) / (max_value - min_value) + min_val for x in values]
            
            return normalized_values

        neighbor_indices = self.query_tree(index)
        perfusion_index = 1
        signal = []
        for i in neighbor_indices:
            signal.append(points[i][:, perfusion_index])
            
        signal = np.asarray(signal)
        error = []
        for time in range(signal.shape[1]):
            spatial_signal = signal[:, time]
            if time-1 >= 0:
                spatial_signal = np.append(spatial_signal, signal[:, time-1])
            if time+1 < signal.shape[1]:
                spatial_signal = np.append(spatial_signal, signal[:, time+1])
            spatial_signal = (spatial_signal,)
            bootstrap_results = stats.bootstrap(spatial_signal, np.std,
                                                n_resamples=100, axis=0,
                                                random_state=0)
            error.append(bootstrap_results.standard_error)
            
        error = np.asarray(error)
        weights = np.asarray(normalize(error, min_val, max_val))
        # Invert so that lowest value corresponds to max_val
        weights = np.rint(np.abs(weights - max_val) + min_val).astype(int)
        return weights


    @staticmethod
    def scale_distribution(weights: np.ndarray, target_sum: int):
        """
        Scales the distribution of weights to achieve a target sum.

        Parameters
        ----------
        weights : np.ndarray
            An array of weights representing the distribution.
        target_sum : int
            The desired sum of the scaled distribution.

        Returns
        -------
        scaled_list : list
            A list of scaled values representing the scaled distribution.

        """
        current_sum = sum(weights)
        scale_factor = target_sum / current_sum
        scaled_list = [int(math.ceil(value * scale_factor)) for value in weights]
        
        diff_sum = target_sum - sum(scaled_list)
        
        # Largest to smallest
        smallest_to_largest = np.array(weights).argsort().tolist()
        largest_to_smallest = smallest_to_largest[::-1]
        
        i = 0
        cond = False
        removed = False
        while diff_sum != 0:
            if diff_sum < 0:
                # Remove from smallest if not 1 - If we looped
                if scaled_list[smallest_to_largest[i]] > 1 or cond:
                    removed = True
                    scaled_list[smallest_to_largest[i]] -= 1
                    diff_sum += 1
            else:
                # Add to largest
                scaled_list[largest_to_smallest[i]] += 1
                diff_sum -= 1
            
            i = (i+1) % len(weights)
            # If we looped over the data and we didin't remove any values,
            # then relax the min(1) constrain
            if i == 0:
                if not removed:
                    cond = True
                removed = False
                
        return scaled_list
    
    
    @staticmethod
    def repeat_rows(point: np.ndarray, weights: np.ndarray):
        """
        Repeats each row in the `point` array according to the corresponding value in the `weights` array.

        Parameters
        ----------
        point : np.ndarray
            The input array of shape (n, m), where n is the number of rows and m is the number of columns.
        weights : np.ndarray
            The array of shape (n,) containing the number of times each row in `point` should be repeated.

        Returns
        -------
        np.ndarray
            The resulting array with repeated rows.

        """
        indexes = []
        for i in range(len(weights)):
            indexes.extend([i] * weights[i])
            
        return point[indexes]
