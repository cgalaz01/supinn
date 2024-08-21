import numpy as np

from typing import Tuple, List


class SignalData():
    
    def __init__(self, FA: float = 35, T1b: float = 1700, T1t: float = 1330,
                 blood_lambda: float = 0.98, bolus_duration: float = 0.98,
                 dt: float = 600, M0b: float = 1000, a: float = 0.91,
                 cerebral_blood_flow: float = 0.0004, arrival_time: float = 900,
                 min_t: float = 150, max_t: float = 3450):
        """
        Initialize the Signal class with the given parameters.

        Parameters
        ----------
        FA : float, optional
            Flip angle in degrees. The default is 35.
        T1b : float, optional
            T1 relaxation time of blood in ms. The default is 1700.
        T1t : float, optional
            T1 relaxation time of tissue in ms. The default is 1330.
        blood_lambda : float, optional
            Blood-brain partition coefficient. The default is 0.98.
        bolus_duration : float, optional
            Bolus duration in seconds. The default is 0.98.
        dt : float, optional
            Time step size in ms. The default is 600.
        M0b : float, optional
            Equilibrium magnetization of blood. The default is 1000.
        a : float, optional
            Labeling efficiency. The default is 0.91.
        cerebral_blood_flow : float, optional
            Cerebral blood flow in mL/100g/min. The default is 0.0004.
        arrival_time : List[float], optional
            List of arrival time values in ms. The default is [900].
        min_t : float, optional
            Minimum time in ms. The default is 150.
        max_t : float, optional
            Maximum time in ms. The default is 3450.

        Returns
        -------
        None

        """
        # Initialise with default parameters
        self.FA_rads = np.deg2rad(FA)
        self.T1b_ms = T1b        
        self.T1t_ms = T1t      
        self.blood_lambda = blood_lambda
        self.bolus_duration_ms = bolus_duration
        self.dt_ms = dt
        self.M0b_au = M0b
        self.a = a
        
        self.cerebral_blood_flow = cerebral_blood_flow
        self.arrival_time = arrival_time
        
        self.min_t_ms = min_t
        self.max_t_ms = max_t
    
    
    def generate_time(self, time_range: Tuple[float], samples: int) -> np.ndarray:
        """
        Generate an array of time values within the specified range.

        Parameters
        ----------
        time_range : Tuple[float]
            A tuple representing the range of time values. The first element is the start time, and
            the second element is the end time.
        samples : int
            The number of samples to generate within the time range.

        Returns
        -------
        time : np.ndarray
            An array of time values evenly spaced within the specified range.

        """
        return np.linspace(start=time_range[0], stop=time_range[1], endpoint=True, num=samples)


    def buxton_signal(self, time: np.ndarray) -> np.ndarray:
        """
        Calculate the Buxton signal for a given time array.

        Parameters
        ----------
        time : np.ndarray
            The time array for which to calculate the Buxton signal.

        Returns
        -------
        delta_s : np.ndarray
            The calculated Buxton signal.

        """
        # t < ATT
        cond_1 = time < self.arrival_time
        ds_lower = np.zeros_like(time)
        
        # ATT <= t < ATT + bolus
        ds_range = self.M0b_au * self.a * self.cerebral_blood_flow * (time - self.arrival_time) * np.exp(-time / self.T1b_ms)
        
        # ATT + bolus <= t
        cond_3 = self.arrival_time + self.bolus_duration_ms <= time
        ds_upper = self.M0b_au * self.a * self.cerebral_blood_flow * self.bolus_duration_ms * np.exp(-time / self.T1b_ms)
        
        delta_s = np.where(cond_1, ds_lower, ds_range)
        delta_s = np.where(cond_3, ds_upper, delta_s)
        
        return delta_s
    
    
    @staticmethod
    def shift_tanh(x: np.ndarray, shift: float = 0, smooth: float = 1) -> np.ndarray:
        """
        Apply a shifted and smoothed hyperbolic tangent function to the input array.

        Parameters
        ----------
        x : np.ndarray
            The input array.
        shift : float, optional
            The shift value for the hyperbolic tangent function. The default is 0.
        smooth : float, optional
            The smoothing factor for the hyperbolic tangent function. The default is 1.

        Returns
        -------
        signal : np.ndarray
            The output array after applying the shifted and smoothed hyperbolic tangent function.
        """
        # The 0.5 are related to y scale and shift
        return 0.5 + 0.5 * np.tanh((x - shift) / smooth)

    
    @staticmethod
    def inverse_shift_tanh(x: np.ndarray, shift: float = 0, smooth: float = 1) -> np.ndarray:
        """
        Apply the inverse shift tanh function to the input array.

        Parameters
        ----------
        x : np.ndarray
            The input array.
        shift : float, optional
            The shift value. The default is 0.
        smooth : float, optional
            The smoothness parameter. The default is 1.

        Returns
        -------
        signal : np.ndarray
            The transformed array.

        """
        return 0.5 - 0.5 * np.tanh((x - shift) / smooth)
    
    
    def buxton_signal_continuous(self, time: np.ndarray, smoothness: float = 0.01) -> np.ndarray:
        """
        Calculate the continuous Buxton signal.

        Parameters
        ----------
        time : np.ndarray
            Array of time values.
        smoothness : float, optional
            Smoothing parameter for the signal. The default is 0.01.

        Returns
        -------
        delta_s : np.ndarray
            Array of the calculated Buxton signal values.

        """
        # ATT <= t < ATT + bolus
        ds_range = self.M0b_au * self.a * self.cerebral_blood_flow * (time - self.arrival_time) * np.exp(-time / self.T1b_ms)
        
        # ATT + bolus <= t
        ds_upper = self.M0b_au * self.a * self.cerebral_blood_flow * self.bolus_duration_ms * np.exp(-time / self.T1b_ms)
        
        
        t_range = SignalData.shift_tanh(time, self.arrival_time, smoothness) * \
            SignalData.inverse_shift_tanh(time, self.bolus_duration_ms + self.arrival_time, smoothness)
            
        t_upper = SignalData.shift_tanh(time, self.bolus_duration_ms + self.arrival_time, smoothness)
        
        delta_s = t_range * ds_range + t_upper * ds_upper
        
        return delta_s
     
        
    def add_noise(self, signal: np.ndarray, time: np.ndarray, mean: float, std: float,
                    seed: int = 0) -> np.ndarray:
        """
        Add Gaussian noise to the input signal.

        Parameters
        ----------
        signal : np.ndarray
            The input signal to which noise will be added.
        time : np.ndarray
            The time array corresponding to the signal.
        mean : float
            The mean of the Gaussian noise distribution.
        std : float
            The standard deviation of the Gaussian noise distribution.
        seed : int, optional
            The seed value for the random number generator. Default is 0.

        Returns
        -------
        noisy_signal : np.ndarray
            The signal with added Gaussian noise.

        """
        rng = np.random.RandomState(seed)
            
        noise = rng.normal(mean, std, signal.shape)
        noisy_signal = np.copy(signal)
        noisy_signal += noise
                    
        return noisy_signal
    
    

class SignalPairData(SignalData):
    
    def __init__(self, FA: float = 35, T1b: float = 1700, T1t: float = 1330,
                 blood_lambda: float = 0.98, bolus_duration: float = 0.98,
                 dt: float = 600, M0b: float = 1000, a: float = 0.91,
                 cerebral_blood_flow: float = 0.0004, cerebral_blood_flow_pair: List[float] = [0.0004],
                 arrival_time: List[float] = 900, arrival_time_pair: List[float] = [900],
                 min_t: float = 150, max_t: float = 3450):
        """
        Initialize the Signal class.

        Parameters
        ----------
        FA : float, optional
            Flip angle in degrees. The default is 35.
        T1b : float, optional
            T1 relaxation time of blood in ms. The default is 1700.
        T1t : float, optional
            T1 relaxation time of tissue in ms. The default is 1330.
        blood_lambda : float, optional
            Blood-brain partition coefficient. The default is 0.98.
        bolus_duration : float, optional
            Bolus duration in seconds. The default is 0.98.
        dt : float, optional
            Time step size in ms. The default is 600.
        M0b : float, optional
            Equilibrium magnetization of blood. The default is 1000.
        a : float, optional
            Labeling efficiency. The default is 0.91.
        cerebral_blood_flow : float, optional
            Cerebral blood flow in mL/100g/min. The default is 0.0004.
        cerebral_blood_flow_pair : List[float], optional
            List of cerebral blood flow values for paired ASL. The default is [0.0004].
        arrival_time : List[float], optional
            List of arrival time values in ms. The default is [900].
        arrival_time_pair : List[float], optional
            List of arrival time values for paired ASL in ms. The default is [900].
        min_t : float, optional
            Minimum time in ms. The default is 150.
        max_t : float, optional
            Maximum time in ms. The default is 3450.

        Returns
        -------
        None

        """
        self.signal_data_list = []
        signal_data = SignalData(FA=FA, T1b=T1b, T1t=T1t, blood_lambda=blood_lambda,
                                 bolus_duration=bolus_duration, dt=dt, M0b=M0b, a=a,
                                 cerebral_blood_flow=cerebral_blood_flow,
                                 arrival_time=arrival_time, min_t=min_t, max_t=max_t)
        self.signal_data_list.append(signal_data)

        for i in range(len(cerebral_blood_flow_pair)):
            signal_data = SignalData(FA=FA, T1b=T1b, T1t=T1t, blood_lambda=blood_lambda,
                                     bolus_duration=bolus_duration, dt=dt, M0b=M0b, a=a,
                                     cerebral_blood_flow=cerebral_blood_flow_pair[i],
                                     arrival_time=arrival_time_pair[i], min_t=min_t, max_t=max_t)
            self.signal_data_list.append(signal_data)
        
    
    def buxton_signal(self, time: np.ndarray) -> np.ndarray:
        """
        Calculate the Buxton signal for the given time array.

        Parameters
        ----------
        time : np.ndarray
            The time array for which to calculate the Buxton signal.

        Returns
        -------
        buxton_signal : np.ndarray
            The calculated Buxton signal as a numpy array.

        """
        delta_s_list = []
        for i in range(len(self.signal_data_list)):
            delta_s_list.append(self.signal_data_list[i].buxton_signal(time=time))
        
        return np.hstack(delta_s_list)
    
    
    def buxton_signal_continuous(self, time: np.ndarray, smoothness: float = 0.01) -> np.ndarray:
        """
        Generate a continuous Buxton signal based on the given time and smoothness.

        Parameters
        ----------
        time : np.ndarray
            The time at which the signal is generated.
        smoothness : float, optional
            The smoothness parameter that controls the smoothness of the signal.
            The default is 0.01.

        Returns
        -------
        buxton_signal : np.ndarray
            The generated Buxton signal as a numpy array.

        """
        delta_s_list = []
        for i in range(len(self.signal_data_list)):
            delta_s_list.append(self.signal_data_list[i].buxton_signal_continuous(time=time,
                                                                                  smoothness=smoothness))
        
        return np.hstack(delta_s_list)