import random
from typing import Any, Dict, Union, Tuple

import numpy as np
import scipy.io as sio
from scipy.spatial import KDTree

from data.data_processing import ASLFit, ASLFilter, ASLUncertainty


# ArterialSpinLabeling (ASL)
class ASLData():            
    
    def __init__(self, normalize: bool = True, spatial_average: bool = False,
                 spatial_weight: bool = False, pair: int = 0):
        """
        Initialize the ASLData object.

        Parameters
        ----------
        normalize : bool, optional
            Flag indicating whether to normalize the perfusion image. The default is True.
        spatial_average : bool, optional
            Flag indicating whether to perform spatial averaging. The default is False.
        spatial_weight : bool, optional
            Flag indicating whether to apply spatial weighting. The default is False.
        pair : int, optional
            PThe number of point pairs (cases: pair+1) to use for the ASL data. The default is 0.

        Returns
        -------
        None.

        """
        random.seed(0)
        
        self.normalize_perfusion_image = normalize
        self.spatial_average = spatial_average
        self.spatial_weight = spatial_weight
        self.pair = pair
        
        self.perfusion_image = None # (Au)
        self.brain_mask = None
        self.white_matter_mask = None
        self.grey_matter_mask = None
    
        # Model parameters
        self.parameters = {
            'T1b': None,            # t1 blood (msec)
            'T1WM': None,           # t1 white matter (msec)
            'T1GM': None,           # t1 grey matter (msec)
            'lambdawhole': None,    # blood-brain coefficient
            't': None,              # time (msec)
            'FA': None,             # excitation flip angle (degrees)
            'M0b': None,            # fully uncovered signal from blood (Au)
            'dt': None,             # temporal interval between sampled time points (msec)
            'tau': None             # duration of the bolus (msec)
            }
        
        # Target data (outpASL)
        self.output_asl = {
            'CBFmap': None,    # perfusion / cerebral blood flow (mL/100g/min)
            'ATmap': None,     # arrival time (msec)
            }
        
        self.points = None
        
    
    # Function taken from: https://stackoverflow.com/a/29126361
    # This deals with nested structures
    @staticmethod
    def loadmat(filename):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''
        def _check_keys(d):
            '''
            checks if entries in dictionary are mat-objects. If yes
            todict is called to change them to nested dictionaries
            '''
            for key in d:
                if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                    d[key] = _todict(d[key])
            return d

        def _todict(matobj):
            '''
            A recursive function which constructs from matobjects nested dictionaries
            '''
            d = {}
            for strg in matobj._fieldnames:
                elem = matobj.__dict__[strg]
                if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                    d[strg] = _todict(elem)
                elif isinstance(elem, np.ndarray):
                    d[strg] = _tolist(elem)
                else:
                    d[strg] = elem
            return d

        def _tolist(ndarray):
            '''
            A recursive function which constructs lists from cellarrays
            (which are loaded as numpy ndarrays), recursing into the elements
            if they contain matobjects.
            '''
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_tolist(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return elem_list
        
        data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)
    
    
    @staticmethod
    def get_avasl_from_mat(mat_data: Dict[str, Any]) -> Union[None, np.ndarray]:
        """
        Returns the perfusion image ('avasl'), if it exists in the MatLab dictionary.

        Parameters
        ----------
        mat_data : Dict[str, Any]
            Dictionary of the MatLab data.

        Returns
        -------
        perfusion_image : np.ndarray
            The perfusion image, if it exists in the dictionary. Otherwise returns
            None.

        """
        key = 'avasl'
        
        if key not in mat_data:
            return None
        
        perfusion_image = np.squeeze(mat_data[key])
        return perfusion_image
    
    
    @staticmethod
    def get_roi_from_mat(mat_data: Dict[str, Any], roi_type: str) -> Union[None, np.ndarray]:
        """
        Return the region of interest for either the whole brain, gray matter or
        white matter.

        Parameters
        ----------
        mat_data : Dict[str, Any]
            Dictionary of the MatLab data.
        roi_type : str
            The region of interest to return. Available options: 'brain' (whole brain)
            'gm' (gray matter) or 'wm' (white matter).

        Returns
        -------
        roi_mask : np.ndarray
            Returns the specified 2D mask, if it exists. Otherwise returns None.

        """
        if roi_type.lower() == 'brain':
            key = 'roibrain'
        elif roi_type.lower() == 'gm':
            key = 'roiGM'
        elif roi_type.lower() == 'wm':
            key = 'roiWM'
        
        if key not in mat_data:
            return None
        
        roi_mask = np.squeeze(mat_data[key])
        return roi_mask
    
    
    @staticmethod
    def get_pars_from_mat(mat_data: Dict[str, Any]) -> Union[None, Dict[str, Union[np.ndarray, float]]]:
        """
        Returns the global parameters of interest from the MatLab dicitonary.

        Parameters
        ----------
        mat_data : Dict[str, Any]
            Dictionary of the MatLab data.

        Returns
        -------
        parameters : Dictionary
            Returns the values of the following paramters: 'T1b', 'T1WM', 'T1GM',
            'lambdawhole', 't', 'FA', 'M0b', 'dt' and 'tau'.

        """
        key = 'pars'
        # Assumes the following keys are present
        item_list = ['T1b', 'T1WM', 'T1GM', 'lambdawhole',
                     't', 'FA', 'M0b', 'dt', 'tau']
        
        if key not in mat_data:
            return None
        
        parameters_data = mat_data[key]
        
        if len(parameters_data) < len(item_list):
            return None
        
        parameters = {}
        for item in item_list:
            if not item in parameters_data:
                return None
            parameters[item] = parameters_data[item]
        
        return parameters
    
    
    @staticmethod
    def get_output_asl_from_mat(mat_data: Dict[str, Any], key: str = 'outputASL') -> Union[None, Dict[str, Union[np.ndarray, float]]]:
        """
        Returns the cerebral blood flow (CBF) arrival time (AT) maps.

        Parameters
        ----------
        mat_data : Dict[str, Any]
            Dictionary of the MatLab data.
        key : str, optional
            The ASL key in the dictionary. The default is 'outputASL'.

        Returns
        -------
        output_asl : Dictionary
            Dictionary with 'CBFmap' and 'ATmap' values.

        """
        # Assumes the following keys are present
        item_list = ['CBFmap', 'ATmap']
        
        if key not in mat_data:
            return None
        
        output_data = mat_data[key]
        
        if len(output_data) < len(item_list):
            return None
        
        output_asl = {}
        for item in item_list:
            if not item in output_data:
                return None
            output_asl[item] = np.squeeze(np.asarray(output_data[item]))
            
        return output_asl
    
    @staticmethod
    def set_units(units: str, output_asl: Dict[str, Any], parameters: Dict[str, Any]):
        """
        Converts the data to the specified time unit.

        Parameters
        ----------
        units : str
            Time units to convert the data. Supports either 's' (seconds) or
            'ms' (milliseconds).
        output_asl : Dictionary
            Dictionary with the CBF and AT maps.
        parameters : Dictionary
            Dictionary with the global parameters.

        Returns
        -------
        output_asl : Dictionary
            Updated CBF and AT maps.
        parameters : Dictionary
            Updated global parameters.

        """
        if units == 'ms':
            cbf_scale = 60000
            time_scale = 1
        elif units == 's':
            cbf_scale = 60
            time_scale = 1000
            
        output_asl['CBFmap'] = output_asl['CBFmap'] / cbf_scale
        output_asl['ATmap'] = output_asl['ATmap'] / time_scale
        parameters['t'] = [i / time_scale for i in parameters['t']]
        parameters['M0b'] = parameters['M0b'] / cbf_scale
        parameters['T1b'] = parameters['T1b'] / time_scale
        parameters['T1GM'] = parameters['T1GM'] / time_scale
        parameters['T1WM'] = parameters['T1WM'] / time_scale
        parameters['tau'] = parameters['tau'] / time_scale
        parameters['dt'] = parameters['dt'] / time_scale
        
        return output_asl, parameters
        
    
    @staticmethod
    def normalize_perfusion_points(perfusion_points, m0b) -> np.ndarray:
        """
        Normalizes the signal against the fully uncovered signal from blood (M0b).

        Parameters
        ----------
        perfusion_points : np.ndarray
            The perfusion points across time.
        m0b : float
            The full uncovered signal from blood.

        Returns
        -------
        perfusion_points : np.ndarray
            The normalized perfusion points across time.

        """
        a = 0.91
        normalize_by = 2 * m0b * a
        perfusion_points = perfusion_points / normalize_by
                
        return perfusion_points
    
        
    def load_from_mat(self, filename: str, units: str = 'ms') -> None:
        """
        Loads the ASL and global parameters from the MatLab files.

        Parameters
        ----------
        filename : str
            The file path to the MatLab data file.
        units : str, optional
            The time units to convert the data. Supported options are 's' (seconds)
            and 'ms' (milliseconds). The default is 'ms'.

        Returns
        -------
        None.

        """
        # Assumes consistent key structure
        mat_data = ASLData.loadmat(filename)
        
        # Manually assign to variables, as inner varible naming is lost
        self.perfusion_image = ASLData.get_avasl_from_mat(mat_data)
        self.brain_mask = ASLData.get_roi_from_mat(mat_data, roi_type='brain')
        self.white_matter_mask = ASLData.get_roi_from_mat(mat_data, roi_type='wm')
        self.grey_matter_mask = ASLData.get_roi_from_mat(mat_data, roi_type='gm')

        self.parameters = ASLData.get_pars_from_mat(mat_data)
        self.output_asl = ASLData.get_output_asl_from_mat(mat_data)
        if self.output_asl is None:
            self.output_asl = ASLData.get_output_asl_from_mat(mat_data, 'outpASL_GM')
        
        self.output_asl, self.parameters = ASLData.set_units(units, self.output_asl, self.parameters)
        
    
    def construct_tree(self, indices: np.ndarray) -> None:
        """
        Constructs a kd-tree based on the brain mask indices. The function
        'query_tree()' can be used to find the nearest indices.

        Parameters
        ----------
        indices : np.ndarray
            Indices of the brain mask.

        Returns
        -------
        None.

        """
        self.index_point_mapping = np.transpose(indices) # index to point
        self.tree = KDTree(self.index_point_mapping)

    
    def query_tree(self, index: int) -> np.ndarray:
        """
        Finds the nearest indices (radius of 1.5 pixels) to the given index.

        Parameters
        ----------
        index : int
            The index to find its nearest neighbors.

        Returns
        -------
        neighbor_indices : np.ndarray
            All the closest indices to the index.

        """
        point = self.index_point_mapping[index]
        # Define radius
        radius = 1.5
        # Query the tree for points within radius
        neighbor_indices = self.tree.query_ball_point(point, radius)
        return neighbor_indices
        
    
    def spatial_average_points(self, points, output_points, point_index: int,
                               point_neighbor_indices: np.ndarray, point_weight: float = 1.0):
        """
        Calculates the weighted average of a point's neighborhood.

        Parameters
        ----------
        points : TYPE
            The input dictionary that consists if the index (key) and the value.
        output_points : Dictionary
            The output dictionary that consists of the index (key) and average (value).
        point_index : int
            Which point to estimate the average.
        point_neighbor_indices : np.ndarray
            The neighboring indices of the point.
        point_weight : float, optional
            DESCRIPTION. The default is 1.0.

        Returns
        -------
        output_points : Dictionary
            The output dictionary that consists of the index (key) and average (value).

        """
        time_length = points[point_index].shape[0]
        output_points[point_index] = np.copy(points[point_index])
        perfusion_index = 1

        for t in range(time_length):
            values = []
            weights = []
            for i in point_neighbor_indices:
                values.append(points[i][t][perfusion_index])
                if i == point_index:
                    weights.append(point_weight)
                else:
                    weights.append(1.0)
            output_points[point_index][t][perfusion_index] = np.average(values, weights=weights)
        return output_points
    
    
    def append_point(self, points: np.ndarray, point_index: int, point_to_append_index: int):
        """
        Stacks the 'point_to_append_index' points across the column to the 'point_index'
        position.

        Parameters
        ----------
        points : np.ndarray
            The list of temporal points
        point_index : int
            The index to append to.
        point_to_append_index : int
            The index to copy from.

        Returns
        -------
        points : np.ndarray
            The stacked points.

        """
        
        perfusion_index = 1
        cbf_index = 2
        at_index = 3
        
        points[point_index] = np.column_stack((points[point_index],
                                               points[point_to_append_index][:, cbf_index]))
        points[point_index] = np.column_stack((points[point_index],
                                               points[point_to_append_index][:, at_index]))
        points[point_index] = np.column_stack((points[point_index],
                                               points[point_to_append_index][:, perfusion_index]))
        
        return points
            
    
    def get_mask_indices(self, filter_points: bool = True) -> Tuple[np.ndarray]:
        """
        Selects the mask indices and the rrespective cerebral blood float and
        arrival time values.

        Parameters
        ----------
        filter_points : bool, optional
            Whether to remove outlier points. The default is True.

        Returns
        -------
        mask_indices : np.ndarray
            The mask indices.
        cerebral_blood_flow_points : np.ndarray
            The corresponding cerebral blood flow values.
        arrival_time_points : np.ndarray
            The corresponding arrival times.

        """
        mask_indices = self.grey_matter_mask.nonzero()
        
        cerebral_blood_flow_points = self.output_asl['CBFmap'][mask_indices]
        arrival_time_points = self.output_asl['ATmap'][mask_indices]
        
        if filter_points:
            (mask_indices,
              cerebral_blood_flow_points,
              arrival_time_points) = ASLFilter.point_filter(mask_indices,
                                                            cerebral_blood_flow_points,
                                                            arrival_time_points,
                                                            units='s')
                                                        
        return mask_indices, cerebral_blood_flow_points, arrival_time_points
                                                        
    
    def generate_points(self) -> np.ndarray:
        """
        Generates the values of per spatial position across time.

        Returns
        -------
        points : np.ndarray
            Returns all spatial-temporal points.

        """
        (mask_indices,
          cerebral_blood_flow_points,
          arrival_time_points) = self.get_mask_indices()
        
        st_points = {}
        # Per temporal slice
        for t_slice in range(self.perfusion_image.shape[-1]):
            perfusion_slice = self.perfusion_image[..., t_slice]
            perfusion_points = perfusion_slice[mask_indices]    # M(t)
            time_points = np.zeros_like(perfusion_points) + self.parameters['t'][t_slice]
        
            fa_points = np.zeros_like(perfusion_points) + self.parameters['FA']
            
            t1b_points = np.zeros_like(perfusion_points) + self.parameters['T1b']

            grey_matter_points = self.grey_matter_mask[mask_indices] * self.parameters['T1GM']  # Note: CBF/AT is already masked based on GM
            t1t_points = grey_matter_points
            
            blood_lambda_points = np.zeros_like(perfusion_points) + self.parameters['lambdawhole']
            
            bolus_duration_points = np.zeros_like(perfusion_points) + self.parameters['tau']
        
            dt_points = np.zeros_like(perfusion_points) + self.parameters['dt']
            
            m0b_points = np.zeros_like(perfusion_points) + self.parameters['M0b']
        
            if self.normalize_perfusion_image:
                perfusion_points = ASLData.normalize_perfusion_points(perfusion_points, m0b_points)
                m0b_points = np.ones_like(perfusion_points) # Normalized, so set to 1
                
            # Per spatial point
            stacked_points = np.stack((time_points,                 
                                       perfusion_points,            
                                       cerebral_blood_flow_points,  
                                       arrival_time_points,
                                       fa_points,
                                       t1b_points,
                                       t1t_points,
                                       blood_lambda_points,
                                       bolus_duration_points,
                                       dt_points,
                                       m0b_points)).T
            
            for i_point in range(len(mask_indices[0])):
                current_stack = st_points.get(i_point, [])
                current_stack.append(stacked_points[i_point:i_point+1])
                st_points[i_point] = current_stack

        for i_point in range(len(mask_indices[0])):
            st_points[i_point] = np.concatenate(st_points[i_point])
            st_points[i_point] = st_points[i_point]#.astype(np.float64)
            
        self.points = st_points
        
        # Refit to take into account the normalization
        asl_fit = ASLFit(train_cbf=True, train_at=False, train_t1b=False)
        for i_point in range(len(self.points)):
            self.points[i_point] = asl_fit.fit_and_update(self.points[i_point])
            
            
        if self.spatial_weight:
            asl_uncertainty = ASLUncertainty(radius=1.5, indices=mask_indices)
            self.point_weights = []
            for i_point in range(len(self.points)):
                weights = asl_uncertainty.get_point_weights(self.points,
                                                            index=i_point,
                                                            min_val=1, max_val=10)
                weights = asl_uncertainty.scale_distribution(weights, target_sum=100)
                self.point_weights.append(weights)
        
        
        if self.spatial_average:
            averaged_points = {}
            self.construct_tree(mask_indices)
            for i_point in range(len(self.points)):
                neighbor_indices = self.query_tree(i_point)
                averaged_points = self.spatial_average_points(self.points, averaged_points,
                                                              i_point, neighbor_indices,
                                                              point_weight=3.0)
            self.points = averaged_points
        
        
        if self.spatial_weight:
            for i_point in range(len(self.points)):
                self.points[i_point] = asl_uncertainty.repeat_rows(self.points[i_point],
                                                                    self.point_weights[i_point])
        
        
        if self.pair > 0:
            for i_point in range(len(self.points)):
                valid_indices = list(range(len(self.points)))
                valid_indices.remove(i_point)
                
                # Shuffle valid_indices with a seed of i_point
                random.seed(i_point)
                random.shuffle(valid_indices)
                
                # Select the first n pairs for selection
                random_indices = valid_indices[:self.pair]
                
                for new_i in random_indices: #neighbor_index
                    self.points = self.append_point(self.points, i_point, new_i)
        
        
        return self.points
    
