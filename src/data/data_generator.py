import os

import random
from enum import Enum

from typing import Dict, Generator, Tuple, List, Optional, Union
from pathlib import Path

import numpy as np

from data.data_structure import ASLData


class DatasetType(Enum):
    train = 'train'
    validation = 'validation'
    test = 'test'
        
        
        
class DataLoader():
    
    def __init__(self, data_path: Optional[Union[str, Path]], normalize: bool = True,
                 spatial_average: bool = False, pair: int = 0,
                 spatial_weight: bool = False, units: str = 'ms',
                 memory_cache: bool = True) -> None:
        """
        Initialize the DataGenerator object.

        Parameters
        ----------
        data_path : Optional[Union[str, Path]]
            The path to the data directory. If not provided, the default raw data path will be used.
        normalize : bool, optional
            Flag indicating whether to normalize the data. Default is True.
        spatial_average : bool, optional
            Flag indicating whether to perform spatial averaging. Default is False.
        pair : int, optional
            The number of point pairs (pair+1) to use for the data. Default is 0.
        spatial_weight : bool, optional
            Flag indicating whether to apply spatial weighting. Default is False.
        units : str, optional
            The units of the data. Default is 'ms'.
        memory_cache : bool, optional
            Flag indicating whether to use memory caching. Default is True.

        Returns
        -------
        None

        """
        if data_path:                                          
            self.data_directory = data_path
        else:
            self.data_directory = DataLoader.get_default_raw_data_path()
            
        self.data_list = DataLoader.get_mat_file_paths(self.data_directory)
        
        self.units = units
        self.normalize = normalize
        self.spatial_average = spatial_average
        self.spatial_weight = spatial_weight
        self.pair = pair
        
        self.memory_cache = memory_cache
        self.data_in_memory = {}

        self.list_shuffle = random.Random(0)
        
    
    @staticmethod
    def get_default_raw_data_path() -> str:
        """
        Get the default raw (MatLab) data path.

        Returns
        -------
        data_path : str
            The default raw data path.

        """
        raw_data_path = os.path.join('..', '..', 'data_mat')
        
        return raw_data_path

    
    @staticmethod
    def get_mat_file_paths(folder_path: Union[str, Path]) -> List[str]:
        """
        Get the paths of all .mat files in the specified folder.

        Parameters
        ----------
        folder_path : Union[str, Path]
            The path to the folder containing the .mat files.

        Returns
        -------
        List[str]
            A list of file paths for all .mat files in the folder.

        """
        mat_files = sorted(os.listdir(folder_path))
        mat_files = [os.path.join(folder_path, file) for file in mat_files if file.lower().endswith('.mat')]    
        
        return mat_files
    
    
    @staticmethod
    def load_mat_as_structure(mat_file_path: str, units: str, normalize: bool,
                              spatial_average: bool, spatial_weight: bool,
                              pair: int) -> ASLData:
        """
        Load ASL data from a .mat file and return it as an ASLData object.

        Parameters
        ----------
        mat_file_path : str
            The file path to the .mat file containing the ASL data.
        units : str
            The units of the ASL data.
        normalize : bool
            Flag indicating whether to normalize the ASL data.
        spatial_average : bool
            Flag indicating whether to perform spatial averaging on the ASL data.
        spatial_weight : bool
            Flag indicating whether to apply spatial weighting to the ASL data.
        pair : int
            The number of point pairs (cases: pair+1) to use for the ASL data.

        Returns
        -------
        asl_data : ASLData
            An ASLData object containing the loaded ASL data.

        """
        asl_data = ASLData(normalize=normalize, spatial_average=spatial_average,
                            spatial_weight=spatial_weight, pair=pair)
        asl_data.load_from_mat(mat_file_path, units)
        
        return asl_data
    
    
    def is_in_memory(self, patient_directory: Union[str, Path]) -> bool:
        """
        Check if the patient directory is already in memory.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The directory of the patient.

        Returns
        -------
        in_memory : bool
            True if the patient directory is in memory, False otherwise.
        """
        if patient_directory in self.data_in_memory:
            return True

        return False
    
    
    def save_memory(self, patient_directory: Union[str, Path],
                    patient_data: Dict[str, ASLData]) -> None:
        """
        Save the patient data to memory cache.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The directory where the patient data is stored.
        patient_data : Dict[str, ASLData]
            A dictionary containing the patient data.

        Returns
        -------
        None

        """
        if self.memory_cache:
            self.data_in_memory[patient_directory] = patient_data
            
            
    def get_memory(self, patient_directory: Union[str, Path]) -> Dict[str, ASLData]:
        """
        Retrieve the cached ASL data for a given patient directory.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The directory path of the patient.

        Returns
        -------
        patient_data : Dict[str, ASLData]
            A dictionary containing the ASL data for the patient.

        """
        patient_data = self.data_in_memory[patient_directory]
        return patient_data
    
    
    @staticmethod 
    def preprocess_data(data: ASLData) -> ASLData:   
        """
        Preprocesses and genrates points from the given ASLData object.

        Parameters
        ----------
        data : ASLData
            The ASLData object to be preprocessed.

        Returns
        -------
        data : ASLData
            The preprocessed ASLData object.

        """
        _ = data.generate_points()
        return data
    
    
    def generator(self, patient_directory: Union[str, Path]) -> ASLData:
        """
        Retrieve ASL data from a given patient directory.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The directory path where the patient data is stored.

        Returns
        -------
        patient_data : ASLData
            The retrieved ASL data.

        """
        if self.is_in_memory(patient_directory):
            patient_data = self.get_memory(patient_directory)
        else:
            patient_data = DataLoader.load_mat_as_structure(patient_directory,
                                                            units=self.units,
                                                            normalize=self.normalize,
                                                            spatial_average=self.spatial_average,
                                                            spatial_weight=self.spatial_weight,
                                                            pair=self.pair)
            patient_data = DataLoader.preprocess_data(patient_data)
            self.save_memory(patient_directory, patient_data)
        
        return patient_data


    @staticmethod
    def clear_data(data: Tuple[Dict[str, np.ndarray]]) -> None:
        """
        Clear the data by deleting all arrays in the given data.

        Parameters
        ----------
        data : Tuple[Dict[str, np.ndarray]]
            The data to be cleared.

        Returns
        -------
        None

        """
        for i in range(len(data)):
            for key, array in data[i].items():
                del array
                
        del data
        

    def data_generator(self, patient_directory: Union[Path, str], point_index: int,
                       verbose: int = 0) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Generate data for a given patient directory and point index.

        Parameters
        ----------
        patient_directory : Union[Path, str]
            The directory path or string representing the patient directory.
        point_index : int
            The index of the data point within the patient's data.
        verbose : int, optional
            Verbosity level. Set to 0 for no output, higher values for more output. Default is 0.

        Yields
        ------
        Dict[str, np.ndarray]
            A dictionary containing the generated data.

        """
        if verbose > 0:
            print('Generating patient: ', patient_directory)
        patient_data = self.generator(patient_directory)

        points = patient_data.points[point_index]
        for i in range(points.shape[0]):
            yield points[i]
    
    
    def data_generator_index(self, index: int, point_index: int, dataset: Union[DatasetType, int],
                             verbose: int = 0) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Generate data for a specific index and point index in the dataset.

        Parameters
        ----------
        index : int
            The index of the patient in the dataset.
        point_index : int
            The index of the data point within the patient's data.
        dataset : Union[DatasetType, int]
            The type of dataset or the dataset ID.
        verbose : int, optional
            Verbosity level. Set to 0 for no output, higher values for more output. Default is 0.

        Yields
        ------
        Dict[str, np.ndarray]
            A dictionary containing the generated data.

        """
        patient_directory = self.data_list[index]
        
        yield from self.data_generator(patient_directory, point_index, verbose=verbose)
            
        
    def prepare_generator_index(self, index: int, verbose: int = 0) -> int:
        """
        Prepare the data generator for a given index.

        Parameters
        ----------
        index : int
            The index of the patient in the dataset.
        verbose : int, optional
            Verbosity level. Set to 0 for no output, higher values for more output. Default is 0.

        Returns
        -------
        points_size : int
            The number of points in the patient data.

        """
        patient_directory = self.data_list[index]
        if verbose > 0:
            print('Generating patient: ', patient_directory)    
        patient_data = self.generator(patient_directory)
        return len(patient_data.points)
    
    
    def train_generator_index(self, index: int, point_index: int,
                              verbose: int = 0) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Generates batches of training data for a given index and point index.

        Parameters
        ----------
        index : int
            The index of the patient in the dataset.
        point_index : int
            The index of the data point within the patient's data.
        verbose : int, optional
            Verbosity level. Set to 0 for no output, higher values for more output. Default is 0.

        Yields
        ------
        Dict[str, np.ndarray]
            A dictionary containing the generated data.

        """
        yield from self.data_generator_index(index, point_index, DatasetType.train, verbose=verbose)
    