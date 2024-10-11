import os
import pandas as pd
import numpy as np
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from tifffile import imread
from src.utils import get_behaviour_timestamps, smooth_and_shrink_label
from typing import List, Dict, Tuple
from skimage.measure import regionprops_table

class Session:
    def __init__(self, data_row):
        """
        Initialize the Session object with data from a pandas DataFrame row.
        
        :param data_row: A pandas Series representing a row from a DataFrame
        """

        # Initialize attributes
        self.row = data_row
        self.cohort = data_row['Cohort']
        self.session = data_row['Session']
        self.animal_id = data_row['Mouse']
        self.sex = data_row['Sex']
        self.session_length = data_row['Ca Imaging Length(s)']
        self.latency = data_row['LatencyMinConverted']
        self.session_outcome = data_row['Session Outcome']
        self.final_outcome = data_row['Final Outcome']
        self.calcium_fps = data_row['FPS (CI downsampled)']
        self.victim_lib_timecode = data_row['TimeVictimHeadCrossing(s)']
        self.day_of_helping = data_row['DayofHelping']
        self.model_path = data_row['CNMFE Model Path'].replace('.hdf5', '_accepted_only.hdf5')
        self.path_to_pnr = data_row['Peak-to-Noise Ratio Path']
        self.number_of_components_total = data_row['Total Components']
        self.number_of_accepted_components = data_row['Num Accepted']

        # Check if column Num of Liberation cells exist and and as an attribute (int)
        if 'Num of Liberation cells' in data_row.keys():
            self.number_of_liberation_cells = data_row['Num of Liberation cells']
        else:
            self.number_of_liberation_cells = None
        
        # Check if column Num of Random cells exist and and as an attribute (int)
        if 'Num of Random cells' in data_row.keys():
            self.number_of_random_cells = data_row['Num of Random cells']
        else:
            self.number_of_random_cells = None

        # Construct the annotations path
        annotations_filename = data_row['ManualAnnotationSide']
        self.annotations_path = os.path.join('annotations', self.cohort, self.session, f'{annotations_filename}.csv')
        
        # Construct the DFF path
        self.dff_path = self.model_path.replace('_accepted_only.hdf5', '_Dff_z_scored.npy').replace('cnmfe_model', 'npy_data')
        
        # Construct the accepted components path
        output_dir = os.path.dirname(self.dff_path)
        # self.accepted_components_path = os.path.join(output_dir, 'accepted_components_{}.npy'.format(os.path.basename(self.model_path).replace('.hdf5', '')))
        
        # Initialize other attributes
        self.pnr_image = None
        self.caiman_model = None
        self.liberation_cells_ids = None
        self.random_cells_ids = None
        self.snr_values = None 
        self.pnr_image_dims = None  # To store dimensions of the PNR image
        self.neuron_labels = None
        self.neuron_positions = None
        self.neuron_circularity = None
        
    def load_caiman_model(self):
            """Load the CaImAn fitted model."""
            if self.caiman_model is None:
                if not os.path.exists(self.model_path):
                    print(f"Warning: CaImAn model file not found at {self.model_path}")
                    return None
                try:
                    self.caiman_model = load_CNMF(self.model_path)
                    self.load_snr_values()  # Load SNR values after loading the model
                    return self.caiman_model
                except Exception as e:
                    print(f"Error loading CaImAn model: {str(e)}")
                    return None
            return self.caiman_model
    
    def load_snr_values(self):
        """Load SNR values from the CaImAn model."""
        if self.caiman_model is not None:
            try:
                self.snr_values = self.caiman_model.estimates.SNR_comp
                return self.snr_values
            except AttributeError:
                print("Warning: SNR_comp not found in the CaImAn model estimates.")
            except Exception as e:
                print(f"Error loading SNR values: {str(e)}")
        else:
            print("Warning: CaImAn model not loaded. Call load_caiman_model() first.")
        return None

    def load_pnr_image(self):
        """Load the peak-to-noise ratio image."""
        if not os.path.exists(self.path_to_pnr):
            print(f"Warning: PNR image file not found at {self.path_to_pnr}")
            return None
        try:
            self.pnr_image = imread(self.path_to_pnr)
            self.pnr_image_dims = self.pnr_image.shape
            return self.pnr_image
        except Exception as e:
            print(f"Error loading PNR image: {str(e)}")
            return None

    def load_dff_signal(self):
        """Load the DFF calcium signal, pad it with an index column, and assert the number of components matches."""
        if not os.path.exists(self.dff_path):
            print(f"Warning: DFF signal file not found at {self.dff_path}")
            return None
        try:
            dff_signal = np.load(self.dff_path)
            
            # Assert that the number of components matches the expected total
            assert dff_signal.shape[0] == self.number_of_accepted_components, (
                f"Mismatch in number of components: expected {self.number_of_accepted_components}, "
                f"but got {dff_signal.shape[0]} in DFF signal."
            )
            
            # # Create an index array to pad the dff_signal with
            # indices = np.arange(0, self.number_of_accepted_components).reshape(-1, 1)
            
            # # Concatenate the index array with the DFF signal array
            # dff_signal_padded = np.hstack((indices, dff_signal))
            
            return dff_signal
        except Exception as e:
            print(f"Error loading DFF signal: {str(e)}")
            return None

    def load_registration_files(self) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        """
        Load the registration files.

        Returns:
            registration_file (np.ndarray): The registration file.
            indexing_scheme (Dict[str, List[int]]): The indexing scheme.
        """
        # A small workaround to ensure proper file reading
        self.animal_id = str(self.animal_id).zfill(4)

        try:
            registration_file = np.load(os.path.join(f'processed-data/{self.cohort}', 'registration', f'{self.animal_id}_accepted_only_registered_ids.npy'))
            indexing_scheme = np.load(os.path.join(f'processed-data/{self.cohort}', 'registration', f'{self.animal_id}_accepted_only_session_indexing.npy'), allow_pickle=True).item()
            return registration_file, indexing_scheme
        except Exception as e:
            print(f"Error loading registration file: {str(e)}")
            return None, None
        

    # def filter_dff_signal(self, dff_signal_padded: np.ndarray, target_list: List[int]) -> np.ndarray:
    #     """
    #     Filter the DFF signal to include only the rows corresponding to the provided target_list.
        
    #     :param dff_signal_padded: The DFF signal matrix padded with an index column
    #     :param target_list: List of component indices to retain (either accepted components or liberation components)
    #     :return: Filtered DFF signal matrix containing only the needed components
    #     """
    #     # Get the index column from the padded DFF signal
    #     indices = dff_signal_padded[:, 0].astype(int)
        
    #     # Create a mask to filter rows where the index is in the target_list
    #     mask = np.isin(indices, target_list)
        
    #     # Apply the mask to filter the DFF signal
    #     filtered_dff_signal = dff_signal_padded[mask, :]
        
    #     # Remove the index column after filtering (if you want just the DFF data)
    #     # filtered_dff_signal = filtered_dff_signal[:, 1:]
        
    #     return filtered_dff_signal


    def load_liberation_metadata(self, event_type: str):
        """
        Load and filter liberation metadata for the current session, and assert the number of liberation cells matches.
        
        :param metadata_path: Path to the liberation metadata CSV file
        :return: Filtered pandas DataFrame containing liberation metadata for this session,
                or None if the file is not found or data for this session is not present
        """
        
        metadata_path = self.row[f'{event_type} Associated Cells Metadata Path']

        if metadata_path is None or not os.path.exists(metadata_path):
            print(f"Warning: No {event_type} metadata found for Cohort: {self.cohort}, Mouse: {self.animal_id}, Session: {self.session}")
            return None

        full_metadata = pd.read_csv(metadata_path)
        filtered_metadata = full_metadata[
            (full_metadata['Cohort'] == self.cohort) &
            (full_metadata['Mouse'] == self.animal_id) &
            (full_metadata['Session'] == self.session)
        ]

        if not filtered_metadata.empty:

            if event_type == 'Liberation':
                self.liberation_cells_ids = filtered_metadata['Component ID'].to_list()
                assert self.number_of_liberation_cells == len(self.liberation_cells_ids), (
                    f"Wrong number of liberation cells found in the metadata file: expected {self.number_of_liberation_cells}, "
                    f"but got {len(self.liberation_cells_ids)}."
                )

            elif event_type == 'Random':
                self.random_cells_ids = filtered_metadata['Component ID'].to_list()
                assert self.number_of_random_cells == len(self.random_cells_ids), (
                    f"Wrong number of random cells found in the metadata file: expected {self.number_of_random_cells}, "
                    f"but got {len(self.random_cells_ids)}."
                )

            else:
                raise ValueError(f"Invalid event type: {event_type}")

        else:
            print(f"Warning: No {event_type} metadata found for Cohort: {self.cohort}, Mouse: {self.animal_id}, Session: {self.session}")
            return None
        
        return filtered_metadata
    
    def get_neuron_labels(self, binary_output=False, smooth_sigma=3, shrink_pixels=1):
        #TODO: at this point if labels overlap, nonexpected behaviour can occur. I should handle this case. 
        """
        Generate labels for all neurons in the session.

        :param binary_output: If True, return a binary mask. If False, return a labeled image.
        :param smooth_sigma: Sigma for Gaussian smoothing.
        :param shrink_pixels: Number of pixels to shrink the labels by erosion.
        :return: Labeled image or binary mask with neuron labels.
        """
        # Load CaImAn model and PNR image
        if self.caiman_model is None:
            self.load_caiman_model()
        if self.pnr_image is None:
            self.load_pnr_image()
        
        pnr_image = self.pnr_image
        img_shape = pnr_image.shape

        if self.caiman_model is None or pnr_image is None:
            print("Model or PNR image not available. Cannot generate neuron labels.")
            return None
        
        # Initialize the label matrix efficiently
        A_labels = np.zeros(img_shape, dtype=int)

        # Precompute reshaped components outside the loop
        A_reshaped = self.caiman_model.estimates.A.toarray().reshape((*img_shape, -1), order='F')

        for i, comp in enumerate(range(A_reshaped.shape[-1])):
            label = (A_reshaped[..., comp] > 0).astype(int)
            if smooth_sigma > 0 or shrink_pixels > 0:
                label = smooth_and_shrink_label(label, smooth_sigma=smooth_sigma, shrink_pixels=shrink_pixels)
            A_labels = np.maximum(A_labels, label * (i + 1))

        # Assert that the number of labels matches the expected total
        assert np.max(A_labels) == self.number_of_accepted_components, (
            f"Mismatch in number of labels: expected {self.number_of_accepted_components}, "
            f"but got {np.max(A_labels)}."
        )

        # Store the labels and positions
        self.neuron_labels = (A_labels > 0).astype(int) if binary_output else A_labels
        self.neuron_positions = self.extract_neuron_positions(self.neuron_labels)

        return self.neuron_labels

    def extract_neuron_positions(self, labels: np.ndarray) -> np.ndarray:
        """
        Extract the centroid positions of neurons from the labeled image.

        Parameters
        ----------
        labels : np.ndarray
            Labeled image or binary mask.

        Returns
        -------
        np.ndarray
            Array of centroid positions (row, col) for each neuron.

        Notes
        -----
        This function iterates over the labels in the input image, and for each
        label, it computes the centroid position of the corresponding neuron by
        taking the mean of the row and column indices where the label is present.

        If a label is not present in the image, the centroid position is set to
        NaN.

        The function returns an array of shape `(number_of_accepted_components, 2)`
        where each row is a centroid position (row, col) for a neuron.
        """
        positions = []
        # Check that np.max(labels) matches self.number_of_accepted_components
        assert np.max(labels) == self.number_of_accepted_components, (
            f"Mismatch in number of labels: expected {self.number_of_accepted_components}, "
            f"but got {np.max(labels)}."
        )

        # Iterate over the labels
        for label in range(1, self.number_of_accepted_components + 1):
            mask = labels == label
            if np.any(mask):
                rows, cols = np.where(mask)
                centroid_row = np.mean(rows).astype(int)
                centroid_col = np.mean(cols).astype(int)
            else:
                centroid_row = np.nan
                centroid_col = np.nan
            positions.append((centroid_row, centroid_col))
        return np.array(positions, dtype=float)

    def load_annotations(self):
        """Load annotations."""
        if not os.path.exists(self.annotations_path):
            self.annotations_path = self.annotations_path.replace('_regrouped.csv', '_manual_annotations.csv')
            if not os.path.exists(self.annotations_path):
                print(f"Warning: Annotations file not found at {self.annotations_path}")
                return None
        try:
            annotations_table = pd.read_csv(self.annotations_path)
            unique_behaviours = annotations_table['Behavior'].unique()
            all_behaviour_timestamps = {behaviour: get_behaviour_timestamps(annotations_table, behaviour) for behaviour in unique_behaviours}
            return all_behaviour_timestamps
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            return None

    def get_event_associated_cells_ids(self, event_type):
        """
        Get the liberation cell IDs for this session.
        
        :return: List of liberation cell IDs or None if not loaded yet
        """
        if event_type == 'Liberation':
            return self.liberation_cells_ids if self.liberation_cells_ids is not None else None
        elif event_type == 'Random':
            return self.random_cells_ids if self.random_cells_ids is not None else None
    
    def get_snr_values(self):
        """
        Get the SNR values for this session.
        
        :return: Array of SNR values or None if not loaded yet
        """
        return self.snr_values
    
    def measure_neuron_circularity(self):
        #TODO: fix. doesn't work as of right now. return array smaller then original size
        """
        Measure the circularity of each neuron in the neuron labels.
        """
        if self.neuron_labels is None:
            raise ValueError("Neuron labels not loaded. Call load_neuron_labels() first.")
        
        print(np.max(self.neuron_labels))
        # Measure the circularity of each neuron
        regionprops_table_dict = regionprops_table(self.neuron_labels, properties=['eccentricity'])
        df = pd.DataFrame(regionprops_table_dict)
        print(df.head())
        circularity = regionprops_table_dict['eccentricity']
        # Save to self.neuron_circularity
        self.neuron_circularity = circularity
        print(len(circularity))
        return circularity
    