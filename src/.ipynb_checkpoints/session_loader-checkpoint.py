import os
import pandas as pd
import numpy as np
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from tifffile import imread
from src.utils import get_behaviour_timestamps, smooth_and_shrink_label
from scipy import ndimage

class Session:
    def __init__(self, data_row):
        """
        Initialize the Session object with data from a pandas DataFrame row.
        
        :param data_row: A pandas Series representing a row from a DataFrame
        """
        self.cohort = data_row['Cohort']
        self.session = data_row['Session']
        self.animal_id = data_row['Mouse']
        self.sex = data_row['Sex']
        self.session_length = data_row['Ca Imaging Length(s)']
        self.session_outcome = data_row['Session Outcome']
        self.calcium_fps = data_row['FPS (CI downsampled)']
        self.victim_lib_timecode = data_row['TimeVictimHeadCrossing(s)']
        self.day_of_helping = data_row['DayofHelping']
        self.model_path = data_row['CNMFE Model Path']
        self.path_to_pnr = data_row['Peak-to-Noise Ratio Path']
        self.number_of_components_total = data_row['Total Components']
        self.number_of_accepted_components = data_row['Num Accepted']
        self.number_of_liberation_cells = data_row['Num of liberation cells']

        # Construct the annotations path
        annotations_filename = data_row['ManualAnnotationSide']
        self.annotations_path = os.path.join('annotations', self.cohort, self.session, f'{annotations_filename}.csv')
        
        # Construct the DFF path
        self.dff_path = self.model_path.replace('.hdf5', '_Dff_z_scored.npy').replace('cnmfe_model', 'npy_data')
        
        # Construct the accepted components path
        output_dir = os.path.dirname(self.dff_path)
        self.accepted_components_path = os.path.join(output_dir, 'accepted_components_{}.npy'.format(os.path.basename(self.model_path).replace('.hdf5', '')))
        
        self.caiman_model = None
        self.liberation_cells_ids = None
        self.snr_values = None 
        self.pnr_image_dims = None  # To store dimensions of the PNR image

        # New attributes
        self.neuron_labels = None
        self.neuron_positions = None

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
            assert dff_signal.shape[0] == self.number_of_components_total, (
                f"Mismatch in number of components: expected {self.number_of_components_total}, "
                f"but got {dff_signal.shape[0]} in DFF signal."
            )
            
            # Create an index array to pad the dff_signal with
            indices = np.arange(1, self.number_of_components_total + 1).reshape(-1, 1)
            
            # Concatenate the index array with the DFF signal array
            dff_signal_padded = np.hstack((indices, dff_signal))
            
            return dff_signal_padded
        except Exception as e:
            print(f"Error loading DFF signal: {str(e)}")
            return None

    def filter_dff_signal(self, dff_signal_padded, component_list):
        """
        Filter the DFF signal to include only the rows corresponding to the provided component list.
        
        :param dff_signal_padded: The DFF signal matrix padded with an index column
        :param component_list: List of component indices to retain (either accepted components or liberation components)
        :return: Filtered DFF signal matrix containing only the needed components
        """
        # Ensure the component_list is a numpy array
        component_list = np.array(component_list)
        
        # Get the index column from the padded DFF signal
        indices = dff_signal_padded[:, 0]
        
        # Create a mask to filter rows where the index is in the component_list
        mask = np.isin(indices, component_list)
        
        # Apply the mask to filter the DFF signal
        filtered_dff_signal = dff_signal_padded[mask, :]
        
        # Remove the index column after filtering (if you want just the DFF data)
        # filtered_dff_signal = filtered_dff_signal[:, 1:]
        
        return filtered_dff_signal

    def load_liberation_metadata(self, metadata_path):
        """
        Load and filter liberation metadata for the current session, and assert the number of liberation cells matches.
        
        :param metadata_path: Path to the liberation metadata CSV file
        :return: Filtered pandas DataFrame containing liberation metadata for this session,
                or None if the file is not found or data for this session is not present
        """
        if not os.path.exists(metadata_path):
            print(f"Warning: Liberation metadata file not found at {metadata_path}")
            return None

        try:
            full_metadata = pd.read_csv(metadata_path)
            filtered_metadata = full_metadata[
                (full_metadata['Cohort'] == self.cohort) &
                (full_metadata['Mouse'] == self.animal_id) &
                (full_metadata['Session'] == self.session)
            ]

            if filtered_metadata.empty:
                print(f"Warning: No liberation metadata found for Cohort: {self.cohort}, Mouse: {self.animal_id}, Session: {self.session}")
                return None

            self.liberation_cells_ids = filtered_metadata['Component ID'].tolist()

            # Assert that the number of liberation cells matches the expected total
            assert len(self.liberation_cells_ids) == self.number_of_liberation_cells, (
                f"Mismatch in number of liberation cells: expected {self.number_of_liberation_cells}, "
                f"but got {len(self.liberation_cells_ids)}."
            )

            return filtered_metadata

        except pd.errors.EmptyDataError:
            print(f"Warning: The liberation metadata file at {metadata_path} is empty")
        except KeyError as e:
            print(f"Error: The liberation metadata file is missing expected column: {str(e)}")
        except Exception as e:
            print(f"Error occurred while loading liberation metadata: {str(e)}")
        return None

    def load_accepted_components(self):
        """Load accepted components and assert the number matches the expected count."""
        if self.caiman_model is not None:
            accepted_components = self.caiman_model.estimates.idx_components
        elif os.path.exists(self.accepted_components_path):
            try:
                accepted_components = np.load(self.accepted_components_path)
            except Exception as e:
                print(f"Error loading accepted components: {str(e)}")
                return None
        else:
            print(f"Warning: Accepted components file not found at {self.accepted_components_path}")
            return None

        # Assert that the number of accepted components matches the expected total
        assert len(accepted_components) == self.number_of_accepted_components, (
            f"Mismatch in number of accepted components: expected {self.number_of_accepted_components}, "
            f"but got {len(accepted_components)}."
        )
        
        return accepted_components
    
    def get_neuron_labels(self, binary_output=False, smooth_sigma=3, shrink_pixels=2):
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

        # Store the labels and positions
        self.neuron_labels = (A_labels > 0).astype(int) if binary_output else A_labels
        self.neuron_positions = self.extract_neuron_positions(self.neuron_labels)

        return self.neuron_labels

    def extract_neuron_positions(self, labels):
        """
        Extract the centroid positions of neurons from the labeled image.

        :param labels: Labeled image or binary mask.
        :return: List of centroid positions (row, col) for each neuron.
        """
        return np.array(ndimage.center_of_mass(labels, labels, range(1, np.max(labels) + 1)))

    def load_annotations(self):
        """Load annotations."""
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

    def get_liberation_cells_ids(self):
        """
        Get the liberation cell IDs for this session.
        
        :return: List of liberation cell IDs or None if not loaded yet
        """
        return self.liberation_cells_ids
    
    def get_snr_values(self):
        """
        Get the SNR values for this session.
        
        :return: Array of SNR values or None if not loaded yet
        """
        return self.snr_values
    