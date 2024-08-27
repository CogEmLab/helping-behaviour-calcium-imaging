import os
import pandas as pd
import numpy as np
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from tifffile import imread

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
        
        # Construct the annotations path
        annotations_filename = data_row['ManualAnnotationSide']
        self.annotations_path = os.path.join('annotations', self.cohort, self.session, f'{annotations_filename}.csv')
        
        # Construct the DFF path
        self.dff_path = self.model_path.replace('.hdf5', '_Dff_z_scored.npy').replace('cnmfe_model', 'npy_data')
        
        # Construct the accepted components path
        output_dir = os.path.dirname(self.model_path)
        self.accepted_components_path = os.path.join(output_dir, 'accepted_components_{}.npy'.format(os.path.basename(self.model_path).replace('.hdf5', '')))
        
        self.caiman_model = None
        self.liberation_cells_ids = None
        self.snr_values = None 

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
        pnr_path = f"{self.model_path.rsplit('.', 1)[0]}_pnr.npy"
        if not os.path.exists(pnr_path):
            print(f"Warning: PNR image file not found at {pnr_path}")
            return None
        try:
            return imread(pnr_path)
        except Exception as e:
            print(f"Error loading PNR image: {str(e)}")
            return None

    def load_dff_signal(self):
        """Load the DFF calcium signal."""
        if not os.path.exists(self.dff_path):
            print(f"Warning: DFF signal file not found at {self.dff_path}")
            return None
        try:
            return np.load(self.dff_path)
        except Exception as e:
            print(f"Error loading DFF signal: {str(e)}")
            return None

    def load_liberation_metadata(self, metadata_path):
        """
        Load and filter liberation metadata for the current session.
        
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
            return filtered_metadata

        except pd.errors.EmptyDataError:
            print(f"Warning: The liberation metadata file at {metadata_path} is empty")
        except KeyError as e:
            print(f"Error: The liberation metadata file is missing expected column: {str(e)}")
        except Exception as e:
            print(f"Error occurred while loading liberation metadata: {str(e)}")
        return None

    def load_accepted_components(self):
        """Load accepted components."""
        if self.caiman_model is not None:
            return self.caiman_model.estimates.idx_components
        elif os.path.exists(self.accepted_components_path):
            try:
                return np.load(self.accepted_components_path)
            except Exception as e:
                print(f"Error loading accepted components: {str(e)}")
        else:
            print(f"Warning: Accepted components file not found at {self.accepted_components_path}")
        return None

    def load_annotations(self):
        """Load annotations."""
        if not os.path.exists(self.annotations_path):
            print(f"Warning: Annotations file not found at {self.annotations_path}")
            return None
        try:
            return pd.read_csv(self.annotations_path)
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
    