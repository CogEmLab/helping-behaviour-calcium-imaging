import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import seaborn as sns
import matplotlib.pyplot as plt
import neurokit2 as nk
from src.session_loader import Session
import os
from datetime import datetime
from src.specificity import calculate_event_specificity

plt.switch_backend('agg')

def compute_multiscale_entropy(data, scales=5, m=2, r=0.15):
    def coarse_grain(data, scale):
        reshaped = data[:len(data) - len(data) % scale]
        return np.mean(reshaped.reshape(-1, scale), axis=1)
    
    max_scale = min(scales, len(data) // (m + 1))
    mse = np.zeros(max_scale)
    valid_scales = []
    
    for i in range(max_scale):
        scale = i + 1
        try:
            coarse_data = coarse_grain(data, scale)
            if len(coarse_data) > m + 1:
                entropy_value = nk.entropy_sample(coarse_data, dimension=m, tolerance=r)[0]
                if not np.isinf(entropy_value) and not np.isnan(entropy_value):
                    mse[i] = entropy_value
                    valid_scales.append(i)
        except:
            continue
    
    if len(valid_scales) == 0:
        return np.array([0.0]), [0]
    return mse[valid_scales], valid_scales

class CommunityAnalyzer:
    SESSION_ORDER = [
        'HBTD1', 'HBTD2', 'HBTD3', 'HBTD4', 'HBTD5', 'HBTD6',
        'Stranger', 'RECNvlCtx', 'Toy', 'Pool', 'Empty', 'REMNvlCtx', 'REMHBTD'
    ]

    def __init__(self, base_dir):
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            raise ValueError(f"Base directory {base_dir} does not exist")
        
        self.communities_data = None
        self.features = None
        self.tracked_communities = {}
        self.registration_mapping = {}

    def load_data(self, session_file):
        # Filter out excluded sessions
        self.session_file = session_file[session_file['excluded'] != True].copy()
        self.sessions = [Session(row) for _, row in self.session_file.iterrows()]
        
        communities_list = []
        for session in self.sessions:
            filename = f"community_data_{session.animal_id}_{session.session}.csv"
            filepath = os.path.join(self.base_dir, session.cohort, session.session, filename)
            print(f"Attempting to load community data from: {filepath}")

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    # Standardize Mouse column to 4-digit format
                    df['Mouse'] = df['Mouse'].astype(str).str.zfill(4)
                    communities_list.append(df)
                    print(f"Successfully loaded community data for animal {session.animal_id}, session {session.session}")
                except Exception as e:
                    print(f"Error loading {filepath}: {str(e)}")
            else:
                print(f"Community data file not found: {filepath}")

        if not communities_list:
            print("No community data loaded for any session")
            self.communities_data = pd.DataFrame()
        else:
            self.communities_data = pd.concat(communities_list, ignore_index=True)
            self.communities_data['Mouse'] = self.communities_data['Mouse'].astype(str)
            self.communities_data['Session'] = self.communities_data['Session'].astype(str)
            required_columns = ['pre_auc', 'post_auc', 'response_index', 'response_category']
            for col in required_columns:
                if col not in self.communities_data.columns:
                    self.communities_data[col] = np.nan  # Initialize with NaNs

        if 'Reg_ID' not in self.communities_data.columns:
            self.communities_data['Reg_ID'] = np.nan

    def initialize_session_data(self):
        """Initialize all session-related data and mappings"""
        if not hasattr(self, 'communities_data') or self.communities_data is None:
            raise ValueError("Communities data not loaded. Call load_data first.")
        
        # Add history cache
        self._neuron_history_cache = {}
        self._session_indices = {
        session: idx for idx, session in enumerate(self.SESSION_ORDER)
        }
        self._neuron_session_map = {}
        processed_sessions = set()

        # Process all sessions once
        for session in self.sessions:
            if not self.process_session(session):
                print(f"Warning: Failed to process session {session.session}")
                continue
        
        processed_sessions.add(session.session)

        # Verify all sessions were processed
        if len(processed_sessions) < len(self.sessions):
            print("Warning: Not all sessions were successfully processed")

        # Build registration mapping
        self.build_reg_id_mapping()
        
        # Save registration mapping
        self.save_registration_mapping()

        # Build optimized neuron session mapping
        for mouse in self.registration_mapping:
            self._neuron_session_map[mouse] = {}
            for reg_id, appearances in self.registration_mapping[mouse].items():
                # Sort appearances once and store
                self._neuron_session_map[mouse][reg_id] = sorted(
                    [app['session'] for app in appearances],
                    key=lambda x: self._session_indices[x]
                )

    def process_session(self, session):
        try:
            animal_id = str(int(session.animal_id)).zfill(4)
            fps = session.calcium_fps
            print(f"\nProcessing session {session.session} for mouse {session.animal_id} with FPS: {fps}")

            # Load registration files
            reg_matrix, reg_indexing = session.load_registration_files()
            if reg_matrix is None or reg_indexing is None:
                print(f"Warning: Registration files not found for animal {session.animal_id}, session {session.session}")
                return False
            
            # Get registration vector for current session
            reg_vector = reg_matrix[:, reg_indexing[session.session]]
                      
            session_communities = self.communities_data[
                (self.communities_data['Mouse'].str.zfill(4) == animal_id) &
                (self.communities_data['Session'] == session.session)
            ]
            
            if session_communities.empty:
                print(f"No communities found for session {session.session}, mouse {session.animal_id}")
                return False
            
            print(f"Found {len(session_communities)} communities for mouse {session.animal_id}, session {session.session}")
            
            # Assign Reg_IDs to each community
            for comm_id in session_communities['Community_ID'].unique():
                comm_data = session_communities[session_communities['Community_ID'] == comm_id]
                comm_neurons = comm_data['Neuron_ID'].values

                # Get registration IDs using the same logic as the event cells code
                reg_ids = np.array([np.where(reg_vector == i)[0][0] for i in comm_neurons])
                
                # Ensure lengths match before assignment
                mask = (
                    (self.communities_data['Mouse'] == animal_id) &
                    (self.communities_data['Session'] == session.session) &
                    (self.communities_data['Community_ID'] == comm_id)
                )
                rows_to_update = self.communities_data.loc[mask]
                if len(rows_to_update) == len(reg_ids):
                    self.communities_data.loc[mask, 'Reg_ID'] = reg_ids
                    print(f"Processing community {comm_id} with neurons: {comm_neurons}")
                    print(f"Assigned Reg IDs: {reg_ids}")
                else:
                    print(f"Warning: Length mismatch for community {comm_id} - skipping assignment")

                # Store in registration mapping
                for neuron_id, reg_id in zip(comm_neurons, reg_ids):
                    if not np.isnan(reg_id):
                        self.store_registration_mapping(
                            session.animal_id,
                            session.session,
                            neuron_id,
                            int(reg_id)
                        )

               
            # Check if 'Reg_ID' column exists after processing
            if 'Reg_ID' not in self.communities_data.columns:
                print(f"Warning: Failed to add 'Reg_ID' column for session {session.session}")
                return False
            return True
        
        except Exception as e:
            print(f"Error processing session {session.session}: {str(e)}")
            return False

    def track_communities_across_sessions(self):
        for animal_id in self.communities_data['Mouse'].unique():
            print(f"Processing mouse: {animal_id}")

            # Get all communities for this mouse
            mouse_comms = self.communities_data[
                self.communities_data['Mouse'] == animal_id
            ]           
          
            # Sort sessions chronologically
            sessions = sorted(
                mouse_comms['Session'].unique(),
                key=lambda x: self.SESSION_ORDER.index(x),
                reverse=True
            )

            self.tracked_communities[animal_id] = {}

            # Compare communities between consecutive sessions
            for i in range(len(sessions)-1):
                session2, session1 = sessions[i], sessions[i+1]  # Note the reversed order
                comm2_data = mouse_comms[mouse_comms['Session'] == session2]
                comm1_data = mouse_comms[mouse_comms['Session'] == session1] 
                                       
                for comm2_id in comm2_data['Community_ID'].unique():
                    comm2_reg_ids = comm2_data[comm2_data['Community_ID'] == comm2_id]['Reg_ID'].values

                    for comm1_id in comm1_data['Community_ID'].unique():
                        comm1_reg_ids = comm1_data[comm1_data['Community_ID'] == comm1_id]['Reg_ID'].values
                                
                        # Calculate overlap using reg_ids
                        common_reg_ids = np.intersect1d(comm2_reg_ids, comm1_reg_ids)
                        if len(common_reg_ids) > 0:
                            overlap_ratio = len(common_reg_ids) / len(comm2_reg_ids)
                            
                            if overlap_ratio >= 0.2:
                                key = f"{animal_id}_{comm2_id}"
                                if key not in self.tracked_communities[animal_id]:
                                    self.tracked_communities[animal_id][key] = []
                                
                                self.tracked_communities[animal_id][key].append({
                                    'Session': session1,
                                    'Community_ID': comm1_id,
                                    'Overlap_Ratio': overlap_ratio,
                                    'Common_Reg_IDs': common_reg_ids.tolist(),
                                    'Common_Neurons': len(common_reg_ids) 
                                })
        
        return self.tracked_communities

    def get_neuron_history(self, mouse, current_session, reg_id):
        """Get neuron history with optimized lookup"""
        mouse = str(int(mouse)).zfill(4)
        cache_key = f"{mouse}_{current_session}_{reg_id}"
        
        # Check cache first
        if cache_key in self._neuron_history_cache:
            return self._neuron_history_cache[cache_key]
        
        # Use pre-sorted session list from mapping
        if mouse not in self._neuron_session_map or reg_id not in self._neuron_session_map[mouse]:
            return ""
            
        current_idx = self._session_indices[current_session]
        history = []
        prev_session = None
        
        # Single pass through pre-sorted sessions
        for session in self._neuron_session_map[mouse][reg_id]:
            if self._session_indices[session] >= current_idx:
                break
            if session != prev_session:
                history.append(f"Session {session}")
                prev_session = session
        
        result = f" (From: {' -> '.join(history)})" if history else ""
        self._neuron_history_cache[cache_key] = result
        return result
        

    def build_reg_id_mapping(self):
        self.reg_id_mapping = {}
        for mouse in self.communities_data['Mouse'].unique():
            mouse_data = self.communities_data[self.communities_data['Mouse'] == mouse]
            self.reg_id_mapping[mouse] = {}
            
            # Sort sessions chronologically first
            sorted_sessions = sorted(
                mouse_data['Session'].unique(),
                key=lambda x: self.SESSION_ORDER.index(x)
            )
            
            # Process reg_ids session by session
            for session in sorted_sessions:
                session_data = mouse_data[mouse_data['Session'] == session]
                for _, row in session_data.iterrows():
                    reg_id = row['Reg_ID']
                    if pd.isna(reg_id):
                        continue
                        
                    reg_id_key = int(reg_id)
                    if reg_id_key not in self.reg_id_mapping[mouse]:
                        self.reg_id_mapping[mouse][reg_id_key] = []
                    
                    self.reg_id_mapping[mouse][reg_id_key].append({
                        'Session': session,
                        'Neuron_ID': row['Neuron_ID'],
                        'Community_ID': row['Community_ID']
                    })


    def store_registration_mapping(self, mouse, session, neuron_id, reg_id):
        """Store registration mapping with community information"""
        mouse = str(int(mouse)).zfill(4)  # Standardize mouse ID format
        
        if mouse not in self.registration_mapping:
            self.registration_mapping[mouse] = {}
        if reg_id not in self.registration_mapping[mouse]:
            self.registration_mapping[mouse][reg_id] = []
            
        # Get community information
        comm_info = self.communities_data[
            (self.communities_data['Mouse'] == mouse) &
            (self.communities_data['Session'] == session) &
            (self.communities_data['Neuron_ID'] == neuron_id)
        ]
        
        community_id = comm_info['Community_ID'].iloc[0] if not comm_info.empty else None
        
        self.registration_mapping[mouse][reg_id].append({
            'session': session,
            'neuron_id': neuron_id,
            'community_id': community_id
        })

    def save_registration_mapping(self):
        """Save registration mapping to file"""
        mapping_dir = os.path.join('processed-data', 'reg_id_mapping')
        os.makedirs(mapping_dir, exist_ok=True)
        mapping_file = os.path.join(mapping_dir, 'registration_mapping.npy')
        np.save(mapping_file, self.registration_mapping)

    def load_registration_mapping(self):
        """Load registration mapping from file"""
        mapping_file = os.path.join('processed-data', 'reg_id_mapping', 'registration_mapping.npy')
        if os.path.exists(mapping_file):
            self.registration_mapping = np.load(mapping_file, allow_pickle=True).item()

    def analyze_community_event_response(self, community_signals, victim_lib_timecode, fps, mouse_id, session_id, comm_id, comm_neurons, window_seconds=34, snr=None):
        """Analyze if a community shows liberation-related activity and categorize its response type"""
        # Calculate event window frames
        half_window = int(window_seconds/2)
        half_window_frames = int(half_window * fps)
        event_frame = int(victim_lib_timecode * fps)
        total_frames = community_signals.shape[1]
        
        event_window_tuple = (event_frame - half_window_frames, event_frame + half_window_frames)

        # Initialize rest_baseline before any conditional blocks
        rest_baseline = None
        event_window = None
            
        # Adaptive window selection based on event timing
        if event_frame < half_window_frames:
            # Early event handling
            # Use minimum window size of 5 seconds before event
            min_pre_window = min(event_frame, int(5 * fps))
            post_window = min(half_window_frames, total_frames - event_frame)
            
            # Get pre-event window
            pre_event = community_signals[:, max(0, event_frame - min_pre_window):event_frame]
            # Get post-event window
            post_event = community_signals[:, event_frame:event_frame + post_window]
            
            # Calculate baseline using sliding windows in post-event period
            sliding_windows = []
            for i in range(event_frame + post_window, total_frames - half_window_frames, int(fps)):
                window = community_signals[:, i:i + half_window_frames]
                sliding_windows.append(window)
            
            if sliding_windows:
                # Use median of sliding windows as baseline
                baseline_stack = np.concatenate(sliding_windows, axis=1)
                rest_baseline = np.percentile(baseline_stack, 10, axis=1)[:, np.newaxis]
            else:
                # Fallback to global minimum if no sliding windows available
                rest_baseline = np.percentile(community_signals, 5, axis=1)[:, np.newaxis]
            
            # Combine pre and post event windows
            event_window = np.concatenate([pre_event, post_event], axis=1)
            
            # Calculate rest metrics using sliding windows
            rest_metrics = []
            for window in sliding_windows:
                window_baselined = window - rest_baseline
                rest_metrics.append({
                    'mean': np.mean(window_baselined, axis=0),
                    'sync': self._calculate_synchronization(window),
                    'auc': np.trapz(np.mean(window_baselined, axis=0))
                })
        else:
            # Normal case - enough data on both sides
            event_window = community_signals[:, event_frame-half_window_frames:event_frame+half_window_frames]
            rest_session = np.concatenate([
                community_signals[:, :event_frame-half_window_frames],
                community_signals[:, event_frame+half_window_frames:]
            ], axis=1)
            rest_baseline = np.percentile(rest_session, 10, axis=1)[:, np.newaxis]
            
            # Calculate rest metrics using sliding windows
            rest_metrics = []
            for i in range(0, rest_session.shape[1] - half_window_frames, int(fps)):
                window = rest_session[:, i:i + half_window_frames]
                window_baselined = window - rest_baseline
                rest_metrics.append({
                    'mean': np.mean(window_baselined, axis=0),
                    'sync': self._calculate_synchronization(window),
                    'auc': np.trapz(np.mean(window_baselined, axis=0))
                })
        
        # Calculate event metrics
        event_signal_baselined = event_window - rest_baseline
        event_signal_mean = np.mean(event_signal_baselined, axis=0)
        event_auc = np.trapz(event_signal_mean[event_signal_mean > 0])
        event_sync_score = self._calculate_synchronization(event_window)
        
        # Calculate rest metrics statistics
        rest_auc_mean = np.mean([m['auc'] for m in rest_metrics]) if rest_metrics else 0
        rest_sync_mean = np.mean([m['sync'] for m in rest_metrics]) if rest_metrics else 0
        
        # Calculate changes with protection against zero division
        auc_change = ((event_auc - rest_auc_mean) / (rest_auc_mean + 1e-10)) if rest_auc_mean != 0 else event_auc
        sync_change = ((event_sync_score - rest_sync_mean) / (rest_sync_mean + 1e-10)) if rest_sync_mean != 0 else event_sync_score
        
        # Calculate entropy with adaptive window size
        event_entropy, _ = compute_multiscale_entropy(
            event_signal_mean,
            scales=min(9, len(event_signal_mean) // 4),
            m=2,
            r=0.15
        )
        
        # Calculate rest entropy using concatenated rest windows
        rest_signal = np.concatenate([m['mean'] for m in rest_metrics]) if rest_metrics else np.array([])
        if len(rest_signal) > 0:
            rest_entropy, _ = compute_multiscale_entropy(
                rest_signal,
                scales=min(9, len(rest_signal) // 4),
                m=2,
                r=0.15
            )
            entropy_change = (np.mean(event_entropy) - np.mean(rest_entropy)) / (np.mean(rest_entropy) + 1e-10)
        else:
            rest_entropy = np.zeros_like(event_entropy)
            entropy_change = 0

        #Calculate whole signal entropy once
        whole_signal = np.mean(community_signals, axis=0)
        whole_entropy, _ = compute_multiscale_entropy(
            whole_signal,
            scales=min(9, len(whole_signal) // 4),
            m=2,
            r=0.15
        )

        #Calculate event-specificity  

        p_value = calculate_event_specificity(
            np.mean(community_signals, axis=0),
            event_window_tuple,
            window_seconds,
            fps,
            plot=True
        )[0]

        print('----------------------------------------------------------------------------------------------------------------------')
        print('P VALUE:', p_value)
        print('----------------------------------------------------------------------------------------------------------------------')
        # Determine if it's a liberation community first
        is_liberation = (
            p_value <= 0.00001 and
            len(community_signals) >= 2
        )
        print(f"Marked as liberation community: {is_liberation}")

        response_type = ''
        response_score = 0
        if is_liberation:
            # Calculate response scores
            pre_response = np.sum(np.abs(event_signal_mean[:half_window_frames]))
            post_response = np.sum(np.abs(event_signal_mean[half_window_frames:]))
            total_response = pre_response + post_response

            if total_response != 0:
                response_score = (post_response - pre_response) / total_response
            else:
                response_score = 0

            # Determine response type
            if response_score < -0.1:
                response_type = 'Pre-responsive'
            elif response_score > 0.1:
                response_type = 'Post-responsive'
            else:
                response_type = 'Mixed-responsive'
        else:
                response_type = 'None'

        # Calculate per-neuron metrics
        neuron_data = []
        for idx, neuron_signal in enumerate(community_signals):
            neuron_id = comm_neurons[idx]
            
            # Calculate indices for pre- and post-windows
            start_pre = max(0, event_frame - half_window_frames)
            end_pre = event_frame
            start_post = event_frame
            end_post = min(len(neuron_signal), event_frame + half_window_frames)

            pre_window = neuron_signal[start_pre:end_pre]
            post_window = neuron_signal[start_post:end_post]
            
            # Create combined event window for peak detection
            event_window = np.concatenate([pre_window, post_window])
            
            # Detect peaks in the entire event window
            peaks, _ = find_peaks(event_window, 
                                distance=int(0.45*fps),
                                width=(int(0.3 * fps), int(10.0 * fps)),
                                height=3.0,
                                prominence=0.5,
                                rel_height=0.6
                                )
            
            if len(peaks) == 0:
                # No peaks detected - mark as non-responsive
                neuron_data.append({
                    'Mouse': mouse_id,
                    'Session': session_id,
                    'Community_ID': comm_id,
                    'is_liberation': is_liberation,
                    'Neuron_ID': neuron_id,
                    'response_category': 'non-responsive',
                    'pre_auc': np.nan,
                    'post_auc': np.nan,
                    'response_index': np.nan
                })
                continue  # Skip remaining calculations

            # Calculate baseline from pre-window (only if we have peaks)
            rest_baseline = np.percentile(pre_window, 10) if len(pre_window) > 0 else 0

            # Calculate AUC components
            pre_auc = np.trapz(pre_window - rest_baseline) if len(pre_window) > 0 else 0
            post_auc = np.trapz(post_window - rest_baseline) if len(post_window) > 0 else 0

            # Response index calculation with absolute values in denominator
            sum_auc = abs(pre_auc) + abs(post_auc)
            response_index = (post_auc - pre_auc) / (sum_auc + 1e-10) if sum_auc != 0 else 0
            
            # Categorization logic with stricter thresholds
            if response_index > 0.159:
                category = 'post-responsive'
            elif response_index < -0.159:
                category = 'pre-responsive'
            else:
                category = 'mixed'

            # --- Visualization of EVENT WINDOW ONLY ---
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Create time axis for the event window
            window_duration = (len(pre_window) + len(post_window)) / fps
            window_time = np.linspace(-len(pre_window)/fps, len(post_window)/fps, 
                                    len(pre_window) + len(post_window))
            
            # Plot baseline-corrected signal
            baselined_signal = np.concatenate([pre_window - rest_baseline, post_window - rest_baseline])
            ax.plot(window_time, baselined_signal, label='Baselined Signal', color='blue')
            
            # Add event line at 0 seconds
            ax.axvline(x=0, color='red', linestyle='--', label='Liberation Event')
            
            # Add shaded regions for pre/post windows
            ax.axvspan(window_time[0], 0, alpha=0.1, color='gray', label='Pre-event')
            ax.axvspan(0, window_time[-1], alpha=0.1, color='green', label='Post-event')
            
            ax.set_title(f"Neuron {neuron_id} | Mouse {mouse_id}\nSession {session_id} | Community {comm_id}")
            ax.set_xlabel("Time relative to event (s)")
            ax.set_ylabel("ΔF/F (baselined)")
            
            # Annotate with response details
            ax.text(0.05, 0.95, 
                    f"Response Index: {response_index:.2f}\nCategory: {category}",
                    transform=ax.transAxes, 
                    fontsize=10, 
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

            ax.legend()
            fig.tight_layout()

            # Save figure
            fig_dir = os.path.join("results", "figures", "neuron_response_figures")
            os.makedirs(fig_dir, exist_ok=True)
            fig_filename_png = f"neuron_{neuron_id}_mouse{mouse_id}_session{session_id}_community{comm_id}.png"
            fig_filename_eps = f"neuron_{neuron_id}_mouse{mouse_id}_session{session_id}_community{comm_id}.eps"
            fig_path_png = os.path.join(fig_dir, fig_filename_png)
            fig_path_eps = os.path.join(fig_dir, fig_filename_eps)

            plt.savefig(fig_path_png)    # Save as PNG
            plt.savefig(fig_path_eps, format='eps')  # Save as EPS

            neuron_data.append({
                'Mouse': mouse_id,
                'Session': session_id,
                'Community_ID': comm_id,
                'is_liberation': is_liberation,
                'Neuron_ID': neuron_id,
                'pre_auc': pre_auc,
                'post_auc': post_auc,
                'response_index': response_index,
                'response_category': category
            })
        
        # Update communities data
        update_df = pd.DataFrame(neuron_data)
        output_dir = os.path.join("results", "neuron_response_metrics")
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = f"neuron_metrics_mouse{mouse_id}_session{session_id}_community{comm_id}.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # Save the DataFrame to CSV
        update_df.to_csv(csv_path, index=False)
        print(f"Saved neuron metrics to {csv_path}")        

        index_cols = ['Mouse', 'Session', 'Community_ID', 'Neuron_ID']
        self.communities_data.set_index(index_cols, inplace=True)
        update_df.set_index(index_cols, inplace=True)
        for col in ['pre_auc', 'post_auc', 'response_index', 'response_category']:
            self.communities_data[col].update(update_df[col])
        self.communities_data.reset_index(inplace=True)
        
        # Convert neuron_data to DataFrame for easy manipulation
        neuron_df = pd.DataFrame(neuron_data)

        # Count each category
        n_total = len(neuron_df)
        n_post = (neuron_df['response_category'] == 'post-responsive').sum()
        n_pre = (neuron_df['response_category'] == 'pre-responsive').sum()
        n_mixed = (neuron_df['response_category'] == 'mixed').sum()
        n_non_responsive = (neuron_df['response_category'] == 'non-responsive').sum()

        # Calculate percentages
        pct_post = n_post / n_total if n_total > 0 else 0
        pct_pre = n_pre / n_total if n_total > 0 else 0
        pct_mixed = n_mixed / n_total if n_total > 0 else 0
        pct_non_responsive = n_non_responsive / n_total if n_total > 0 else 0

        return {
            'event_auc': event_auc,
            'auc_change': auc_change,
            'sync_score': event_sync_score,
            'sync_change': sync_change,
            'entropy': np.mean(event_entropy),
            'rest_entropy': np.mean(rest_entropy),
            'whole_entropy': np.mean(whole_entropy),
            'entropy_change': entropy_change,
            'p-value': p_value,
            'is_liberation_community': is_liberation,
            'response_score': response_score,
            'response_type': response_type,
            'pct_post_responsive': pct_post,
            'pct_pre_responsive': pct_pre,
            'pct_mixed_responsive': pct_mixed,
            'pct_non_responsive': pct_non_responsive,
            'n_post_responsive': n_post,
            'n_pre_responsive': n_pre,
            'n_mixed_responsive': n_mixed,
            'n_non_responsive': n_non_responsive,
            'n_total_neurons': n_total
        }
    
    def _calculate_synchronization(self, signals):
        corr_matrix = np.corrcoef(signals)
        return np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])

    def get_previous_session_info(self, mouse, current_session):
        available_sessions = self.session_file[self.session_file['mouse'] == mouse]['session'].unique()
        sorted_sessions = [s for s in reversed(self.SESSION_ORDER) if s in available_sessions]
        
        try:
            current_index = sorted_sessions.index(current_session)
            if current_index < len(sorted_sessions) - 1:
                return sorted_sessions[current_index + 1]
        except ValueError:
            print(f"Warning: Session {current_session} not found in order list")
        return None

    def extract_community_features(self):
        if not hasattr(self, 'reg_id_mapping'):
            raise ValueError("Registration mapping not initialized. Run initialize_session_data first.")

        features_list = []
        communities_processed = 0              
        print(f"\nProcessing {len(self.session_file)} sessions...")

        for _, session_row in self.session_file.iterrows():
            session_obj = Session(session_row)
            fps = session_obj.calcium_fps
            animal_id = str(int(session_obj.animal_id)).zfill(4)
            session_obj.load_caiman_model()
            snrs = session_obj.load_snr_values()

            if not self.process_session(session_obj):
                print(f"Skipping session {session_obj.session} due to processing error")
                continue

            liberation_time_sec = session_obj.victim_lib_timecode
            liberation_frame = int(liberation_time_sec) if not pd.isna(liberation_time_sec) else None
            
            figures_dir = os.path.join('figures', 'communities',
                                        f"mouse_{session_obj.animal_id}",
                                        f"session_{session_obj.session}")
            os.makedirs(figures_dir, exist_ok=True)
            
            # Get communities for this session
            session_communities = self.communities_data[
                (self.communities_data['Mouse'] == animal_id) & 
                (self.communities_data['Session'] == session_obj.session)
            ]
            print(f"Found {len(session_communities)} communities for mouse {session_obj.animal_id}, session {session_obj.session}")
            print(f"Processing session {session_row['session']}")

            if session_communities.empty:
                print(f"Warning: No communities found for session {session_obj.session}, mouse {session_obj.animal_id}")
                continue
                
            # Load DFF signals
            dff_signals = session_obj.load_dff_signal()
            if dff_signals is None:
                print(f"Warning: Could not load DFF signals for session {session_obj.session}, mouse {session_obj.animal_id}")
                continue
            
            # Group by Community_ID and filter for communities with 2 or more neurons
            community_sizes = session_communities.groupby('Community_ID').size()
            large_communities = community_sizes[community_sizes >= 2].index
            
            for comm_id in large_communities:
                comm_data = session_communities[session_communities['Community_ID'] == comm_id]
                
                comm_neurons = comm_data['Neuron_ID'].values
                community_signals = dff_signals[comm_neurons]
                community_snrs = snrs[comm_neurons]

                # Baseline correction
                baseline = np.median(community_signals[:, :int(10*fps)], axis=1)  # Use first 10 seconds
                corrected_signals = community_signals - baseline[:, np.newaxis]

                # Calculate event-related metrics
                if not pd.isna(liberation_time_sec):
                    event_metrics = self.analyze_community_event_response(
                        corrected_signals,
                        liberation_time_sec,
                        fps,
                        animal_id,
                        session_obj.session,
                        comm_id,
                        comm_neurons,
                        window_seconds=34,
                        snr = np.mean(community_snrs)
                    )
                else:
                    event_metrics = {
                        'event_auc': 0,
                        'auc_change': 0,
                        'sync_score': 0,
                        'sync_change': 0,
                        'entropy': 0,
                        'rest_entropy': 0,
                        'whole_entropy': 0,
                        'entropy_change': 0,
                        'p-value': 0,
                        'is_liberation_community': False,
                        'response_score': 0,
                        'response_type': 'None',
                        'pct_post_responsive': 0.0,
                        'pct_pre_responsive': 0.0,
                        'pct_mixed_responsive': 0.0,
                        'pct_non_responsive': 0.0,
                        'n_post_responsive': 0,
                        'n_pre_responsive': 0,
                        'n_mixed_responsive': 0,
                        'n_non_responsive': 0,
                        'n_total_neurons': 0
                    }

                # Then initialize features dictionary with all metrics
                features = {
                    'Mouse': session_obj.animal_id,
                    'Session': session_row['session'],
                    'Cohort': session_row['cohort'],
                    'Session Outcome': session_row['session outcome'],
                    'DayOfHelping': session_row['dayofhelping'],
                    'Latency': session_row['latencyminconverted'],
                    'Community_ID': comm_id,
                    'Size': len(comm_neurons),
                    'Sync_Score': self._calculate_synchronization(community_signals),
                    'Event_AUC': event_metrics['event_auc'],
                    'Event_Sync': event_metrics['sync_score'],
                    'Sync_Change': event_metrics['sync_change'],
                    'Event_Entropy': event_metrics['entropy'],
                    'Rest_Entropy': event_metrics['rest_entropy'],
                    'Whole_Entropy': event_metrics['whole_entropy'],
                    'p-value': event_metrics['p-value'],
                    'Is_Liberation_Community': event_metrics['is_liberation_community'],
                    'pct_post_responsive': event_metrics['pct_post_responsive'],
                    'pct_pre_responsive': event_metrics['pct_pre_responsive'],
                    'pct_mixed_responsive': event_metrics['pct_mixed_responsive'],
                    'pct_non_responsive': event_metrics['pct_non_responsive'],
                    'n_post_responsive': event_metrics['n_post_responsive'],
                    'n_pre_responsive': event_metrics['n_pre_responsive'],
                    'n_mixed_responsive': event_metrics['n_mixed_responsive'],
                    'n_non_responsive': event_metrics['n_non_responsive'],
                    'n_total_neurons': event_metrics['n_total_neurons']
                }

                if hasattr(self, 'tracked_communities'):
                    mouse = features['Mouse']
                    comm_key = f"{mouse}_{comm_id}"
                    if mouse in self.tracked_communities and comm_key in self.tracked_communities[mouse]:
                        tracking_info = self.tracked_communities[mouse][comm_key]
                        features.update({
                            'Is_Tracked': True,
                            'Next_Session': tracking_info[0]['Session'] if tracking_info else None,
                            'Next_Community_ID': tracking_info[0]['Community_ID'] if tracking_info else None,
                            'Tracking_Overlap': tracking_info[0]['Overlap_Ratio'] if tracking_info else 0,
                            'Common_Neurons': tracking_info[0]['Common_Neurons'] if tracking_info else 0
                        })
                    else:
                        features.update({
                            'Is_Tracked': False,
                            'Next_Session': None,
                            'Next_Community_ID': None,
                            'Tracking_Overlap': 0,
                            'Common_Neurons': 0
                        })


                # Calculate features
                mean_signal = np.mean(corrected_signals, axis=0)
              
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 6))

                # Calculate time vector in seconds
                time_vector = np.arange(len(mean_signal)) / fps  # Convert frames to seconds
                
                # Dynamic figure sizing based on number of neurons
                base_height = 8
                height_per_neuron = 0.4  # Increase height allocation per neuron
                fig_height = max(base_height, 4 + (len(comm_neurons) * height_per_neuron))
                
                # Create figure with two subplots stacked vertically
                fig = plt.figure(figsize=(12, fig_height))
                gs = plt.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
                
                # Top subplot for average signal
                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                time_vector = np.arange(len(mean_signal)) / fps
                # Plot average signal
                ax1.plot(time_vector, mean_signal, 'b-', label='Average Signal')
                if liberation_frame is not None:
                    ax1.axvline(x=liberation_frame, color='r', linestyle='--', label='Liberation Event')
                handles, labels = ax1.get_legend_handles_labels()
                if handles:
                    ax1.legend()
                ax1.set_title(f'Community {comm_id} Average Signal')
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('ΔF/F')

                # Add tracking information
                tracking_text = []
                # Look through all communities from previous sessions that were tracked to this one
                for prev_mouse, prev_communities in self.tracked_communities.items():
                    if prev_mouse == mouse:  # Only check same mouse
                        for prev_key, tracking_info in prev_communities.items():
                            for track in tracking_info:
                                if (track['Session'] == session_obj.session and 
                                    track['Community_ID'] == comm_id):
                                    next_comm_id = prev_key.split('_')[1]
                                    next_session = self.get_previous_session_info(mouse, session_obj.session)
                                    tracking_text.append(
                                        f"Tracked from: Session {next_session}, "
                                        f"Community {next_comm_id}, "
                                        f"Overlap: {track['Overlap_Ratio']:.2f}, "
                                        f"Common Neurons: {track['Common_Neurons']}"
                                    )
                
                if tracking_text:
                    ax1.text(0.05, 0.95, '\n'.join(tracking_text), 
                            transform=ax1.transAxes,
                            verticalalignment='top', fontsize=8, 
                            bbox=dict(facecolor='white', alpha=0.7))
                else:
                    ax1.text(0.05, 0.95, "No tracking information from previous sessions",
                            transform=ax1.transAxes,
                            verticalalignment='top', fontsize=8, 
                            bbox=dict(facecolor='white', alpha=0.7))
                    
                # Add metrics as text to the upper left corner
                metrics_text = (
                    f'Size: {len(comm_neurons)}\n'
                    f'Sync Change: {event_metrics["sync_change"]:.3f}\n'
                    f'Event Entropy: {event_metrics["entropy"]:.3f}\n'
                    f'Whole Entropy: {event_metrics["whole_entropy"]:.3f}\n'
                    f'Event AUC: {event_metrics["event_auc"]:.3f}\n'
                    f'Is Liberation Community: {features["Is_Liberation_Community"]}\n'
                    f'Response Score: {event_metrics["response_score"]}\n'
                    f'Response Type: {event_metrics["response_type"]}'
                )
                
                # Add background shading for liberation communities
                if features["Is_Liberation_Community"]:
                    ax1.set_facecolor('#e6ffe6')  # Light green background
                    ax2.set_facecolor('#e6ffe6')
                ax1.text(0.95, 0.95, metrics_text,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                        transform=ax1.transAxes,
                        verticalalignment='top',
                        horizontalalignment='left')
                
                # Calculate first peak time for each neuron for sorting
                def get_first_peak_time(signal, fps=30):
                    peaks, _ = find_peaks(signal,
                                        distance=int(2*0.225*fps),
                                        width=int(0.225*fps),
                                        height=1.5)
                    return peaks[0] if len(peaks) > 0 else np.inf

                # Sort neurons by first peak timing with error handling
                try:
                    peak_times = []
                    sorted_indices = []
                    
                    # First collect all peak times
                    for signal in corrected_signals:
                        peak_time = get_first_peak_time(signal, fps=fps)
                        peak_times.append(peak_time)
                    
                    # Convert to numpy array for sorting
                    peak_times = np.array(peak_times)
                    
                    # Handle case where no peaks are found
                    if np.all(np.isinf(peak_times)):
                        sorted_indices = np.arange(len(corrected_signals))
                    else:
                        # Sort only finite values first, then infinite values
                        finite_mask = ~np.isinf(peak_times)
                        sorted_finite = np.argsort(peak_times[finite_mask])
                        sorted_infinite = np.where(~finite_mask)[0]
                        sorted_indices = np.concatenate([sorted_finite, sorted_infinite])
                    
                    sorted_neurons = comm_neurons[sorted_indices]
                    sorted_signals = corrected_signals[sorted_indices]
                    
                except Exception as e:
                    # If sorting fails, use original order
                    sorted_neurons = comm_neurons
                    sorted_signals = corrected_signals

                # Bottom subplot for individual neurons
                # Calculate offset based on signal amplitude
                signal_range = np.ptp(community_signals)
                offset = signal_range * 1.2

                # Plot each neuron's signal with offset
                for i, (neuron_id, neuron_signal) in enumerate(zip(sorted_neurons, sorted_signals)):
                    shifted_signal = neuron_signal + (i * offset)
                    
                    # Check if neuron is a liberation cell
                    is_liberation_cell = comm_data[comm_data['Neuron_ID'] == neuron_id]['Is_Liberation_Cell'].iloc[0]
                    reg_id = int(comm_data[comm_data['Neuron_ID'] == neuron_id]['Reg_ID'].iloc[0])
                    print(f"Processing neuron {neuron_id} with reg_id {reg_id}")
                    # Get complete history for this neuron
                    
                    neuron_text = f'Neuron {neuron_id} (Reg ID: {reg_id:03d})'
                    history = self.get_neuron_history(mouse=session_row['mouse'], current_session=session_obj.session, reg_id=reg_id)
                    print(f"History for neuron {neuron_id}: {history}")
                    if history:
                        neuron_text += history

                    if is_liberation_cell:
                        neuron_text += ' ★'
                        ax2.plot(time_vector, shifted_signal, 'r-', linewidth=1.5)
                    else:
                        ax2.plot(time_vector, shifted_signal, 'b-', linewidth=1)

                    ax2.text(time_vector[-1] + 2, shifted_signal[-1], 
                            neuron_text,
                            color='red' if is_liberation_cell else 'black', 
                            fontsize=8, va='center')
                
                # Add liberation event line
                if not pd.isna(liberation_time_sec):
                    ax2.axvline(x=liberation_frame, color='r', linestyle='--')
                
                # Customize bottom subplot
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('ΔF/F (offset)')
                ax2.set_title('Individual Neuron Signals')
                
                # Add legend with smaller font for large communities
                if len(comm_neurons) > 15:
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                            fontsize='small', ncol=max(1, len(comm_neurons)//20))
                else:
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Adjust layout to prevent overlap
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4)
                
                # Save figure
                base_name = f'community_{session_obj.animal_id}_{session_obj.session}_comm{comm_id}'
                # Save in both formats with quality settings
                for fmt in ['png', 'eps']:
                    plt.savefig(os.path.join(figures_dir, f"{base_name}.{fmt}"), 
                                dpi=300, 
                                bbox_inches='tight',
                                format=fmt)
                plt.close('all')
            
                features_list.append(features)
                communities_processed += 1                                 
                print(f"Processed {communities_processed} communities for session {session_obj.session}")                        
        
        if not features_list:
            print("No features could be extracted from any session")
            return pd.DataFrame()
        
        self.features = pd.DataFrame(features_list)

        # Save features to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join('processed-data', f'liberation_communities_{timestamp}.csv')
        self.features.to_csv(output_file, index=False)
        
        print(f"\nTotal features extracted: {len(self.features)}")
        print(f"Unique sessions: {len(self.features['Session'].unique())}")
        print(f"Unique mice: {len(self.features['Mouse'].unique())}")
        self.communities_data['Is_Liberation_Community'] = features['Is_Liberation_Community']

        return self.features
    
    def _plot_tracking_validation(self, validation_df):
        """Create validation plots for community tracking"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        for animal_id in self.tracked_communities:
            for key, tracking_data in self.tracked_communities[animal_id].items():
                # Plot tracking validation using Common_Reg_IDs instead of overlap calculations
                overlaps = [d['Overlap_Ratio'] for d in tracking_data]
                plt.plot(overlaps, label=f"Community {key}")
        
        # Plot tracking ratio distribution
        sns.histplot(data=validation_df, x='Tracking_Ratio', ax=axes[0,0])
        axes[0,0].set_title('Distribution of Community Tracking Ratios')
        axes[0,0].set_xlabel('Tracking Ratio')
      
        # Plot tracked neurons by mouse
        sns.boxplot(data=validation_df, x='Mouse', y='N_Tracked_Neurons', ax=axes[1,0])
        axes[1,0].set_title('Number of Tracked Neurons by Mouse')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot tracking success over sessions
        session_success = validation_df.groupby('Session_Pair')['Tracking_Ratio'].mean()
        sns.barplot(x=session_success.index, y=session_success.values, ax=axes[1,1])
        axes[1,1].set_title('Mean Tracking Success by Session Pair')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.join('figures', 'communities', 
                    f'tracking_validation_{timestamp}.png'))
        plt.close()

    def validate_community_tracking(self):
        """Validate community tracking across sessions and generate validation metrics"""
        if not hasattr(self, 'tracked_communities') or not self.tracked_communities:
            print("Debug - No tracked communities found")
            return pd.DataFrame(columns=['Mouse', 'Session_Pair', 'Community_ID', 'Next_Community_ID', 'N_Neurons', 'N_Tracked_Neurons', 'Tracking_Ratio'])

        print(f"Debug - Number of mice with tracked communities: {len(self.tracked_communities)}")

        validation_metrics = {
            'Mouse': [],
            'Session_Pair': [],
            'Community_ID': [],
            'Next_Community_ID': [],
            'N_Neurons': [],
            'N_Tracked_Neurons': [],
            'Tracking_Ratio': []
        }

        for mouse, sessions in self.tracked_communities.items():
            print(f"\nDebug - Processing mouse: {mouse}")
            for comm_key, tracking_info in sessions.items():
                print(f"Debug - Processing community: {comm_key}")
                print(f"Debug - Tracking info: {tracking_info}")
                for track in tracking_info:
                    try:
                        curr_comm_id = int(comm_key.split('_')[1])
                        curr_comm = self.communities_data[
                            (self.communities_data['Mouse'] == mouse) &
                            (self.communities_data['Community_ID'] == curr_comm_id)
                        ]

                        if curr_comm.empty:
                            print(f"Warning: No data found for community {curr_comm_id} in mouse {mouse}")
                            continue

                        curr_session = curr_comm['Session'].iloc[0]
                        
                        validation_metrics['Mouse'].append(mouse)
                        validation_metrics['Session_Pair'].append(f"{curr_session}->{track['Session']}")
                        validation_metrics['Community_ID'].append(curr_comm_id)
                        validation_metrics['Next_Community_ID'].append(track['Community_ID'])
                        validation_metrics['N_Neurons'].append(len(curr_comm))
                        validation_metrics['N_Tracked_Neurons'].append(track['Common_Neurons'])
                        validation_metrics['Tracking_Ratio'].append(track['Overlap_Ratio'])
                    except Exception as e:
                        print(f"Error processing community key {comm_key}: {str(e)}")
                        continue

        # Check if all lists have the same length
        if len(set(len(v) for v in validation_metrics.values())) != 1:
            print("Warning: Validation metrics have inconsistent lengths. Some data may be missing.")
            min_length = min(len(v) for v in validation_metrics.values())
            validation_metrics = {k: v[:min_length] for k, v in validation_metrics.items()}

        validation_df = pd.DataFrame(validation_metrics)

        if validation_df.empty:
            print("Warning: No valid tracking data found")
            return pd.DataFrame(columns=['Mouse', 'Session_Pair', 'Community_ID', 'Next_Community_ID', 'N_Neurons', 'N_Tracked_Neurons', 'Tracking_Ratio'])

        return validation_df

    def plot_liberation_communities_temporal(self):
        """Plot temporal relationships between liberation and non-liberation communities' averaged signals in two stacked plots."""
        required_columns = ['Mouse', 'Session', 'Community_ID', 'Is_Liberation_Community']
        self.communities_data['Is_Liberation_Community'] = (
            self.communities_data['Is_Liberation_Community']
            .fillna(False)
            .astype(bool)
        )
        if not all(col in self.communities_data.columns for col in required_columns):
            raise ValueError("Missing required columns in communities_data")

        grouped = self.communities_data.groupby(['Mouse', 'Session'])
        for (mouse, session), group in grouped:
            liberation_comms = group[group['Is_Liberation_Community'] == True]
            non_liberation_comms = group[group['Is_Liberation_Community'] == False]

            if liberation_comms.empty and non_liberation_comms.empty:
                print(f"No communities found for Mouse {mouse} Session {session}")
                continue

            fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

            session_obj = next((s for s in self.sessions if 
                                str(s.animal_id) == mouse and 
                                s.session == session), None)
            if not session_obj:
                print(f"Session metadata missing for Mouse {mouse} Session {session}")
                plt.close(fig)
                continue

            fps = session_obj.calcium_fps
            lib_timecode = session_obj.victim_lib_timecode

            if lib_timecode is None or (isinstance(lib_timecode, float) and np.isnan(lib_timecode)):
                liberation_frame = None
            else:
                liberation_frame = int(lib_timecode * fps)

            # --- Liberation communities (top) ---
            if not liberation_comms.empty:
                # Filter to only communities with >1 neuron
                valid_liberation = []
                for comm_id, comm_group in liberation_comms.groupby('Community_ID'):
                    if len(comm_group['Neuron_ID'].unique()) > 1:
                        valid_liberation.append((comm_id, comm_group))
                n_liberation = len(valid_liberation)
                colors_liberation = plt.cm.viridis(np.linspace(0, 1, max(1, n_liberation)))
                for idx, (comm_id, comm_group) in enumerate(valid_liberation):
                    neuron_ids = comm_group['Neuron_ID'].unique()
                    signals = session_obj.load_dff_signal()[neuron_ids]
                    baseline = np.median(signals[:, :int(10*fps)], axis=1)
                    corrected = signals - baseline[:, np.newaxis]
                    mean_signal = np.mean(corrected, axis=0)
                    time_vector = np.arange(len(mean_signal)) / fps
                    ax_top.plot(
                        time_vector, mean_signal, 
                        color=colors_liberation[idx],
                        label=f'Liberation Community {comm_id}',
                        linewidth=2.5,
                        alpha=0.85,
                    )
                if liberation_frame is not None:
                    ax_top.axvline(
                        x=liberation_frame/fps, color='r', linestyle='--', 
                        linewidth=2, label='Liberation Event'
                    )
                ax_top.set_title('Liberation Communities')
                ax_top.set_ylabel('ΔF/F (baselined)')
                ax_top.legend(fontsize='small', loc='upper right')
            else:
                ax_top.text(0.5, 0.5, 'No liberation communities', ha='center', va='center', fontsize=14, transform=ax_top.transAxes)
                ax_top.set_title('Liberation Communities')
                ax_top.set_ylabel('ΔF/F (baselined)')
                ax_top.set_xticks([])
                ax_top.set_yticks([])

            # --- Non-liberation communities (bottom) ---
            if not non_liberation_comms.empty:
                valid_non_liberation = []
                for comm_id, comm_group in non_liberation_comms.groupby('Community_ID'):
                    if len(comm_group['Neuron_ID'].unique()) > 1:
                        valid_non_liberation.append((comm_id, comm_group))
                n_non_liberation = len(valid_non_liberation)
                colors_non_liberation = plt.cm.plasma(np.linspace(0, 1, max(1, n_non_liberation)))
                for idx, (comm_id, comm_group) in enumerate(valid_non_liberation):
                    neuron_ids = comm_group['Neuron_ID'].unique()
                    signals = session_obj.load_dff_signal()[neuron_ids]
                    baseline = np.median(signals[:, :int(10*fps)], axis=1)
                    corrected = signals - baseline[:, np.newaxis]
                    mean_signal = np.mean(corrected, axis=0)
                    time_vector = np.arange(len(mean_signal)) / fps
                    ax_bottom.plot(
                        time_vector, mean_signal, 
                        color=colors_non_liberation[idx],
                        label=f'Non-Liberation Community {comm_id}',
                        linewidth=1.5,
                        alpha=0.5,
                    )
                if liberation_frame is not None:
                    ax_bottom.axvline(
                        x=liberation_frame/fps, color='r', linestyle='--', 
                        linewidth=2, label='Liberation Event'
                    )
                ax_bottom.set_title('Non-Liberation Communities')
                ax_bottom.set_xlabel('Time (seconds)')
                ax_bottom.set_ylabel('ΔF/F (baselined)')
                ax_bottom.legend(fontsize='small', loc='upper right')
            else:
                ax_bottom.text(0.5, 0.5, 'No non-liberation communities', ha='center', va='center', fontsize=14, transform=ax_bottom.transAxes)
                ax_bottom.set_title('Non-Liberation Communities')
                ax_bottom.set_xlabel('Time (seconds)')
                ax_bottom.set_ylabel('ΔF/F (baselined)')
                ax_bottom.set_xticks([])
                ax_bottom.set_yticks([])

            fig.suptitle(f"Mouse {mouse} - Session {session}\nLiberation (top) vs. Non-Liberation Communities (bottom)")
            fig.tight_layout(rect=[0, 0, 1, 0.97])

            fig_dir = os.path.join("figures", "communities", f"mouse_{mouse}", f"session_{session}")
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, f"temporal_relationships_{mouse}_{session}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved figure to {save_path}")
            plt.close(fig)

def main():
    session_file = pd.read_csv('data/SESSION_FILE_batch_cnmfe.csv')
    
    # Create output directories
    os.makedirs('processed-data', exist_ok=True)
    os.makedirs('figures/communities', exist_ok=True)
    
    # Initialize and run analyzer
    analyzer = CommunityAnalyzer('results/communities')

    # Load and initialize data
    analyzer.load_data(session_file)
    analyzer.initialize_session_data()
   
    # Track communities
    analyzer.track_communities_across_sessions()
    
    # Validate tracking
    validation_results = analyzer.validate_community_tracking()
    print("\nTracking Validation Summary:")
    print(f"Mean tracking ratio: {validation_results['Tracking_Ratio'].mean():.3f}")

    # Extract features with tracking information
    features = analyzer.extract_community_features()
    analyzer.communities_data = (
        analyzer.communities_data
        .drop(columns=['Is_Liberation_Community'], errors='ignore')
        .merge(
            features[['Mouse', 'Session', 'Community_ID', 'Is_Liberation_Community']],
            on=['Mouse', 'Session', 'Community_ID'],
            how='left'
        )
    )
    
    if 'Is_Liberation_Community' not in analyzer.communities_data.columns:
        raise ValueError("'Is_Liberation_Community' column missing after merge")
    print("Columns in communities_data:", analyzer.communities_data.columns.tolist())
    print("Liberation communities count:", analyzer.communities_data['Is_Liberation_Community'].sum())

    analyzer.plot_liberation_communities_temporal()
    
    print("\nAnalysis complete.")
    print("Results saved in processed-data and figures/communities")

if __name__ == "__main__":
    main()
