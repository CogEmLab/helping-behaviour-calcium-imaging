"""
Batch CNMF-E processing script.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from tifffile import imwrite
import matplotlib
import argparse

matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False

from src.loaders.config_loader import load_config
from src.cnmfe.cnmfe_utils import *
from src.utils import analyze_data, plot_random_traces
from src.plotting import quality_control_plotting


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch CNMF-E processing pipeline with config + override support."
    )

    parser.add_argument(
        "--config",
        default="config/master.yaml",
        help="Path to main config file (default: config/master.yaml)."
    )

    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        help="One or more override config files."
    )

    parser.add_argument(
        "--base_dir",
        default=".",
        help="Base directory for resolving relative config paths."
    )

    return parser.parse_args()


def main():

    # -------- Parse CLI --------
    args = parse_args()

    # -------- Load merged config --------
    config = load_config(
        args.config,
        overrides=args.override,
        base_dir=args.base_dir
    )

    # For quick access:
    cfg_general = config["caiman"]["general"]
    cfg_mc = config["caiman"]["motion_correction"]
    cfg_paths = cfg_general["data_paths"]
    cfg_sessions = cfg_general["session_file_paths"]

    skip_if_exists = cfg_general["skip_if_exists"]
    required_columns = cfg_general["required_columns"]
    documentation_path = cfg_general["documentation_path"]
    default_fps = cfg_general["default_fps"]

    # Print effective config at start of run
    print("\n========== RUNNING CNMF-E BATCH ==========")
    print("Loaded config:", args.config)
    if args.override:
        print("Overrides:", args.override)
    print("==========================================\n")

    # -------- Iterate over dataset roots --------
    for GLOBAL_PATH, session_file_path in zip(cfg_paths, cfg_sessions):

        # Set up log
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(
            documentation_path,
            f'batch_cnmfe_{timestamp}.log'
        )
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        logging.info("Starting batch CNMFE run")
        logging.info(f"Root path: {GLOBAL_PATH}")

        # Find TIFFs
        miniscope_paths = glob(
            f'{GLOBAL_PATH}/**/miniscope/**.tiff',
            recursive=True
        )

        # Load session file
        session_file_fullpath = os.path.normpath(
            os.path.join(GLOBAL_PATH, session_file_path)
        )

        session_df = pd.read_csv(session_file_fullpath)
        session_df.columns = [c.lower() for c in session_df.columns]

        # Validate required columns
        if not all(col in session_df.columns for col in required_columns):
            logging.warning(
                f"Session file {session_file_fullpath} missing required columns: "
                f"{required_columns}. Skipping."
            )
            continue

        # Ensure imaging length column
        if "ca imaging length(s)" not in session_df.columns:
            session_df["ca imaging length(s)"] = np.nan

        # Prepare storage
        SNR_dict = {}
        max_shifts_dict = {}

        # -------- Process each recording file --------
        for path in tqdm(miniscope_paths, desc=f"Processing {GLOBAL_PATH}"):

            session = path.split('/')[2]
            filename = os.path.splitext(os.path.basename(path))[0]

            logging.info(f"Processing {filename} in {session}")

            # Match entry for session
            session_entry = session_df[session_df["miniscope video"] == filename]

            if session_entry.empty:
                logging.error(f"No entry found for {filename}. Skipping.")
                continue

            idx = session_entry.index[0]

            # FPS
            if not pd.isna(session_entry["fps (ci downsampled)"].values[0]):
                FPS = int(session_entry["fps (ci downsampled)"].values[0])
            else:
                FPS = default_fps

            # Skip excluded
            if str(session_entry["excluded"].values[0]).upper() == "TRUE":
                logging.info(f"Skipping excluded file: {filename}")
                continue

            # Output directories
            OUTPUT_DIR = GLOBAL_PATH.replace("data", "processed-data")

            save_dir_model = os.path.join(OUTPUT_DIR, session, "cnmfe_model")
            save_dir_data = os.path.join(OUTPUT_DIR, session, "cnmfe_data")
            save_dir_docs = os.path.join(OUTPUT_DIR, session, "cnmfe_docs")
            save_dir_npy = os.path.join(OUTPUT_DIR, session, "npy_data")
            save_dir_quality = os.path.join(OUTPUT_DIR, "quality_control")

            for d in [
                save_dir_model, save_dir_data, save_dir_docs,
                save_dir_npy, save_dir_quality
            ]:
                os.makedirs(d, exist_ok=True)

            # Check if already processed
            if skip_if_exists and os.path.exists(
                os.path.join(save_dir_model, f"{filename}.hdf5")
            ):
                logging.info(f"{filename} already processed. Skipping.")
                continue

            # Run CNMF-E
            try:
                result = analyze_data(
                    path,
                    save_dir_docs,
                    args.config,   # path to main config file
                    fps=FPS
                )
                (
                    cnmfe_model, ci_movie, fname_new, correlation_image,
                    peak_to_noise_ratio, dims, total_components,
                    num_accepted, num_rejected, border_px
                ) = result

            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue

            # ---------------- Save CNMF-E outputs ----------------

            cnmfe_model.estimates.detrend_df_f(detrend_only=True)
            DF_F = cnmfe_model.estimates.F_dff

            # Normalize
            DF_F_z = (DF_F - DF_F.mean(1)[:, None]) / DF_F.std(1)[:, None]

            # Save model
            model_path = os.path.join(save_dir_model, f"{filename}.hdf5")
            cnmfe_model.save(model_path)

            # Save images
            imwrite(os.path.join(save_dir_data, f"correlation_{filename}.tif"),
                    correlation_image)
            imwrite(os.path.join(save_dir_data, f"pnr_{filename}.tif"),
                    peak_to_noise_ratio)

            # Save npy
            np.save(os.path.join(save_dir_npy, f"{filename}.npy"), DF_F_z)

            # Copy motion corrected file
            save_motion_corrected = (
                os.path.dirname(path)
                .replace("data", "processed-data")
                .replace("miniscope", "cnmfe_data")
            )
            os.makedirs(save_motion_corrected, exist_ok=True)
            mc_target = os.path.join(save_motion_corrected, os.path.basename(fname_new))
            os.system(f"cp {fname_new} {mc_target}")
            os.system(f"rm {fname_new}")

            # SNR
            SNR = cnmfe_model.estimates.SNR_comp
            SNR_dict.setdefault(session, []).extend(SNR)
            max_shifts_dict.setdefault(session, []).append(border_px)

            # QC plot
            try:
                plot_random_traces(
                    DF_F_z,
                    snr=SNR,
                    filename=os.path.join(save_dir_docs, f"{filename}_random.png")
                )
            except Exception as e:
                logging.error(f"Failed QC plot for {filename}: {e}")

            # Update session DF
            session_df.loc[idx, "total components"] = total_components
            session_df.loc[idx, "num accepted"] = num_accepted
            session_df.loc[idx, "num rejected"] = num_rejected
            session_df.loc[idx, "dimensions"] = str(dims)
            session_df.loc[idx, "cnmfe model path"] = model_path
            session_df.loc[idx, "motion corrected path"] = mc_target
            session_df.loc[idx, "maximum shift"] = float(border_px)
            session_df.loc[idx, "cnmfe log file location"] = log_filename
            session_df.loc[idx, "ca imaging length(s)"] = DF_F.shape[1] / FPS

        # Save final updated session file
        out_session_file = os.path.join(
            GLOBAL_PATH,
            f"SESSION_FILE_batch_cnmfe.csv"
        )
        session_df.to_csv(out_session_file, index=False)

        # QC plots
        try:
            quality_control_plotting(SNR_dict, max_shifts_dict, save_dir_quality)
        except Exception as e:
            logging.error(f"QC plotting error: {e}")

        logging.info(f"Finished cohort {GLOBAL_PATH}")

    print("Batch CNMF-E complete.\n")

if __name__ == "__main__":
    main()
