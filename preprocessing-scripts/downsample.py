import os
import caiman as cm
from glob import glob
from tqdm import tqdm
import logging
from tifffile import imread, imwrite

# Set up logging
logging.basicConfig(filename='downsample.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step one directory up
os.chdir("..")

def downsample_tiff(filename, input_path, output_name, frame_rate, downsample_time=True, downsampling_z_factor=0.5):
    """
    Downsample a TIFF file by 2x in spatial dimensions and/or time and save the output to a new directory.

    Parameters
    ----------
    input_path : str
        Path to the input TIFF file.
    output_name : str
        Path to the output directory and file name.
    frame_rate : int
        Frame rate of the input TIFF file (should correspond to original acquisition frame rate).
    downsample_time : bool
        Whether to downsample in time.
    downsampling_z_factor : float
        Factor to downsample in the z dimension.
    
    Returns
    -------
    None
    """

    # Log the input parameters
    logging.info(f"Downsampling: {input_path} -> {output_path}")
    logging.info(f"Frame rate: {frame_rate}")
    logging.info(f"Downsample time: {downsample_time}")
    logging.info(f"Time downsampling factor: {downsampling_z_factor}")

    # Read TIFF file
    movie = cm.load(input_path, fr=frame_rate)
    
    # Downsample the movie
    if downsample_time:
        # Downsample by 2x in all dimensions
        downsampled = movie.resize(fx=0.5, fy=0.5, fz=downsampling_z_factor)
    else:
        # Downsample by 2x in spatial dimensions only
        downsampled = movie.resize(fx=0.5, fy=0.5)
    
    # Cast to 16 bit if the movie is not already in 16 bit
    if downsampled.dtype != 'uint16':
        # Cast to 16 bit
        downsampled = downsampled.astype('uint16')

    # Write downsampled TIFF file
    imwrite(output_name, downsampled)
    
    logging.info(f"Processed: {input_path} -> {output_name}")

# Set the input directory and output directory
input_root = r"/home/aagafono/linstore/CogEm/Projects/helping_behavior/data/dHPC_miniscope/"
output_root = "data/"

# Get all the TIFF files in the input directory
# miniscope_videos = glob(os.path.join(input_root, "**/*.tiff"), recursive=True)

miniscope_videos = []

# Read txt file with the list of files
with open("all_dirs.txt", "r") as f:
    for line in f:
        miniscope_videos.append(line.strip())
    
for input_path in tqdm(miniscope_videos, desc="Processing"):

    # Extract the file name from the input directory
    filename = os.path.basename(input_path)

    output_path = input_path.replace(input_root, output_root)
    output_path = os.path.dirname(output_path)
    output_name = os.path.join(output_path,filename)
    # Create the output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    
    logging.info(f"Processing filename: {filename}")
    logging.info(f"Processing input path: {input_path}")

    # Check if os.path.join(output_path,filename) exists, then skip
    if os.path.exists(os.path.join(output_path,filename)):
        logging.info(f"File already exists: {output_path}")
        continue

    logging.info(f"Output: {output_path}")
    
    # Check if it's one of the files not to downsample in time
    if filename in ["tiff_data_0844_day1.tiff", "tiff_data_0847_day1.tiff", "tiff_data_5593_day1.tiff"]:
        downsample_tiff(filename, input_path, output_name, frame_rate=20, downsample_time=False, downsampling_z_factor=None)
    
    # Check if file belongs to the pool, empty, or toy sessions where the frame rate is 40
    elif "Mar" in input_path and ("Pool" in input_path or "Empty" in input_path or "Toy" in input_path):
        downsample_tiff(filename, input_path, output_name, frame_rate=40, downsample_time=True, downsampling_z_factor=0.75)

    else:
        downsample_tiff(filename, input_path, output_name, frame_rate=60, downsample_time=True, downsampling_z_factor=0.5) #^ Downsample by 2x in time; Default.

logging.info("Processing completed.")