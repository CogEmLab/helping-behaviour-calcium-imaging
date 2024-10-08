import os
import caiman as cm
from glob import glob
from tqdm import tqdm
import logging
from tifffile import imread, imwrite

# Set up logging
logging.basicConfig(filename='downsample.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to downsample TIFF file by 2x in all dimensions
def cast_tiff(filename, input_path, output_path):
    try:
        # Read TIFF file
        print('here, loading...')
        movie = imread(input_path)
        print('here, loaded...')

        # Check the type of the movie
        print(movie.dtype)

        # Cast to 16 bit if the movie is not already in 16 bit
        if movie.dtype != 'uint16':
            # Cast to 16 bit

            movie = movie.astype('uint16')

            # Create the output directory if it does not exist
            # get only the directory name from the output path
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write downsampled TIFF file
            imwrite(output_path, movie)
        
        logging.info(f"Processed: {input_path} -> {output_path}")
    except Exception as e:
        logging.error(f"Error processing: {input_path} - {e}")

# Set the input directory and output directory
root = '/home/aagafono/linstore/CogEm/Projects/helping_behavior/Analysis/Calcium-Imaging-Analysis/data/Mar_2024/HBTD1/miniscope'
# Get all the TIFF files in the input directory
miniscope_videos = glob(os.path.join(root, "*.tiff"), recursive=True)
#print(miniscope_videos)

#root = 'vHPC_test_data'
#miniscope_videos = [os.path.join(root, "Moises_B6J_7883_F_HBT01_2024-06-03-11-06-06_video_converted.tiff")]

#print(miniscope_videos)

for input_path in tqdm(miniscope_videos, desc="Processing"):
    try:
        # extract the file name from the input directory
        filename = os.path.basename(input_path)
        # remove "miniscope" from the filename
        #filename = filename.replace("miniscope", "")
        # remove .tiff from the filename
        #filename = filename.replace(".tiff", "")

        logging.info(f"Processing filename: {filename}")
        logging.info(f"Processing: {input_path}")

        # get the day from the input path 
        #day = input_path.split("/")[1]
        # add miniscope folder to the output path in the beginning and add .tiff to the end
        #output_path = os.path.join(root, day, "miniscope", filename + ".tiff")
        #output_path = os.path.join(root, filename + ".tiff")
        #logging.info(f"Output: {output_path}")
        output_path = input_path
        cast_tiff(filename, input_path, output_path)

    except Exception as e:
        logging.error(f"Error processing: {input_path} - {e}")

logging.info("Processing completed.")