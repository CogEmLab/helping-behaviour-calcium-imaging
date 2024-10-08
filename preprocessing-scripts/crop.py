import os
import subprocess
import glob

# Get one step up from the current directory
os.chdir("..")

# Set the input directory and output directory
input_root = "data/May_2024" 
output_root = "data/May_2024" 

# Function to process files in a directory
def process_files_in_dir(input_dir):
    
    # remove the file name from the input directory 
    file_name = os.path.basename(input_dir)
    output_dir = os.path.dirname(input_dir).replace(input_root, output_root)

    print("Processing filename:", file_name)
    
    # Extract the file name without extension
    filename_no_ext = os.path.splitext(file_name)[0]

    # Print the file name
    print("Processing:", input_dir)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Output file path
    output_file = os.path.join(output_dir, f"{filename_no_ext}_cropped.mp4")

    # Run ffmpeg command to crop the video
    subprocess.run(["ffmpeg", "-i", input_dir, "-vf", "crop=500:320:515:560", output_file])

    print("Processed:", input_dir, "->", output_file)
    

# Iterate over all directories and subdirectories in the input root
dirs = glob.glob(input_root + "/**/bottom_camera/*subsampled_to_30fps.mp4")
print(dirs)
for dir in dirs:
    process_files_in_dir(dir)
