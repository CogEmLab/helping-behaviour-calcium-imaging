# Import necessary libraries
import cv2
import os
from glob import glob

# Step one directory up
os.chdir("..")

# in data, find all mp4 files in all subfolders
path = 'data/May_2024'
all_videos = glob(os.path.join(path, '**', '**', '*.mp4'), recursive=True)

print(all_videos)
# for each video, subsample to 30 fps
corr_fps = 30

for video in all_videos:

    # Skip if the file is already subsampled
    if 'subsampled_to_30fps' in video:
        continue

    # Open the video file
    cap = cv2.VideoCapture(video)
    
    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if the frame rate is not 30 fps, then change it to 30 fps
    if fps !=corr_fps:
        print('Changing frame rate for: ', video)

        # Create new file name
        filename_new = video.split('.')[0] + f'_subsampled_to_30fps.mp4'

        # Create a VideoWriter object
        out = cv2.VideoWriter(filename_new, cv2.VideoWriter_fourcc(*'mp4v'), corr_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        while True:
            # Read a frame from the video file
            ret, frame = cap.read()
            if ret:
                # Write only every 2nd frame to the output video file
                out.write(frame) if cap.get(cv2.CAP_PROP_POS_FRAMES) % 2 == 0 else None
            else:
                break
                        
        # Release the VideoCapture and VideoWriter objects
        cap.release()
        out.release()