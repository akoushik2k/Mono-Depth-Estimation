import cv2
import os
import glob
import numpy as np

def create_video_from_images(rgb_folder, disp_folder, output_video, fps=30):
    # Get sorted lists of RGB and disparity images
    rgb_images = sorted(glob.glob(os.path.join(rgb_folder, "*.jpg")))
    disp_images = sorted(glob.glob(os.path.join(disp_folder, "*_disp.jpeg")))

    # Check if both folders have the same number of images
    if len(rgb_images) != len(disp_images):
        raise ValueError("The number of RGB and disparity images must match.")

    # Read the first RGB and disparity images to get dimensions
    rgb_sample = cv2.imread(rgb_images[0])
    disp_sample = cv2.imread(disp_images[0])

    # Ensure both images have the same width
    if rgb_sample.shape[1] != disp_sample.shape[1]:
        raise ValueError("RGB and disparity images must have the same width.")

    # Get combined frame dimensions
    combined_height = rgb_sample.shape[0] + disp_sample.shape[0]
    combined_width = rgb_sample.shape[1]
    frame_size = (combined_width, combined_height)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # Iterate through images and write to the video
    for rgb_path, disp_path in zip(rgb_images, disp_images):
        # Read RGB and disparity images
        rgb_image = cv2.imread(rgb_path)
        disp_image = cv2.imread(disp_path)

        # Stack images vertically
        combined_frame = np.vstack((rgb_image, disp_image))

        # Write the frame to the video
        out.write(combined_frame)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video}")

# Example usage
rgb_folder = "/home/koushik/FAID/RT-MonoDepth-main/kitti_data/2011_09_28/2011_09_28_drive_0001_sync/image_02/data"  # Replace with the path to the folder containing RGB images
disp_folder = "/home/koushik/FAID/RT-MonoDepth-main/kitti_data/2011_09_28/2011_09_28_drive_0001_sync/image_02/data"  # Replace with the path to the folder containing disparity images
output_video = "output_video.mp4"  # Replace with the desired output video file name
fps = 30  # Frames per second

create_video_from_images(rgb_folder, disp_folder, output_video, fps)
