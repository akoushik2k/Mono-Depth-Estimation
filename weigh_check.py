import torch

# Define paths to the weights
weights_dir = "/home/koushik/FAID/RT-MonoDepth-main/log/mono_model/models/weights_19"
encoder_path = f"{weights_dir}/encoder.pth"
depth_path = f"{weights_dir}/depth.pth"
pose_encoder_path = f"{weights_dir}/pose_encoder.pth"
pose_path = f"{weights_dir}/pose.pth"
# python evaluate_depth_full.py --data_path kitti_data --load_weights_folder /home/koushik/FAID/RT-MonoDepth-main/log/mono_model/models/weights_19 --eval_mono

# Load the weights
try:
    encoder_weights = torch.load(encoder_path)
    depth_weights = torch.load(depth_path)
    pose_encoder_weights = torch.load(pose_encoder_path)
    pose_weights = torch.load(pose_path)

    print("Weights loaded successfully:")
    # print(f"Encoder weights: {encoder_weights.keys()}")
    # print(f"Depth weights: {depth_weights.keys()}")
    # print(f"Pose encoder weights: {pose_encoder_weights.keys()}")
    print(f"Pose weights: {pose_weights.keys()}")
except Exception as e:
    print(f"Error loading weights: {e}")
