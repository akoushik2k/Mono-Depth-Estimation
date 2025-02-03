# import torch
# import os
# from networks.RTMonoDepth.RTMonoDepth import MobileNetV2Encoder, DepthDecoder

# # Path to your saved model files
# encoder_path = "/home/koushik/FAID/RT-MonoDepth-main/log/mono_model/models/weights_0/encoder.pth"
# decoder_path = "/home/koushik/FAID/RT-MonoDepth-main/log/mono_model/models/weights_0/depth.pth"

# # Ensure the paths exist
# if not os.path.exists(encoder_path):
#     raise FileNotFoundError(f"Encoder file not found at {encoder_path}")

# if not os.path.exists(decoder_path):
#     raise FileNotFoundError(f"Decoder file not found at {decoder_path}")

# # Initialize the model components
# encoder = MobileNetV2Encoder().to('cuda')
# decoder = DepthDecoder(encoder.num_ch_enc).to('cuda')

# # Load the state dicts
# encoder_state_dict = torch.load(encoder_path, map_location='cuda')
# decoder_state_dict = torch.load(decoder_path, map_location='cuda')

# # Filter the encoder state_dict to include only keys that match the model
# filtered_encoder_state_dict = {
#     k: v for k, v in encoder_state_dict.items() if k in encoder.state_dict()
# }

# # Load the filtered state_dicts into the models
# encoder.load_state_dict(filtered_encoder_state_dict, strict=False)
# decoder.load_state_dict(decoder_state_dict, strict=False)


# print("Models successfully loaded.")

# # Example usage: Forward pass
# # Assuming you have an input tensor `input_tensor`
# # input_tensor = torch.randn(1, 3, 256, 256).to('cuda')  # Example input
# # features = encoder(input_tensor)
# # outputs = decoder(features)
# # print(outputs)


import numpy as np

gt_path = "/home/koushik/FAID/RT-MonoDepth-main/splits/eigen/gt_depths.npz"
data = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)
for key in data.keys():
    print(f"Key: {key}, Shape: {data[key].shape}, Type: {data[key].dtype}")
