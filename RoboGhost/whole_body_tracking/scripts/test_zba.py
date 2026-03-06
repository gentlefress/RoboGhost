
import torch

# Load the original .pt files
file1 = torch.load('/data/lizhe/roboghost/general.pt')
file2 = torch.load('/data/lizhe/zba/motion_latents_500.pt')

# Get the keys from both dictionaries
keys_file1 = set(file1.keys())
keys_file2 = set(file2.keys())

# Find common keys
common_keys = keys_file1.intersection(keys_file2)

# Remove common keys from the first file
for key in common_keys:
    del file1[key]

# Save the modified dictionary to a new .pt file
torch.save(file1, '/data/lizhe/zba/modified_general.pt')

# Load the modified general file and the motion latents file
modified_file = torch.load('/data/lizhe/zba/modified_general.pt')
motion_latents_file = torch.load('/data/lizhe/zba/motion_latents_500.pt')

# Squeeze the first dimension of motion_latents_file
squeezed_motion_latents = {key: torch.squeeze(value, dim=0) / 1000.0 for key, value in motion_latents_file.items()}

# Merge the two dictionaries
merged_file = {**modified_file, **squeezed_motion_latents}

# Save the merged dictionary to a new .pt file
torch.save(merged_file, '/data/lizhe/zba/merged_file.pt')

print("Processed the files and saved as /data/lizhe/zba/merged_file.pt")




# from joblib import load, dump

# # Load the two .pkl files using joblib
# humanml_data = load('/data/lizhe/zba/whole_body_tracking/dataset/pkl/humanml_497_frames_select_merged_0_1.pkl')
# roboghost_data = load('/data/lizhe/zba/whole_body_tracking/dataset/pkl/roboghost_all.pkl')

# # Create a new dictionary to hold the merged data
# merged_data = {}

# # Add humanml data with keys starting from 0
# for i, (key, value) in enumerate(humanml_data.items()):
#     merged_data[i] = value

# # Add roboghost data, continuing from the end of humanml keys
# for i, (key, value) in enumerate(roboghost_data.items(), start=len(humanml_data)):
#     merged_data[i] = value

# # Save the merged dictionary to a new .pkl file using joblib
# dump(merged_data, '/data/lizhe/zba/whole_body_tracking/dataset/pkl/merged_data.pkl')

# print("Merged the pkl files and saved as /data/lizhe/zba/whole_body_tracking/dataset/pkl/merged_data_from_humanml_and_roboghost.pkl")