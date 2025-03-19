import os

import h5py
import pandas as pd

# Path to the HDF5 file
hdf5_path = "output_dataset/episode_1.hdf5"
output_dir = "extracted_data"

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Open the HDF5 file
with h5py.File(hdf5_path, 'r') as hdf5_file:
    # Function to extract dataset and convert it to DataFrame
    def extract_dataset(dataset_path):
        if dataset_path in hdf5_file:
            data = hdf5_file[dataset_path][:]

            # If the data is 3D (like qtorque), reshape it into 2D
            if len(data.shape) == 3:  # (timesteps, 8, 3)
                timesteps, joints, forces = data.shape
                data = data.reshape(timesteps, joints * forces)  # Convert to (timesteps, 24)

            return pd.DataFrame(data)
        else:
            print(f"Warning: {dataset_path} not found in HDF5 file.")
            return None

    # Extract joint positions, velocities, and torques for both arms
    datasets = [
        "observations/left_arm/qpos",
        "observations/right_arm/qpos",
        "observations/left_arm/qvel",
        "observations/right_arm/qvel",
        "observations/left_arm/qtorque",
        "observations/right_arm/qtorque"
    ]

    # Process and save each dataset
    for dataset in datasets:
        df = extract_dataset(dataset)
        if df is not None:
            output_file = os.path.join(output_dir, dataset.replace("/", "_") + ".csv")
            df.to_csv(output_file, index=False, header=False)
            print(f"Saved: {output_file}")

print("Extraction completed.")
