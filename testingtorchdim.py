import torch # Keep this import, even if commented out parts
import numpy as np
from pathlib import Path
import os
from typing import Union

# Commented out example torch.load snippets
# file_path = '/Users/abiralshakya/Documents/Research/GraphVectorTopological/kspace_topology_graphs/SG_038/physics_features.pt'
# loaded_data = torch.load(file_path)
# if isinstance(loaded_data, dict) and 'decomposition_features' in loaded_data:
#     tensor_data = loaded_data['decomposition_features']
#     print(tensor_data.shape[0])
# else:
#     if isinstance(loaded_data, torch.Tensor):
#         print(loaded_data.shape[0])
#     else:
#         print(f"Error: {file_path} does not contain a tensor at 'decomposition_features' key or is not a tensor directly. Loaded type: {type(loaded_data)}")
#         print(f"Loaded content: {loaded_data}")

# import torch
# from torch_geometric.data import Data as PyGData
# kspace_graph_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/kspace_topology_graphs/SG_009/kspace_graph.pt" 
# data = torch.load(kspace_graph_path, weights_only= False)
# print(f"Shape of kspace_graph.x: {data.x.shape}")


def get_npy_dimensions(file_path: Union[str, Path]):
    """
    Loads an .npy file and prints its shape, number of dimensions, and checks for all zeros.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return
    if not file_path.suffix == '.npy':
        print(f"Error: Not an .npy file: {file_path}")
        return

    try:
        data = np.load(file_path)
        print(f"File: {file_path}")
        print(f"  Shape: {data.shape}")
        print(f"  Number of dimensions (ndim): {data.ndim}")
        print(f"  Size (total elements): {data.size}")
        if data.ndim == 1:
            print(f"  Assuming feature dimension (if 1D array): {data.shape[0]}")
        
        # --- NEW CHECK FOR ZEROS ---
        if np.all(data == 0):
            print("  WARNING: This file contains ALL ZERO values.")
        else:
            print("  This file contains NON-ZERO values.")
        # --- END NEW CHECK ---

    except Exception as e:
        print(f"Error loading or processing {file_path}: {e}")

def get_dimensions_in_directory(base_dir: Union[str, Path], filename_pattern: str = "asph_features.npy"):
    """
    Iterates through subdirectories to find and print dimensions of specific .npy files,
    and checks if they contain all zero values.
    Assumes structure like base_dir/JID/filename_pattern.
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        print(f"Error: Directory not found: {base_dir}")
        return

    found_files = list(base_dir.rglob(filename_pattern))
    
    if not found_files:
        print(f"No files matching '{filename_pattern}' found in '{base_dir}' or its subdirectories.")
        return

    print(f"Scanning directory: {base_dir} for '{filename_pattern}'...")
    print("-" * 50)
    
    all_shapes = set()
    all_zero_files_count = 0
    total_files_checked = 0

    for file_path in found_files:
        total_files_checked += 1
        try:
            data = np.load(file_path)
            shape = data.shape
            all_shapes.add(shape)
            
            is_all_zeros = np.all(data == 0)
            if is_all_zeros:
                all_zero_files_count += 1
                status_msg = " (ALL ZEROS)"
            else:
                status_msg = ""

            print(f"File: {file_path.relative_to(base_dir)}, Shape: {shape}{status_msg}")
        except Exception as e:
            print(f"Error loading {file_path.relative_to(base_dir)}: {e}")
    
    print("-" * 50)
    if len(all_shapes) == 1:
        print(f"All '{filename_pattern}' files have a consistent shape: {list(all_shapes)[0]}")
    elif len(all_shapes) > 1:
        print("Warning: Found inconsistent shapes across files:")
        for shape in all_shapes:
            print(f"  - {shape}")
    else:
        print("No valid .npy files processed.")
    
    print(f"Summary: {all_zero_files_count} out of {total_files_checked} files contain all zeros.")
    if all_zero_files_count == total_files_checked and total_files_checked > 0:
        print("WARNING: All processed files contain ONLY ZEROS. This indicates a potential issue with feature generation.")


if __name__ == "__main__":
    #single_file_path = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/multimodal_materials_db_mp/vectorized_features/mp-2051619/asph_features.npy"
    #get_npy_dimensions(single_file_path)

    print("\n" + "="*70 + "\n")
    base_features_dir = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/multimodal_materials_db_mp/vectorized_features/"
    
    get_dimensions_in_directory(base_features_dir, "asph_features_rev2.npy")

    print("\n" + "="*70 + "\n")
    print("Remember to update config.py with the correct ASPH_FEATURE_DIM based on these results.")