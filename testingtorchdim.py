import torch

file_path = '/Users/abiralshakya/Documents/Research/GraphVectorTopological/kspace_topology_graphs/SG_038/physics_features.pt'

loaded_data = torch.load(file_path)

if isinstance(loaded_data, dict) and 'decomposition_features' in loaded_data:
    tensor_data = loaded_data['decomposition_features']
    print(tensor_data.shape[0])
else:
    if isinstance(loaded_data, torch.Tensor):
        print(loaded_data.shape[0])
    else:
        print(f"Error: {file_path} does not contain a tensor at 'decomposition_features' key or is not a tensor directly. Loaded type: {type(loaded_data)}")
        print(f"Loaded content: {loaded_data}")