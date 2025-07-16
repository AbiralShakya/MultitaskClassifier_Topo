import torch
from helper.dataset import MaterialDataset
import helper.config as config

dataset = MaterialDataset(
    master_index_path=config.MASTER_INDEX_PATH,
    kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
    data_root_dir=config.DATA_DIR,
    dos_fermi_dir=config.DOS_FERMI_DIR,
    preload=True
)

# Get the first sample
sample = dataset[0]
kspace_graph = sample['kspace_graph']

print("=== K-space Graph (Brillouin Zone) Data ===")
print(f"Number of k-points: {kspace_graph.x.shape[0]}")
print(f"Node feature shape: {kspace_graph.x.shape}")
print(f"K-point positions shape: {kspace_graph.pos.shape}")
print(f"First 5 k-point positions:\n{kspace_graph.pos[:5]}")

if hasattr(kspace_graph, 'symmetry_labels') and kspace_graph.symmetry_labels is not None:
    print(f"Symmetry labels shape: {kspace_graph.symmetry_labels.shape}")
    print(f"First 5 symmetry labels: {kspace_graph.symmetry_labels[:5]}")
else:
    print("No symmetry labels present.")

if hasattr(kspace_graph, 'edge_index') and kspace_graph.edge_index is not None:
    print(f"Edge index shape: {kspace_graph.edge_index.shape}")
    print(f"First 5 edges:\n{kspace_graph.edge_index[:, :5]}")
else:
    print("No edge_index present.")

print("First 5 node features:")
print(kspace_graph.x[:5]) 