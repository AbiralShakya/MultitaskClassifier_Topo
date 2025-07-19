import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
from helper.config import *
from training.classifier_training import EnhancedMultiModalMaterialClassifier
from helper.dataset import MaterialDataset, custom_collate_fn

# --- Load model and data ---
model_path = 'saved_models/best_model.pt'  # Change as needed
batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset (validation or test)
dataset = MaterialDataset(MASTER_INDEX_PATH, KSPACE_GRAPHS_DIR, DATA_DIR, DOS_FERMI_DIR, preload=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Load model
model = EnhancedMultiModalMaterialClassifier(
    crystal_node_feature_dim=CRYSTAL_NODE_FEATURE_DIM,
    kspace_node_feature_dim=KSPACE_GRAPH_NODE_FEATURE_DIM,
    scalar_feature_dim=SCALAR_TOTAL_DIM,
    decomposition_feature_dim=DECOMPOSITION_FEATURE_DIM,
    num_topology_classes=NUM_TOPOLOGY_CLASSES,
    num_magnetism_classes=2  # or whatever is correct
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Collect features and labels ---
fused_features = []
attention_weights = []
labels = []

with torch.no_grad():
    for batch in tqdm(loader, desc='Extracting features'):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif hasattr(batch[key], 'batch'):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], dict):
                for sub_key in batch[key]:
                    if isinstance(batch[key][sub_key], torch.Tensor):
                        batch[key][sub_key] = batch[key][sub_key].to(device)
        _ = model(batch)
        fused = model.last_fused.cpu().numpy()
        attn = model.last_attention.cpu().numpy()
        fused_features.append(fused)
        attention_weights.append(attn)
        labels.append(batch['topology_label'].cpu().numpy())

fused_features = np.concatenate(fused_features, axis=0)
attention_weights = np.concatenate(attention_weights, axis=0)
labels = np.concatenate(labels, axis=0)

# --- t-SNE ---
tsne = TSNE(n_components=2, random_state=42)
tsne_emb = tsne.fit_transform(fused_features)
plt.figure(figsize=(8,6))
sns.scatterplot(x=tsne_emb[:,0], y=tsne_emb[:,1], hue=labels, palette='Set1', alpha=0.7)
plt.title('t-SNE of Fused Features')
plt.savefig('tsne_fused_features.png')
plt.close()

# --- UMAP ---
umap_emb = umap.UMAP(n_components=2, random_state=42).fit_transform(fused_features)
plt.figure(figsize=(8,6))
sns.scatterplot(x=umap_emb[:,0], y=umap_emb[:,1], hue=labels, palette='Set1', alpha=0.7)
plt.title('UMAP of Fused Features')
plt.savefig('umap_fused_features.png')
plt.close()

# --- Attention Weights ---
# Assume attention is [batch, 1, feature_dim] or similar
avg_attn = attention_weights.mean(axis=0).squeeze()
plt.figure(figsize=(8,4))
plt.bar(range(len(avg_attn)), avg_attn)
plt.title('Average Attention Weights (all modalities/features)')
plt.xlabel('Feature Index')
plt.ylabel('Attention Weight')
plt.savefig('avg_attention_weights.png')
plt.close()

print('t-SNE, UMAP, and attention plots saved.')
print(f'Fused features shape: {fused_features.shape}')
print(f'Labels shape: {labels.shape}')
print(f'Average attention weights shape: {avg_attn.shape}') 