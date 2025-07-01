#!/usr/bin/env python3
"""
Integrate Berry curvature data into k-space graphs for the main classification project.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data as PyGData

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_berry_curvature_data(berry_data_dir: Path, mp_id: str) -> Optional[np.ndarray]:
    """
    Load Berry curvature data for a material.
    
    Args:
        berry_data_dir: Directory containing Berry curvature data
        mp_id: Materials Project ID
        
    Returns:
        Berry curvature array or None if not found
    """
    berry_file = berry_data_dir / f"{mp_id}_berry_curvature.npy"
    kpoints_file = berry_data_dir / f"{mp_id}_kpoints.npy"
    
    if berry_file.exists() and kpoints_file.exists():
        berry_curvature = np.load(berry_file)
        kpoints = np.load(kpoints_file)
        return berry_curvature, kpoints
    else:
        return None

def interpolate_berry_curvature_to_kpoints(
    berry_curvature: np.ndarray, 
    berry_kpoints: np.ndarray, 
    target_kpoints: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate Berry curvature from computed k-grid to target k-points.
    
    Args:
        berry_curvature: Berry curvature values [num_kpoints, num_bands]
        berry_kpoints: K-points where Berry curvature was computed [num_kpoints, 3]
        target_kpoints: Target k-points for interpolation [num_target_kpoints, 3]
        method: Interpolation method ('linear', 'nearest', 'cubic')
        
    Returns:
        Interpolated Berry curvature [num_target_kpoints, num_bands]
    """
    from scipy.interpolate import griddata
    
    # Sum over bands if needed
    if berry_curvature.ndim > 1:
        berry_curvature_summed = np.sum(berry_curvature, axis=1)
    else:
        berry_curvature_summed = berry_curvature
    
    # Interpolate
    interpolated = griddata(
        berry_kpoints, 
        berry_curvature_summed, 
        target_kpoints, 
        method=method,
        fill_value=0.0
    )
    
    return interpolated

def integrate_berry_curvature_into_kspace_graph(
    kspace_graph: PyGData, 
    berry_curvature: np.ndarray,
    config: Dict[str, Any]
) -> PyGData:
    """
    Integrate Berry curvature as additional node features in k-space graph.
    
    Args:
        kspace_graph: Original k-space graph
        berry_curvature: Berry curvature values for each k-point
        config: Integration configuration
        
    Returns:
        Updated k-space graph with Berry curvature features
    """
    # Get current node features
    current_features = kspace_graph.x
    
    # Ensure Berry curvature has the right shape
    if len(berry_curvature) != current_features.shape[0]:
        raise ValueError(f"Berry curvature length ({len(berry_curvature)}) doesn't match number of k-points ({current_features.shape[0]})")
    
    # Normalize Berry curvature if configured
    if config['integration_settings']['normalize_berry_curvature']:
        berry_curvature = (berry_curvature - np.mean(berry_curvature)) / (np.std(berry_curvature) + 1e-8)
    
    # Add Berry curvature as additional features
    berry_curvature_tensor = torch.tensor(berry_curvature, dtype=torch.float32).unsqueeze(-1)
    
    # Concatenate with existing features
    updated_features = torch.cat([current_features, berry_curvature_tensor], dim=-1)
    
    # Create updated graph
    updated_graph = PyGData(
        x=updated_features,
        edge_index=kspace_graph.edge_index,
        pos=kspace_graph.pos,
        batch=kspace_graph.batch
    )
    
    # Copy other attributes
    for key, value in kspace_graph.items():
        if key not in ['x', 'edge_index', 'pos', 'batch']:
            setattr(updated_graph, key, value)
    
    return updated_graph

def process_material_kspace_graphs(
    kspace_graphs_dir: Path,
    berry_data_dir: Path,
    output_dir: Path,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process k-space graphs for a material and integrate Berry curvature.
    
    Args:
        kspace_graphs_dir: Directory containing k-space graphs
        berry_data_dir: Directory containing Berry curvature data
        output_dir: Output directory for updated graphs
        config: Integration configuration
        
    Returns:
        Processing summary
    """
    # Find k-space graph files
    kspace_graph_files = list(kspace_graphs_dir.glob("*.pt"))
    
    if not kspace_graph_files:
        return {'error': f"No k-space graph files found in {kspace_graphs_dir}"}
    
    # Extract material ID from directory name
    material_id = kspace_graphs_dir.name
    
    # Load Berry curvature data
    berry_data = load_berry_curvature_data(berry_data_dir, material_id)
    
    if berry_data is None:
        return {'error': f"No Berry curvature data found for {material_id}"}
    
    berry_curvature, berry_kpoints = berry_data
    
    # Process each k-space graph
    processed_files = []
    for graph_file in kspace_graph_files:
        try:
            # Load k-space graph
            kspace_graph = torch.load(graph_file)
            
            # Get k-point positions from graph
            target_kpoints = kspace_graph.pos.numpy()
            
            # Interpolate Berry curvature to graph k-points
            interpolated_berry = interpolate_berry_curvature_to_kpoints(
                berry_curvature, 
                berry_kpoints, 
                target_kpoints,
                method='linear'
            )
            
            # Integrate into graph
            updated_graph = integrate_berry_curvature_into_kspace_graph(
                kspace_graph, 
                interpolated_berry,
                config
            )
            
            # Save updated graph
            output_file = output_dir / graph_file.name
            torch.save(updated_graph, output_file)
            
            processed_files.append(graph_file.name)
            
        except Exception as e:
            print(f"Error processing {graph_file}: {e}")
            continue
    
    return {
        'material_id': material_id,
        'total_files': len(kspace_graph_files),
        'processed_files': len(processed_files),
        'files': processed_files
    }

def main():
    parser = argparse.ArgumentParser(description="Integrate Berry curvature into k-space graphs")
    parser.add_argument("--kspace-graphs-dir", required=True, 
                       help="Directory containing k-space graphs")
    parser.add_argument("--berry-data-dir", default="berry_data", 
                       help="Directory containing Berry curvature data")
    parser.add_argument("--output-dir", required=True, 
                       help="Output directory for updated graphs")
    parser.add_argument("--config", default="configs/berry_settings.json", 
                       help="Configuration file")
    parser.add_argument("--material", help="Specific material ID to process")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    kspace_graphs_dir = Path(args.kspace_graphs_dir)
    berry_data_dir = Path(args.berry_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find material directories
    if args.material:
        material_dirs = [kspace_graphs_dir / args.material]
    else:
        material_dirs = [d for d in kspace_graphs_dir.iterdir() if d.is_dir()]
    
    if not material_dirs:
        print(f"No material directories found in {kspace_graphs_dir}")
        return
    
    print(f"Integrating Berry curvature for {len(material_dirs)} materials...")
    
    # Process each material
    processing_summary = []
    for material_dir in material_dirs:
        print(f"\nProcessing {material_dir.name}...")
        
        # Create output directory for this material
        material_output_dir = output_dir / material_dir.name
        material_output_dir.mkdir(exist_ok=True)
        
        # Process k-space graphs
        result = process_material_kspace_graphs(
            material_dir, 
            berry_data_dir, 
            material_output_dir, 
            config
        )
        
        processing_summary.append(result)
        
        if 'error' in result:
            print(f"✗ {result['error']}")
        else:
            print(f"✓ Processed {result['processed_files']}/{result['total_files']} files")
    
    # Save processing summary
    summary_path = output_dir / "integration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(processing_summary, f, indent=2)
    
    # Print final summary
    successful = [r for r in processing_summary if 'error' not in r]
    failed = [r for r in processing_summary if 'error' in r]
    
    print(f"\nIntegration Summary:")
    print(f"  Total materials: {len(processing_summary)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Summary saved to: {summary_path}")
    
    if successful:
        total_files = sum(r['total_files'] for r in successful)
        processed_files = sum(r['processed_files'] for r in successful)
        print(f"  Total files processed: {processed_files}/{total_files}")
    
    print(f"\nUpdated k-space graphs saved to: {output_dir}")
    print("Next steps:")
    print("1. Update your dataset loading code to use the new graphs")
    print("2. Update your model to handle the additional Berry curvature features")
    print("3. Retrain your model with the enhanced features")

if __name__ == "__main__":
    main() 