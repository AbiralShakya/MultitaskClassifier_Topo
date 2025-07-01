#!/usr/bin/env python3
"""
Fetch materials from Materials Project API for Berry curvature computation.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import warnings

try:
    from mp_api.client import MPRester
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install mp-api pymatgen")
    exit(1)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def fetch_material_data(mp_id: str, api_key: str, output_dir: Path) -> Dict[str, Any]:
    """
    Fetch material data from Materials Project API.
    
    Args:
        mp_id: Materials Project ID (e.g., 'mp-149')
        api_key: MP API key
        output_dir: Output directory for material data
        
    Returns:
        Dictionary containing material data
    """
    with MPRester(api_key) as mpr:
        try:
            # Fetch structure
            structure = mpr.get_structure_by_material_id(mp_id)
            
            # Fetch basic properties
            material = mpr.get_structure_by_material_id(mp_id)
            
            # Fetch band structure data if available
            try:
                bandstructure = mpr.get_bandstructure_by_material_id(mp_id)
                has_bandstructure = True
            except:
                bandstructure = None
                has_bandstructure = False
            
            # Fetch DOS data if available
            try:
                dos = mpr.get_dos_by_material_id(mp_id)
                has_dos = True
            except:
                dos = None
                has_dos = False
            
            # Create material data dictionary
            material_data = {
                'mp_id': mp_id,
                'structure': structure,
                'has_bandstructure': has_bandstructure,
                'has_dos': has_dos,
                'spacegroup': structure.get_space_group_info(),
                'lattice': structure.lattice.as_dict(),
                'composition': structure.composition.as_dict()
            }
            
            # Save structure as POSCAR
            poscar = Poscar(structure)
            poscar_path = output_dir / f"{mp_id}_POSCAR"
            poscar.write_file(str(poscar_path))
            
            # Save material data as JSON
            data_path = output_dir / f"{mp_id}_data.json"
            with open(data_path, 'w') as f:
                json.dump(material_data, f, default=str, indent=2)
            
            print(f"✓ Fetched {mp_id}: {structure.composition}")
            return material_data
            
        except Exception as e:
            print(f"✗ Failed to fetch {mp_id}: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Fetch materials from MP API")
    parser.add_argument("--api-key", required=True, help="Materials Project API key")
    parser.add_argument("--material-list", default="data/material_list.txt", 
                       help="File containing list of MP IDs")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--config", default="configs/vasp_settings.json", 
                       help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Read material list
    material_list_path = Path(args.material_list)
    if not material_list_path.exists():
        print(f"Material list file not found: {material_list_path}")
        print("Creating example material list...")
        example_materials = [
            "mp-149",  # Si
            "mp-2534", # Bi2Se3
            "mp-10734", # Bi2Te3
            "mp-1000", # Graphene-like
        ]
        with open(material_list_path, 'w') as f:
            f.write('\n'.join(example_materials))
        print(f"Created example material list: {material_list_path}")
    
    with open(material_list_path, 'r') as f:
        material_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Processing {len(material_ids)} materials...")
    
    # Fetch materials
    successful_materials = []
    for mp_id in material_ids:
        material_data = fetch_material_data(mp_id, args.api_key, output_dir)
        if material_data:
            successful_materials.append(material_data)
    
    # Save summary
    summary = {
        'total_materials': len(material_ids),
        'successful_fetches': len(successful_materials),
        'materials': [m['mp_id'] for m in successful_materials]
    }
    
    summary_path = output_dir / "fetch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary:")
    print(f"  Total materials: {summary['total_materials']}")
    print(f"  Successful fetches: {summary['successful_fetches']}")
    print(f"  Summary saved to: {summary_path}")

if __name__ == "__main__":
    main() 