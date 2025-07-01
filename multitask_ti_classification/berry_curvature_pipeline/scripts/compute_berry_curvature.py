#!/usr/bin/env python3
"""
Compute Berry curvature using PAOFLOW or WannierTools.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def run_paoflow_berry_curvature(material_dir: Path, config: Dict[str, Any], output_dir: Path):
    """
    Compute Berry curvature using PAOFLOW.
    
    Args:
        material_dir: Directory containing VASP and Wannier90 outputs
        config: Berry curvature configuration
        output_dir: Output directory for Berry curvature data
    """
    mp_id = material_dir.name
    
    # Create PAOFLOW input
    paoflow_config = config['paoflow_settings']['berry_curvature']
    
    berry_input = {
        "project_dir": str(material_dir),
        "vasp": {
            "chgc": "CHGCAR",
            "wavc": "WAVECAR"
        },
        "prefix": mp_id,
        "tasks": ["berry"],
        "berry": {
            "ngrid": paoflow_config['ngrid'],
            "smearing": paoflow_config['smearing']
        }
    }
    
    # Save PAOFLOW input
    berry_input_path = material_dir / "berry_input.json"
    with open(berry_input_path, 'w') as f:
        json.dump(berry_input, f, indent=2)
    
    # Run PAOFLOW
    try:
        cmd = ["paoflow", str(berry_input_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=material_dir)
        
        if result.returncode == 0:
            print(f"✓ PAOFLOW Berry curvature computed for {mp_id}")
            
            # Copy results to output directory
            berry_output_dir = output_dir / mp_id
            berry_output_dir.mkdir(exist_ok=True)
            
            # Look for berry curvature output files
            berry_files = list(material_dir.glob("*berry*"))
            for berry_file in berry_files:
                if berry_file.is_file():
                    import shutil
                    shutil.copy2(berry_file, berry_output_dir)
            
            return True
        else:
            print(f"✗ PAOFLOW failed for {mp_id}: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print(f"✗ PAOFLOW not found. Install with: pip install PAOFLOW")
        return False
    except Exception as e:
        print(f"✗ Error running PAOFLOW for {mp_id}: {e}")
        return False

def run_wanniertools_berry_curvature(material_dir: Path, config: Dict[str, Any], output_dir: Path):
    """
    Compute Berry curvature using WannierTools.
    
    Args:
        material_dir: Directory containing Wannier90 outputs
        config: Berry curvature configuration
        output_dir: Output directory for Berry curvature data
    """
    mp_id = material_dir.name
    
    # Create WannierTools input
    wt_config = config['wanniertools_settings']['berry_curvature']
    
    wt_input = f"""&CONTROL
  BerryFlag = T
  BerryPath = {wt_config['berry_path']}
  BerryKmesh = {wt_config['berry_kmesh'][0]} {wt_config['berry_kmesh'][1]} {wt_config['berry_kmesh'][2]}
  OutputFile = {wt_config['output_file']}
/
"""
    
    # Save WannierTools input
    wt_input_path = material_dir / "WT.in"
    with open(wt_input_path, 'w') as f:
        f.write(wt_input)
    
    # Run WannierTools
    try:
        cmd = ["./wannier_tools.x"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=material_dir)
        
        if result.returncode == 0:
            print(f"✓ WannierTools Berry curvature computed for {mp_id}")
            
            # Copy results to output directory
            berry_output_dir = output_dir / mp_id
            berry_output_dir.mkdir(exist_ok=True)
            
            # Look for berry curvature output files
            berry_files = list(material_dir.glob("*berry*"))
            for berry_file in berry_files:
                if berry_file.is_file():
                    import shutil
                    shutil.copy2(berry_file, berry_output_dir)
            
            return True
        else:
            print(f"✗ WannierTools failed for {mp_id}: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print(f"✗ WannierTools not found. Please install WannierTools")
        return False
    except Exception as e:
        print(f"✗ Error running WannierTools for {mp_id}: {e}")
        return False

def process_berry_curvature_data(material_dir: Path, output_dir: Path, config: Dict[str, Any]):
    """
    Process and format Berry curvature data for integration.
    
    Args:
        material_dir: Directory containing Berry curvature data
        output_dir: Output directory for processed data
        config: Integration configuration
    """
    mp_id = material_dir.name
    
    # Look for berry curvature files
    berry_files = list(material_dir.glob("*berry*"))
    
    if not berry_files:
        print(f"Warning: No Berry curvature files found for {mp_id}")
        return False
    
    # Process each berry curvature file
    for berry_file in berry_files:
        if berry_file.suffix == '.json':
            # PAOFLOW JSON output
            with open(berry_file, 'r') as f:
                berry_data = json.load(f)
            
            # Extract k-points and Berry curvature values
            kpoints = np.array(berry_data.get('kpoints', []))
            berry_curvature = np.array(berry_data.get('berry_curvature', []))
            
            # Save as numpy arrays
            output_file = output_dir / f"{mp_id}_berry_curvature.npy"
            np.save(output_file, berry_curvature)
            
            kpoints_file = output_dir / f"{mp_id}_kpoints.npy"
            np.save(kpoints_file, kpoints)
            
        elif berry_file.suffix == '.dat':
            # WannierTools DAT output
            try:
                data = np.loadtxt(berry_file)
                if data.size > 0:
                    # Assume first column is k-point index, others are Berry curvature
                    berry_curvature = data[:, 1:]  # Skip first column
                    
                    output_file = output_dir / f"{mp_id}_berry_curvature.npy"
                    np.save(output_file, berry_curvature)
                    
            except Exception as e:
                print(f"Warning: Could not process {berry_file}: {e}")
    
    print(f"✓ Processed Berry curvature data for {mp_id}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Compute Berry curvature")
    parser.add_argument("--vasp-dir", default="vasp_inputs", help="Directory containing VASP outputs")
    parser.add_argument("--output-dir", default="berry_data", help="Output directory for Berry curvature")
    parser.add_argument("--config", default="configs/berry_settings.json", help="Configuration file")
    parser.add_argument("--method", choices=["paoflow", "wanniertools"], default="paoflow", 
                       help="Method to compute Berry curvature")
    parser.add_argument("--material", help="Specific material ID to process")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    vasp_dir = Path(args.vasp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find material directories
    if args.material:
        material_dirs = [vasp_dir / args.material]
    else:
        material_dirs = [d for d in vasp_dir.iterdir() if d.is_dir()]
    
    if not material_dirs:
        print(f"No material directories found in {vasp_dir}")
        return
    
    print(f"Computing Berry curvature for {len(material_dirs)} materials using {args.method}...")
    
    # Process each material
    successful_materials = []
    for material_dir in material_dirs:
        mp_id = material_dir.name
        print(f"\nProcessing {mp_id}...")
        
        # Check if VASP calculation completed
        vasp_files = ['WAVECAR', 'CHGCAR', 'vasprun.xml']
        if not all((material_dir / f).exists() for f in vasp_files):
            print(f"Warning: VASP files not found for {mp_id}")
            continue
        
        # Check if Wannier90 completed
        wannier_files = ['wannier90.amn', 'wannier90.mmn', 'wannier90.eig']
        if not all((material_dir / f).exists() for f in wannier_files):
            print(f"Warning: Wannier90 files not found for {mp_id}")
            continue
        
        # Compute Berry curvature
        if args.method == "paoflow":
            success = run_paoflow_berry_curvature(material_dir, config, output_dir)
        else:
            success = run_wanniertools_berry_curvature(material_dir, config, output_dir)
        
        if success:
            # Process the data
            process_berry_curvature_data(material_dir, output_dir, config)
            successful_materials.append(mp_id)
    
    # Save summary
    summary = {
        'total_materials': len(material_dirs),
        'successful_computations': len(successful_materials),
        'method': args.method,
        'materials': successful_materials
    }
    
    summary_path = output_dir / "berry_computation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary:")
    print(f"  Total materials: {summary['total_materials']}")
    print(f"  Successful computations: {summary['successful_computations']}")
    print(f"  Method: {summary['method']}")
    print(f"  Summary saved to: {summary_path}")

if __name__ == "__main__":
    main() 