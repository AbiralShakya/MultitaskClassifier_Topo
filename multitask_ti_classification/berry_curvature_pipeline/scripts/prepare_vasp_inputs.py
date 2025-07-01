#!/usr/bin/env python3
"""
Generate VASP input files for SOC calculations.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np

from pymatgen.io.vasp import Poscar, Incar, Kpoints, Potcar

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def generate_vasp_inputs(material_data_path: Path, config: Dict[str, Any], output_dir: Path):
    """
    Generate VASP input files for a material.
    
    Args:
        material_data_path: Path to material data JSON file
        config: VASP configuration
        output_dir: Output directory for VASP inputs
    """
    # Load material data
    with open(material_data_path, 'r') as f:
        material_data = json.load(f)
    
    mp_id = material_data['mp_id']
    structure = material_data['structure']
    
    # Create material directory
    material_dir = output_dir / mp_id
    material_dir.mkdir(exist_ok=True)
    
    # Generate POSCAR
    poscar = Poscar(structure)
    poscar_path = material_dir / "POSCAR"
    poscar.write_file(str(poscar_path))
    
    # Generate INCAR
    incar_settings = {}
    
    # SOC settings
    soc_settings = config['vasp_settings']['soc_settings']
    incar_settings.update(soc_settings)
    
    # Convergence settings
    conv_settings = config['vasp_settings']['convergence_settings']
    incar_settings.update(conv_settings)
    
    # Output settings
    output_settings = config['vasp_settings']['output_settings']
    incar_settings.update(output_settings)
    
    # System settings
    system_settings = config['vasp_settings']['system_settings']
    incar_settings.update(system_settings)
    
    # Handle auto MAGMOM
    if incar_settings.get('MAGMOM') == 'auto':
        # Generate magnetic moments based on composition
        magmom = []
        for site in structure:
            if site.specie.symbol in ['Fe', 'Co', 'Ni', 'Mn', 'Cr']:
                magmom.append(5.0)  # High-spin transition metals
            else:
                magmom.append(0.0)  # Non-magnetic
        incar_settings['MAGMOM'] = magmom
    
    incar = Incar(incar_settings)
    incar_path = material_dir / "INCAR"
    incar.write_file(str(incar_path))
    
    # Generate KPOINTS
    kpoint_settings = config['vasp_settings']['kpoint_settings']
    kmesh = kpoint_settings['kmesh']
    gamma_centered = kpoint_settings['gamma_centered']
    kpoint_shift = kpoint_settings['kpoint_shift']
    
    kpoints = Kpoints(
        kpts=kmesh,
        kpts_shift=kpoint_shift,
        comment=f"K-mesh for {mp_id} SOC calculation"
    )
    kpoints_path = material_dir / "KPOINTS"
    kpoints.write_file(str(kpoints_path))
    
    # Generate POTCAR
    try:
        potcar = Potcar.from_structure(structure)
        potcar_path = material_dir / "POTCAR"
        potcar.write_file(str(potcar_path))
    except Exception as e:
        print(f"Warning: Could not generate POTCAR for {mp_id}: {e}")
        # Create empty POTCAR file as placeholder
        with open(material_dir / "POTCAR", 'w') as f:
            f.write("# POTCAR generation failed - manual creation required\n")
    
    # Create job submission script
    job_script = f"""#!/bin/bash
#SBATCH --job-name={mp_id}_soc
#SBATCH --output={mp_id}_soc.out
#SBATCH --error={mp_id}_soc.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G

module load vasp

cd {material_dir}
mpirun -np $SLURM_NTASKS vasp_std

echo "VASP calculation completed for {mp_id}"
"""
    
    job_path = material_dir / f"{mp_id}_soc.sh"
    with open(job_path, 'w') as f:
        f.write(job_script)
    
    # Make job script executable
    os.chmod(job_path, 0o755)
    
    print(f"✓ Generated VASP inputs for {mp_id}")
    print(f"  Directory: {material_dir}")
    print(f"  Job script: {job_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate VASP input files")
    parser.add_argument("--data-dir", default="data", help="Directory containing material data")
    parser.add_argument("--output-dir", default="vasp_inputs", help="Output directory for VASP inputs")
    parser.add_argument("--config", default="configs/vasp_settings.json", help="Configuration file")
    parser.add_argument("--material", help="Specific material ID to process")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find material data files
    if args.material:
        material_files = [data_dir / f"{args.material}_data.json"]
    else:
        material_files = list(data_dir.glob("*_data.json"))
    
    if not material_files:
        print(f"No material data files found in {data_dir}")
        return
    
    print(f"Generating VASP inputs for {len(material_files)} materials...")
    
    # Generate inputs for each material
    for material_file in material_files:
        if material_file.exists():
            generate_vasp_inputs(material_file, config, output_dir)
        else:
            print(f"Warning: Material file not found: {material_file}")
    
    print(f"\nVASP inputs generated in: {output_dir}")
    print("Next steps:")
    print("1. Review and modify INCAR files if needed")
    print("2. Ensure POTCAR files are correct")
    print("3. Submit VASP calculations")
    print("4. Run Wannier90 post-processing")

if __name__ == "__main__":
    main() 