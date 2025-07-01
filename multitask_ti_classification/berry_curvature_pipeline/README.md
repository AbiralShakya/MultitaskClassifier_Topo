# Berry Curvature Computation Pipeline

This folder contains the workflow for computing Berry curvature data for materials, which can be integrated into the main topological insulator classification project.

## Overview

The pipeline follows this workflow:
1. **Materials Project API** → Fetch structure and VASP inputs
2. **VASP with SOC** → Generate WAVECAR/CHGCAR files
3. **Wannier90** → Create maximally localized Wannier functions
4. **PAOFLOW/WannierTools** → Compute Berry curvature on dense k-grid
5. **Integration** → Add Berry curvature to k-space graphs

## Folder Structure

```
berry_curvature_pipeline/
├── README.md                 # This file
├── scripts/                  # Python scripts for automation
│   ├── fetch_mp_data.py     # Fetch materials from MP API
│   ├── prepare_vasp_inputs.py # Generate VASP input files
│   ├── run_wannier90.py     # Wannier90 automation
│   ├── compute_berry_curvature.py # PAOFLOW/WannierTools automation
│   └── integrate_berry_data.py # Add Berry curvature to k-space graphs
├── configs/                  # Configuration files
│   ├── vasp_settings.json   # VASP parameters
│   ├── wannier90_settings.json # Wannier90 parameters
│   └── berry_settings.json  # Berry curvature computation settings
├── data/                     # Input/output data
│   ├── material_list.txt    # List of materials to process
│   └── berry_curvature/     # Computed Berry curvature data
├── vasp_inputs/             # Generated VASP input files
├── wannier90_inputs/        # Generated Wannier90 input files
├── paoflow_inputs/          # PAOFLOW configuration files
├── berry_data/              # Final Berry curvature data
├── outputs/                 # Processing outputs
└── logs/                    # Processing logs
```

## Prerequisites

### Software Requirements
- VASP (with SOC support)
- Wannier90
- PAOFLOW (`pip install PAOFLOW`) or WannierTools
- Python packages: `pymatgen`, `mp-api`, `numpy`, `torch`

### API Access
- Materials Project API key

## Usage

### 1. Setup Configuration
```bash
# Copy and edit configuration files
cp configs/vasp_settings.json.example configs/vasp_settings.json
cp configs/wannier90_settings.json.example configs/wannier90_settings.json
cp configs/berry_settings.json.example configs/berry_settings.json
```

### 2. Prepare Material List
```bash
# Edit the list of materials to process
nano data/material_list.txt
```

### 3. Run the Pipeline
```bash
# Fetch materials from MP API
python scripts/fetch_mp_data.py

# Generate VASP inputs
python scripts/prepare_vasp_inputs.py

# Run VASP calculations (manual or automated)
# ... VASP runs ...

# Run Wannier90
python scripts/run_wannier90.py

# Compute Berry curvature
python scripts/compute_berry_curvature.py

# Integrate with main project
python scripts/integrate_berry_data.py
```

## Integration with Main Project

The computed Berry curvature data can be integrated into the main classification project by:

1. **Adding as node features** in k-space graphs
2. **Using as auxiliary loss targets** for regularization
3. **Providing physics-aware features** for better classification

## Output Format

Berry curvature data is stored as:
- **Per material**: `berry_data/mp-{material_id}/berry_curvature.npy`
- **Shape**: `[num_kpoints, num_bands]` or `[num_kpoints, 1]` (summed over bands)
- **Integration**: Can be concatenated with existing k-space graph node features

## Notes

- This pipeline requires significant computational resources
- VASP calculations with SOC can be time-consuming
- Consider running on a cluster for large material sets
- Backup intermediate results (WAVECAR, CHGCAR, Wannier90 files) 