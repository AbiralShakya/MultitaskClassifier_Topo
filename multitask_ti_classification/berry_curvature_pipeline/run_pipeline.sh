#!/bin/bash
# Berry Curvature Computation Pipeline
# This script runs the complete pipeline from MP API to integrated k-space graphs

set -e  # Exit on any error

# Configuration
API_KEY="YOUR_MP_API_KEY_HERE"  # Replace with your actual API key
MATERIAL_LIST="data/material_list.txt"
CONFIG_DIR="configs"
DATA_DIR="data"
VASP_DIR="vasp_inputs"
BERRY_DIR="berry_data"
OUTPUT_DIR="outputs"
LOGS_DIR="logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python packages
    python -c "import mp_api, pymatgen, numpy, torch" 2>/dev/null || {
        print_error "Missing required Python packages. Install with:"
        print_error "pip install mp-api pymatgen numpy torch torch-geometric"
        exit 1
    }
    
    # Check VASP (optional, for local runs)
    if command_exists vasp_std; then
        print_status "VASP found: $(which vasp_std)"
    else
        print_warning "VASP not found in PATH. You'll need to run VASP calculations manually."
    fi
    
    # Check Wannier90 (optional)
    if command_exists wannier90.x; then
        print_status "Wannier90 found: $(which wannier90.x)"
    else
        print_warning "Wannier90 not found in PATH. You'll need to run Wannier90 manually."
    fi
    
    # Check PAOFLOW (optional)
    if command_exists paoflow; then
        print_status "PAOFLOW found: $(which paoflow)"
    else
        print_warning "PAOFLOW not found. Install with: pip install PAOFLOW"
    fi
    
    print_status "Prerequisites check completed."
}

# Function to create directories
create_directories() {
    print_status "Creating directories..."
    mkdir -p "$DATA_DIR" "$VASP_DIR" "$BERRY_DIR" "$OUTPUT_DIR" "$LOGS_DIR"
}

# Function to fetch materials from MP API
fetch_materials() {
    print_status "Fetching materials from Materials Project API..."
    
    if [ "$API_KEY" = "YOUR_MP_API_KEY_HERE" ]; then
        print_error "Please set your MP API key in the script or provide it as an argument."
        print_error "Get your API key from: https://materialsproject.org/api"
        exit 1
    fi
    
    python scripts/fetch_mp_data.py \
        --api-key "$API_KEY" \
        --material-list "$MATERIAL_LIST" \
        --output-dir "$DATA_DIR" \
        --config "$CONFIG_DIR/vasp_settings.json" \
        2>&1 | tee "$LOGS_DIR/fetch_materials.log"
    
    if [ $? -eq 0 ]; then
        print_status "Materials fetched successfully."
    else
        print_error "Failed to fetch materials. Check logs: $LOGS_DIR/fetch_materials.log"
        exit 1
    fi
}

# Function to generate VASP inputs
generate_vasp_inputs() {
    print_status "Generating VASP input files..."
    
    python scripts/prepare_vasp_inputs.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$VASP_DIR" \
        --config "$CONFIG_DIR/vasp_settings.json" \
        2>&1 | tee "$LOGS_DIR/generate_vasp_inputs.log"
    
    if [ $? -eq 0 ]; then
        print_status "VASP inputs generated successfully."
    else
        print_error "Failed to generate VASP inputs. Check logs: $LOGS_DIR/generate_vasp_inputs.log"
        exit 1
    fi
}

# Function to run VASP calculations (placeholder)
run_vasp_calculations() {
    print_warning "VASP calculations need to be run manually or via job scheduler."
    print_warning "Generated inputs are in: $VASP_DIR"
    print_warning "After VASP completes, run Wannier90 post-processing."
    
    # Example of how to submit jobs (uncomment and modify as needed)
    # for material_dir in "$VASP_DIR"/*/; do
    #     if [ -d "$material_dir" ]; then
    #         material=$(basename "$material_dir")
    #         print_status "Submitting VASP job for $material..."
    #         cd "$material_dir"
    #         sbatch "${material}_soc.sh"
    #         cd - > /dev/null
    #     fi
    # done
}

# Function to run Wannier90 (placeholder)
run_wannier90() {
    print_warning "Wannier90 post-processing needs to be run manually."
    print_warning "For each material, run:"
    print_warning "  wannier90.x -pp <material>"
    print_warning "  vasp2wannier90.x"
    print_warning "  wannier90.x <material>"
}

# Function to compute Berry curvature
compute_berry_curvature() {
    print_status "Computing Berry curvature..."
    
    python scripts/compute_berry_curvature.py \
        --vasp-dir "$VASP_DIR" \
        --output-dir "$BERRY_DIR" \
        --config "$CONFIG_DIR/berry_settings.json" \
        --method paoflow \
        2>&1 | tee "$LOGS_DIR/compute_berry_curvature.log"
    
    if [ $? -eq 0 ]; then
        print_status "Berry curvature computed successfully."
    else
        print_error "Failed to compute Berry curvature. Check logs: $LOGS_DIR/compute_berry_curvature.log"
        exit 1
    fi
}

# Function to integrate Berry curvature into k-space graphs
integrate_berry_data() {
    print_status "Integrating Berry curvature into k-space graphs..."
    
    # This assumes you have k-space graphs in a specific directory
    # Modify the path according to your project structure
    KSPACE_GRAPHS_DIR="../kspace_topology_graphs"  # Adjust this path
    
    if [ ! -d "$KSPACE_GRAPHS_DIR" ]; then
        print_warning "K-space graphs directory not found: $KSPACE_GRAPHS_DIR"
        print_warning "Skipping integration step."
        return
    fi
    
    python scripts/integrate_berry_data.py \
        --kspace-graphs-dir "$KSPACE_GRAPHS_DIR" \
        --berry-data-dir "$BERRY_DIR" \
        --output-dir "$OUTPUT_DIR/enhanced_kspace_graphs" \
        --config "$CONFIG_DIR/berry_settings.json" \
        2>&1 | tee "$LOGS_DIR/integrate_berry_data.log"
    
    if [ $? -eq 0 ]; then
        print_status "Berry curvature integrated successfully."
    else
        print_error "Failed to integrate Berry curvature. Check logs: $LOGS_DIR/integrate_berry_data.log"
        exit 1
    fi
}

# Function to generate summary report
generate_summary() {
    print_status "Generating summary report..."
    
    cat > "$OUTPUT_DIR/pipeline_summary.txt" << EOF
Berry Curvature Pipeline Summary
================================

Date: $(date)
Pipeline completed: $(date)

Directories:
- Data: $DATA_DIR
- VASP inputs: $VASP_DIR
- Berry curvature: $BERRY_DIR
- Output: $OUTPUT_DIR
- Logs: $LOGS_DIR

Log files:
$(ls -la "$LOGS_DIR"/*.log 2>/dev/null || echo "No log files found")

Next steps:
1. Review VASP inputs in $VASP_DIR
2. Run VASP calculations with SOC
3. Run Wannier90 post-processing
4. Check Berry curvature results in $BERRY_DIR
5. Integrate with main classification project

EOF
    
    print_status "Summary report generated: $OUTPUT_DIR/pipeline_summary.txt"
}

# Main execution
main() {
    print_status "Starting Berry Curvature Pipeline..."
    
    # Check prerequisites
    check_prerequisites
    
    # Create directories
    create_directories
    
    # Run pipeline steps
    fetch_materials
    generate_vasp_inputs
    run_vasp_calculations
    run_wannier90
    compute_berry_curvature
    integrate_berry_data
    
    # Generate summary
    generate_summary
    
    print_status "Pipeline completed successfully!"
    print_status "Check $OUTPUT_DIR/pipeline_summary.txt for details."
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --api-key KEY  Set MP API key"
        echo "  --fetch-only   Only fetch materials from MP API"
        echo "  --vasp-only    Only generate VASP inputs"
        echo "  --berry-only   Only compute Berry curvature"
        echo ""
        echo "Environment variables:"
        echo "  API_KEY        Materials Project API key"
        exit 0
        ;;
    --api-key)
        API_KEY="$2"
        shift 2
        ;;
    --fetch-only)
        check_prerequisites
        create_directories
        fetch_materials
        exit 0
        ;;
    --vasp-only)
        check_prerequisites
        create_directories
        generate_vasp_inputs
        exit 0
        ;;
    --berry-only)
        check_prerequisites
        create_directories
        compute_berry_curvature
        exit 0
        ;;
    "")
        # No arguments, run full pipeline
        main
        ;;
    *)
        print_error "Unknown option: $1"
        print_error "Use --help for usage information"
        exit 1
        ;;
esac 