#!/bin/bash
# Test Berry Curvature Pipeline with One Material
# This script runs the pipeline with minimal computational cost

set -e

# Configuration for testing
API_KEY="YOUR_MP_API_KEY_HERE"  # Replace with your actual API key
TEST_MATERIAL_LIST="data/test_material.txt"
CONFIG_DIR="configs"
DATA_DIR="data"
VASP_DIR="vasp_inputs"
BERRY_DIR="berry_data"
OUTPUT_DIR="outputs"
LOGS_DIR="logs"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites for test run..."
    
    # Check Python packages
    python -c "import mp_api, pymatgen, numpy, torch" 2>/dev/null || {
        print_error "Missing required Python packages. Install with:"
        print_error "pip install mp-api pymatgen numpy torch torch-geometric"
        exit 1
    }
    
    print_status "Prerequisites check completed."
}

# Function to create test directories
create_test_directories() {
    print_status "Creating test directories..."
    mkdir -p "$DATA_DIR" "$VASP_DIR" "$BERRY_DIR" "$OUTPUT_DIR" "$LOGS_DIR"
}

# Function to fetch test material
fetch_test_material() {
    print_status "Fetching test material from Materials Project API..."
    
    if [ "$API_KEY" = "YOUR_MP_API_KEY_HERE" ]; then
        print_error "Please set your MP API key:"
        print_error "export API_KEY='your_api_key_here'"
        print_error "Or edit this script and set API_KEY variable"
        exit 1
    fi
    
    python scripts/fetch_mp_data.py \
        --api-key "$API_KEY" \
        --material-list "$TEST_MATERIAL_LIST" \
        --output-dir "$DATA_DIR" \
        --config "$CONFIG_DIR/vasp_settings.json" \
        2>&1 | tee "$LOGS_DIR/test_fetch_materials.log"
    
    if [ $? -eq 0 ]; then
        print_status "Test material fetched successfully."
    else
        print_error "Failed to fetch test material. Check logs: $LOGS_DIR/test_fetch_materials.log"
        exit 1
    fi
}

# Function to generate VASP inputs for test
generate_test_vasp_inputs() {
    print_status "Generating VASP inputs for test material..."
    
    python scripts/prepare_vasp_inputs.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$VASP_DIR" \
        --config "$CONFIG_DIR/vasp_settings.json" \
        2>&1 | tee "$LOGS_DIR/test_generate_vasp_inputs.log"
    
    if [ $? -eq 0 ]; then
        print_status "VASP inputs generated successfully."
        print_status "Check VASP inputs in: $VASP_DIR"
    else
        print_error "Failed to generate VASP inputs. Check logs: $LOGS_DIR/test_generate_vasp_inputs.log"
        exit 1
    fi
}

# Function to show next steps
show_next_steps() {
    print_status "Test setup completed successfully!"
    echo ""
    echo "📁 Generated files:"
    echo "  - Material data: $DATA_DIR/mp-149_data.json"
    echo "  - VASP inputs: $VASP_DIR/mp-149/"
    echo "  - Job script: $VASP_DIR/mp-149/mp-149_soc.sh"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Review VASP inputs in $VASP_DIR/mp-149/"
    echo "  2. Run VASP calculation:"
    echo "     cd $VASP_DIR/mp-149"
    echo "     sbatch mp-149_soc.sh  # or run manually"
    echo "  3. After VASP completes, run Wannier90:"
    echo "     wannier90.x -pp mp-149"
    echo "     vasp2wannier90.x"
    echo "     wannier90.x mp-149"
    echo "  4. Compute Berry curvature:"
    echo "     python scripts/compute_berry_curvature.py --material mp-149"
    echo ""
    echo "⏱️  Expected time for Si (mp-149):"
    echo "  - VASP SOC: ~2-4 hours"
    echo "  - Wannier90: ~30 minutes"
    echo "  - Berry curvature: ~10 minutes"
    echo "  - Total: ~3-5 hours"
    echo ""
    echo "💡 For faster testing, you can:"
    echo "  - Use smaller k-point mesh in configs/vasp_settings.json"
    echo "  - Use fewer bands in Wannier90"
    echo "  - Use coarser Berry curvature grid"
}

# Function to create minimal VASP config for testing
create_test_vasp_config() {
    print_status "Creating minimal VASP config for faster testing..."
    
    # Create a minimal VASP config for testing
    cat > "$CONFIG_DIR/vasp_settings_test.json" << EOF
{
  "vasp_settings": {
    "soc_settings": {
      "LSORBIT": true,
      "LORBIT": 14,
      "ICHARG": 11,
      "ISPIN": 2,
      "MAGMOM": "auto"
    },
    "kpoint_settings": {
      "kmesh": [4, 4, 4],
      "gamma_centered": true,
      "kpoint_shift": [0.0, 0.0, 0.0]
    },
    "convergence_settings": {
      "ENCUT": 400,
      "EDIFF": 1e-5,
      "EDIFFG": -0.02,
      "NELM": 100,
      "NELMIN": 4
    },
    "output_settings": {
      "LWAVE": true,
      "LCHARG": true,
      "LORBIT": 14,
      "LVTOT": false,
      "LVHAR": false
    },
    "system_settings": {
      "PREC": "Normal",
      "ALGO": "Normal",
      "LREAL": "Auto",
      "LPLANE": true,
      "NPAR": 1
    }
  },
  "berry_curvature_settings": {
    "kmesh_dense": [20, 20, 20],
    "smearing": 0.1,
    "bands_range": "auto",
    "fermi_energy": "auto"
  }
}
EOF
    
    print_status "Test VASP config created: $CONFIG_DIR/vasp_settings_test.json"
    print_warning "This uses smaller k-point mesh (4x4x4) for faster computation."
}

# Main test execution
main() {
    print_status "Starting Berry Curvature Pipeline Test..."
    print_status "Testing with material: mp-149 (Silicon)"
    
    # Check prerequisites
    check_prerequisites
    
    # Create directories
    create_test_directories
    
    # Create minimal config for testing
    create_test_vasp_config
    
    # Fetch test material
    fetch_test_material
    
    # Generate VASP inputs
    generate_test_vasp_inputs
    
    # Show next steps
    show_next_steps
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Test Berry Curvature Pipeline with one material (mp-149)"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --api-key KEY  Set MP API key"
        echo ""
        echo "Environment variables:"
        echo "  API_KEY        Materials Project API key"
        echo ""
        echo "Example:"
        echo "  export API_KEY='your_api_key_here'"
        echo "  ./test_pipeline.sh"
        exit 0
        ;;
    --api-key)
        API_KEY="$2"
        shift 2
        ;;
    "")
        # No arguments, run test
        main
        ;;
    *)
        print_error "Unknown option: $1"
        print_error "Use --help for usage information"
        exit 1
        ;;
esac 